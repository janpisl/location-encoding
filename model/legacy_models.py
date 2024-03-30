from torch import nn
import torch.nn.functional as F
from torchvision.models import resnet18, resnet34,resnet50
import pdb
import torch.nn as nn
import torch.nn.functional as F
import torch
import copy
from typing import List
import torch
import torchvision.models as models

from model.model_blocks import Identity, get_resnet


class CNN_Att_LSTM(nn.Module):
    def __init__(self, output_classes=None, return_att_scores=False, feat_extractor_only=False, **kwargs):
        super(CNN_Att_LSTM, self).__init__()

        self.output_classes = output_classes
        if not output_classes:
            assert feat_extractor_only
    
        self.feat_extractor = AttentionFeatureExtractor(**kwargs)
        self.feat_extractor_only = feat_extractor_only
        if not self.feat_extractor_only:
            self.classifier = Classifier(self.feat_extractor.lstm_hidden_size, self.output_classes)

        self.return_att_scores = return_att_scores
        
     
    def forward(self, x, forestloss_patch=None):
        
        feat_extractor_outputs = self.feat_extractor(x, forestloss_patch)

        if self.feat_extractor_only:
            output = feat_extractor_outputs['features']
        else:
            output = self.classifier(feat_extractor_outputs['features'])

        if self.return_att_scores:
            if self.feat_extractor_only:
                raise NotImplementedError() 
            else:
                # Keep the contents from the feature extractor (may include att_scores) except for features which we don't want to keep
                output_dict = {k: v for k, v in feat_extractor_outputs.items() if k != 'features'}
                output_dict['class_logits'] = output
                return output_dict
        else:
            return output


class Classifier(nn.Module):
    def __init__(self, input_size, output_size, n_layers=2):
        super(Classifier, self).__init__()
        #For each layer except the last one, add linear layer and ReLU
        layers = [
            nn.Sequential(*[nn.Linear(input_size, out_features=input_size, bias = True), nn.ReLU(True)])
            for i in range(n_layers - 1)]
        #layers = [item for sublist in layers for item in sublist] #flatten the nested list

        layers.append(nn.Linear(input_size, out_features=output_size, bias = True)) # This will always be there                                                  
        self.mlp = nn.Sequential(*layers)

    def forward(self, x):
        return self.mlp(x)


class FeatureExtractor(nn.Module):
    def __init__(self, **kwargs):
        super(FeatureExtractor, self).__init__()

        try:
            self.resnet_type = kwargs['hyperparams']['image_encoder']['backbone']
        except KeyError:
            self.resnet_type = kwargs['hyperparams']['backbone']

        self.backbone = get_resnet(resnet_type=self.resnet_type, **kwargs)


    def forward(self, x, forestloss):
        pass


class TemporalFeatureExtractor(FeatureExtractor):
    def __init__(self, **kwargs):
        super(TemporalFeatureExtractor, self).__init__(**kwargs)

        self.lstm_hidden_size = kwargs['hyperparams']['embedding_dim']

        self.backbone = get_resnet(resnet_type=self.resnet_type, output_size=self.lstm_hidden_size, **kwargs)

        lstm_layers = 1

        self.lstm_bidirectional = True

        self.lstm = nn.LSTM(
            input_size=self.lstm_hidden_size,
            hidden_size=self.lstm_hidden_size, 
            num_layers=lstm_layers, 
            bidirectional=self.lstm_bidirectional)
        
        


    def _get_spatial_features_from_ts(self, x, forestloss, model=None):
        """_summary_

        Args:
            x (input time series): shape BxTxCxWxH
            model (optional)

        Returns:
            _type_: _description_
        """
        if not model:
            model = self.backbone

        if len(x.shape) == 4: #If single-temporal (ie. no T dimenison), add the T axis
            x = x.unsqueeze(1)
        model_outputs  = []
        #Length of temporal series
        temporal_steps = x.shape[1] 
        
        #TODO: I can reshape x to move the temporal dimension to batch to remove the for loop and speed it up
        #I tried the below but that doesn't work
        #flattened_x = torch.reshape(x, (x.shape[0]*x.shape[1], x.shape[2], x.shape[3], x.shape[4]))
        #y = model(flattened_x)
        #model_outputs = torch.reshape(y, (x.shape[1],x.shape[0], -1))

        for i in range(temporal_steps):
            if forestloss is not None:
                model_outputs.append(model(x[:,i], forestloss))
            else:
                model_outputs.append(model(x[:,i]))
        #Stack along new axis
        model_outputs = torch.stack(model_outputs, dim=0)
        
        return model_outputs


    def forward(self, x):
        """
        x has a shape of BxTxCxWxH
            where
                B --- batch
                T --- temporal dimension
                C --- channels
                W, H --- width, height
        
        """
        spatial_feature_vectors = self._get_spatial_features_from_ts(x)

        all_hidden_states, (hn, cn) = self.lstm(spatial_feature_vectors)

        lstm_output = hn

        if self.lstm_bidirectional:
            lstm_output = torch.cat((lstm_output[-2], lstm_output[-1]),dim=1) #If n_layers > 1, use the outputs of the LAST layer
        else:
            lstm_output =  lstm_output[-1] 

        
        output = {}
        output['features'] = lstm_output

        return output


class AttentionFeatureExtractor(TemporalFeatureExtractor):
    def __init__(self, **kwargs):
        super(AttentionFeatureExtractor, self).__init__(**kwargs)

        #assert kwargs['feature_extractor']['lstm_bidirectional'], "AttentionFeatureExtractor must have bidirectional LSTM"
        #self.attend_to_images = kwargs['feature_extractor']['attend_to_images']

        try:
            self.return_att_scores = kwargs['feature_extractor']['return_att_scores']
        except KeyError:
            self.return_att_scores = False


    def _get_temporal_features(self, feature_maps, lstm=None):
        
        if lstm:
            all_hidden_states, (hn, cn) = lstm(feature_maps)
        else:
            all_hidden_states, (hn, cn) = self.lstm(feature_maps)

        cn = torch.cat((cn[-2], cn[-1]),dim=1) #If n_layers > 1, use the outputs of the LAST layer

        return all_hidden_states, cn


    def _apply_attention(self, hidden_states, query, feature_maps=None):

        #query is Bx1024, all_hidden_states is TxBx1024
        attention_scores = hidden_states.transpose(0,1).matmul(query.unsqueeze(2)).squeeze(-1)

        #attention_scores.shape should be BxT
        attention_scores = F.softmax(attention_scores, dim=1)

        assert feature_maps is not None
        spatiotemporal_features = feature_maps.transpose(1,2).transpose(0,2).matmul(attention_scores.unsqueeze(2)).squeeze(2)

        return spatiotemporal_features, attention_scores


    def forward(self, x, forestloss):
        """
        x has a shape of BxTxCxWxH
            where
                B --- batch
                T --- temporal dimension
                C --- channels
                W, H --- width, height
        
        """
        backbone_outputs = self._get_spatial_features_from_ts(x, forestloss)

        all_hidden_states, cn = self._get_temporal_features(backbone_outputs)
        
        attended_features, attention_scores =  self._apply_attention(all_hidden_states, cn, backbone_outputs)

        output = {}
        output['features'] = attended_features
        if self.return_att_scores:
            output.update({'att_scores': attention_scores})

        return output









    



