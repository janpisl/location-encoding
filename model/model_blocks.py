from torch import nn
import torch.nn.functional as F
from torchvision.models import resnet18, resnet34,resnet50
import pdb
import torch.nn as nn
import torch.nn.functional as F
import torch
import copy
from typing import List



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

        self.backbone = get_resnet(resnet_type=kwargs['hyperparams']['backbone'], **kwargs)
        self.backbone_output_size = self.backbone.fc.in_features
        #Remove the last fully connected layer so that the output of resnet is a vector of size 512 (or 2048 for resnet50)
        self.backbone.fc = Identity()


    def forward(self, x):
        
        output = {}
        output['features'] = self.backbone(x)

        return output


class TemporalFeatureExtractor(FeatureExtractor):
    def __init__(self, **kwargs):
        super(TemporalFeatureExtractor, self).__init__(**kwargs)

        self.lstm_hidden_size = 512

        lstm_layers = 1

        self.lstm_bidirectional = True

        self.lstm = nn.LSTM(
            input_size=self.backbone_output_size, 
            hidden_size=self.lstm_hidden_size, 
            num_layers=lstm_layers, 
            bidirectional=self.lstm_bidirectional)


    def _get_spatial_features_from_ts(self, x, model=None):
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
            self.return_att_scores = True


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


    def forward(self, x):
        """
        x has a shape of BxTxCxWxH
            where
                B --- batch
                T --- temporal dimension
                C --- channels
                W, H --- width, height
        
        """
        
        backbone_outputs = self._get_spatial_features_from_ts(x)

        all_hidden_states, cn = self._get_temporal_features(backbone_outputs)
        
        attended_features, attention_scores =  self._apply_attention(all_hidden_states, cn, backbone_outputs)

        output = {}
        output['features'] = attended_features
        if self.return_att_scores:
            output.update({'att_scores': attention_scores})

        return output




def get_resnet(resnet_type, **kwargs):

    pretrained = kwargs['hyperparams']['pretrained_backbone']

    if resnet_type == 'resnet18':
        net = resnet18(weights="DEFAULT" if pretrained else None)
    elif resnet_type == 'resnet34':
        net = resnet34(weights="DEFAULT" if pretrained else None)
    elif resnet_type == 'resnet50':
        net = resnet50(weights="DEFAULT" if pretrained else None)
    else:
        raise ValueError("Unknown resnet type")

    #TODO: test if it's better to spread the RGB weights as opposed to just init the first layer from scratch

    if pretrained: # Create a convlayer and spread the RGB pretrained weights to as many channels as there in the inputs
        net = set_up_pretrained_resnet(net, in_channels=kwargs['hyperparams']['in_channels'])  
    else: #Create a convlayer and initialize its weights
        net.conv1 = nn.Conv2d(kwargs['hyperparams']['in_channels'], 
                            net.conv1.out_channels, 
                            kernel_size=net.conv1.kernel_size, 
                            stride=net.conv1.stride, 
                            padding=net.conv1.padding,
                            bias=False)                
        nn.init.kaiming_normal_(net.conv1.weight, mode="fan_out", nonlinearity="relu")
    

    net.fc = nn.Linear(net.fc.in_features, out_features=10, bias = True) # The number of output classes doesn't matter, it will get re-initialized

    return net


def set_up_pretrained_resnet(net, in_channels):
    new_weights = net.conv1.weight
    
    #repeats = in_channels // net.conv1.weight.shape[1]
    #conv1_weights = conv1_weights.repeat(1,repeats,1,1)

    while new_weights.shape[1] < in_channels: #Keep adding channels in the same order until number of in_channels matches
        new_weights = torch.cat([new_weights, new_weights[:,new_weights.shape[1]-3,:,:].unsqueeze(1)], axis=1)

    net.conv1 = nn.Conv2d(in_channels, 
                        net.conv1.out_channels, 
                        kernel_size=net.conv1.kernel_size, 
                        stride=net.conv1.stride, 
                        padding=net.conv1.padding,
                        bias=False)

    net.conv1.weight = nn.Parameter(new_weights)

    return net



#Just a dummy model that does nothing
class Identity(nn.Module):
    def __init__(self):
        super(Identity, self).__init__()
        
    def forward(self, x):
        return x