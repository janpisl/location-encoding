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
import math


class TransformerClassifier(nn.Module):
    def __init__(self, output_classes, **config):
        super(TransformerClassifier, self).__init__()

        embedding_dim = config['hyperparams']['embedding_dim']

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embedding_dim, 
            nhead=config['hyperparams']['classifier']['attention_heads'])
        self.transformer_enc = nn.TransformerEncoder(
            encoder_layer, 
            num_layers=config['hyperparams']['classifier']['transformer_layers'],
            enable_nested_tensor=False #Get rid of annoying warnings
            )

        self.cls_token = nn.Parameter(torch.randn(embedding_dim))

        self.classifier = nn.Linear(embedding_dim, out_features=output_classes, bias = True)

        
    def forward(self, tokens):
        """_summary_

        Args:
            tokens (list): set of tokens, each item in list is either a single token of shape BxD (batch, embedding_dim),
                            or a group of tokens (shape NxBxD)

        """
        #Get all tokens in shape NxBxD and concat them together into a tensor of NxBxD (num_tokens, batch, embed_dim)
        #tokens = torch.cat([token if len(token.shape) == 3 else token.unsqueeze(dim=0) for token in tokens], dim=0)

        #Make cls_token tensor in shape (1,B,D)
        cls_tokens = self.cls_token.repeat(tokens.shape[1],1).unsqueeze(dim=0)

        features = torch.cat([cls_tokens, tokens])

        cls_token_output = self.transformer_enc(features)[0]
        predictions = self.classifier(cls_token_output) 
        
        return predictions

    

class TransformerClassifierV2(TransformerClassifier):
    def __init__(self, output_classes, token_size=10, **config):
        super(TransformerClassifierV2, self).__init__(output_classes, **config)

        self.output_classes = output_classes
        self.classifier = None        
        self.token_size = token_size
        self.fc_layers = nn.ModuleList([nn.Linear(token_size, 1) for _ in range(output_classes)])

        
    def forward(self, tokens):
        """_summary_

        Args:
            tokens (list): set of tokens, each item in list is either a single token of shape BxD (batch, embedding_dim),
                            or a group of tokens (shape NxBxD)

        """

        #Get all tokens in shape NxBxD and concat them together into a tensor of NxBxD (num_tokens, batch, embed_dim)
        tokens = torch.cat([token if len(token.shape) == 3 else token.unsqueeze(dim=0) for token in tokens], dim=0)

        #The last token is the TAB token - given by the order in which they are appended in model.forward
        tab_token_output = self.transformer_enc(tokens)[-1]
        
        batch = tab_token_output.shape[0]
        #tab_token_output is of shape (BxD), ie 64x120

        predictions = torch.full((batch, self.output_classes), float('nan'), device='cuda')
        for i in range(self.output_classes):
            predictions[:, i] = self.fc_layers[i](tab_token_output[:, i*self.token_size:(i+1)*self.token_size ]).squeeze()
                
        return predictions



class TransformerClassifierV3_____(TransformerClassifier):
    def __init__(self, output_classes, token_size=10, **config):
        super(TransformerClassifierV3, self).__init__(output_classes, **config)

        self.output_classes = output_classes
        self.token_size = token_size
        self.embedding_dim = config['hyperparams']['embedding_dim']
        self.classifier = nn.Linear(self.embedding_dim+token_size, out_features=1, bias = True)

        
    def forward(self, tokens):
        """_summary_

        Args:
            tokens (list): set of tokens, each item in list is either a single token of shape BxD (batch, embedding_dim),
                            or a group of tokens (shape NxBxD)

        """
        #Get all tokens in shape NxBxD and concat them together into a tensor of NxBxD (num_tokens, batch, embed_dim)
        tokens = torch.cat([token if len(token.shape) == 3 else token.unsqueeze(dim=0) for token in tokens], dim=0)

        #Make cls_token tensor in shape (1,B,D)
        cls_tokens = self.cls_token.repeat(tokens.shape[1],1).unsqueeze(dim=0)

        features = torch.cat([cls_tokens, tokens])
        encoded_tokens = self.transformer_enc(features)
        #The last token is the TAB token - given by the order in which they are appended in model.forward
        cls_token_output, tab_token_output = encoded_tokens[0], encoded_tokens[-1]
        
        batch = tab_token_output.shape[0]

        predictions = torch.full((batch, self.output_classes), float('nan'), device='cuda')

        for i in range(self.output_classes):
            #Output token created by concat'ing the cls token with the section of the tab token for given class
            output_token = torch.cat([cls_token_output, tab_token_output[:, i*self.token_size:(i+1)*self.token_size ]], dim=1)
            predictions[:, i] = self.classifier(output_token).squeeze()
                            
        return predictions
    


class TransformerClassifierV3(TransformerClassifier):
    def __init__(self, output_classes, token_size=10, **config):
        super(TransformerClassifierV3, self).__init__(output_classes, **config)

        self.output_classes = output_classes
        self.token_size = token_size
        self.embedding_dim = config['hyperparams']['embedding_dim']
        self.fc_layers = nn.ModuleList([nn.Linear(self.embedding_dim+token_size, 1) for _ in range(output_classes)])

        
    def forward(self, tokens):
        """_summary_

        Args:
            tokens (list): set of tokens, each item in list is either a single token of shape BxD (batch, embedding_dim),
                            or a group of tokens (shape NxBxD)

        """
        #Get all tokens in shape NxBxD and concat them together into a tensor of NxBxD (num_tokens, batch, embed_dim)
        tokens = torch.cat([token if len(token.shape) == 3 else token.unsqueeze(dim=0) for token in tokens], dim=0)

        #Make cls_token tensor in shape (1,B,D)
        cls_tokens = self.cls_token.repeat(tokens.shape[1],1).unsqueeze(dim=0)

        features = torch.cat([cls_tokens, tokens])
        encoded_tokens = self.transformer_enc(features)
        #The last token is the TAB token - given by the order in which they are appended in model.forward
        cls_token_output, tab_token_output = encoded_tokens[0], encoded_tokens[-1]
        
        batch = tab_token_output.shape[0]

        predictions = torch.full((batch, self.output_classes), float('nan'), device='cuda')

        for i in range(self.output_classes):
            #Output token created by concat'ing the cls token with the section of the tab token for given class
            output_token = torch.cat([cls_token_output, tab_token_output[:, i*self.token_size:(i+1)*self.token_size ]], dim=1)
            predictions[:, i] = self.fc_layers[i](output_token).squeeze()
                            
        return predictions







class TabularEncoder(nn.Module):
    def __init__(self, n_classes=12, out_dim=120):
        super(TabularEncoder, self).__init__()
        
        #assert n_classes*token_size == out_dim #Doesn't have to be true, but then some parameters are wasted

        self.token_size = math.floor(out_dim/n_classes)

        self.padding = out_dim % n_classes

        self.missing_tab_value_token = nn.Parameter(torch.zeros(self.token_size))
        torch.nn.init.normal_(self.missing_tab_value_token, std=1e-5)

        self.n_classes = n_classes
        self.token_size = self.token_size

        self.out_dim = out_dim

        self.fc_layers = nn.ModuleList([nn.Linear(1, self.token_size) for _ in range(n_classes)])


    def forward(self, x):
        outputs = []

        for i in range(self.n_classes):
            current_class_tab_values = x[:, i].unsqueeze(1)  # Select the i-th class for all batch items

            # Create a mask for NaN values
            mask = torch.isnan(current_class_tab_values)

            #encoded_cls_tab_values = self.fc_layers[i](current_class_tab_values)
            
            #Only feeding non-NaN values to the fc layers
            encoded_cls_tab_values_non_nan = self.fc_layers[i](current_class_tab_values[~mask].unsqueeze(1))

            # Create 10 values of the output token; for each item in batch, it's either 
            # the missing_tab_value_token  (if no tab data provided) 
            # or the encoded tab values, which are written in the next step
            encoded_cls_tab_values_no_data_token = torch.where(mask, self.missing_tab_value_token, float('nan'))

            encoded_cls_tab_values_no_data_token[~mask.squeeze()] = encoded_cls_tab_values_non_nan

            outputs.append(encoded_cls_tab_values_no_data_token)

        result = torch.cat(outputs, dim=1)
        
        if self.padding > 0:
            pad = (1, self.padding-1)
            result = F.pad(result, pad, "constant", 0)

        return result






class TemporalWrapper(nn.Module):
    def __init__(self, model):
        super(TemporalWrapper, self).__init__()

        self.model = model
        
    def forward(self, x, forestloss=None):
        """_summary_

        Args:
            x (input time series): shape BxTxCxWxH
            model (optional)

        Returns:
            _type_: _description_
        """

        if len(x.shape) == 4: #If single-temporal (ie. no T dimenison), add the T axis
            x = x.unsqueeze(1)
        
        model_outputs  = []
        #Length of temporal series
        temporal_steps = x.shape[1] 

        for i in range(temporal_steps):
            if forestloss is not None:
                model_outputs.append(self.model(x[:,i], forestloss))
            else:
                model_outputs.append(self.model(x[:,i]))
        #Stack along new axis
        model_outputs = torch.stack(model_outputs, dim=0)
        
        return model_outputs


class FullyConnectedClassifier(nn.Module):
    def __init__(self, output_classes, input_dimension, **config):
        super(FullyConnectedClassifier, self).__init__()

        self.classifier = nn.Linear(input_dimension, out_features=output_classes, bias = True)

        
    def forward(self, tokens):
        """_summary_

        Args:
            tokens (list): set of tokens, each item in list is either a single token of shape BxD (batch x any_dimension),
                            or a group of tokens (shape NxBxD)
        """

        '''
        #TODO: Fix this, i don't know how to make it work with view or reshape 
        n_tokens, batch_size, embedding_dim = tokens.shape
        tokens.view(batch_size, embedding_dim*n_tokens)
        '''
        #TODO: this is pretty stupid but i can't figure out the normal way to do it atm
        processed_tokens = []
        for token in tokens:
            if len(token.shape) == 1:
                processed_tokens.append(token.unsqueeze(1))
            if len(token.shape) == 2:
                processed_tokens.append(token)
            elif len(token.shape) == 3:
                token = [i for i in token]
                for i in token:
                    processed_tokens.append(i)
        
        tokens = torch.cat(processed_tokens, dim=1)

        return self.classifier(tokens)



def get_resnet(resnet_type, output_size=None, **kwargs):

    pretrained = kwargs['hyperparams']['image_encoder']['pretrained_backbone']

    if resnet_type == 'resnet18':
        net = resnet18(weights="DEFAULT" if pretrained else None)
    elif resnet_type == 'resnet34':
        net = resnet34(weights="DEFAULT" if pretrained else None)
    elif resnet_type == 'resnet50':
        net = resnet50(weights="DEFAULT" if pretrained else None)
    else:
        raise ValueError("Unknown resnet type")


    if kwargs['hyperparams']['use_forestloss']:
        assert resnet_type == 'resnet18', 'Other types of resnet not supported atm'
        modified_net = ModifiedResNet18(models.resnet.BasicBlock, [2, 2, 2, 2], apply_after_layer=kwargs['hyperparams']['forestloss_params']['apply_after_layer'])
        modified_net.load_state_dict(net.state_dict())
        net = modified_net

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
    

    #If output_size is specified, create a fully connected layer with that out_features
    #Otherwise, the default fc layer with 1000 output size remains
    if output_size:
        net.fc = nn.Linear(net.fc.in_features, out_features=output_size, bias = True) # The number of output classes doesn't matter, it will get re-initialized

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


class ModifiedResNet18(models.ResNet):
    #Resnet modified to multiply feature map with smoothened forest loss map to make the model focus on relevant parts of image
    def __init__(self, block, layers, apply_after_layer, num_classes=1000, zero_init_residual=False,
                 groups=1, width_per_group=64, replace_stride_with_dilation=None,
                 norm_layer=None):
        super(ModifiedResNet18, self).__init__(block, layers, num_classes, zero_init_residual,
                                               groups, width_per_group, replace_stride_with_dilation, norm_layer)
        # Initialize any additional layers or parameters here if needed

        self.apply_after_layer = apply_after_layer

    def forward(self, x, forestloss):
        
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        #print(x.shape) torch.Size([64, 64, 25, 25])
        if self.apply_after_layer == 1:
            x = torch.mul(x, forestloss) 

        x = self.layer1(x)
        #print(x.shape) torch.Size([64, 64, 25, 25])
        if self.apply_after_layer == 2:
            x = torch.mul(x, forestloss) 

        x = self.layer2(x) 
        #print(x.shape) torch.Size([64, 64, 13, 13])
        if self.apply_after_layer == 3:
            x = torch.mul(x, forestloss) 
        
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x
    



