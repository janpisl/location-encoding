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



class TransformerClassifier(nn.Module):
    def __init__(self, output_classes, **config):
        super(TransformerClassifier, self).__init__()

        embedding_dim = config['hyperparams']['embedding_dim']

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embedding_dim, 
            nhead=config['hyperparams']['classifier']['attention_heads'])
        self.transformer_enc = nn.TransformerEncoder(
            encoder_layer, 
            num_layers=config['hyperparams']['classifier']['transformer_layers'])

        self.cls_token = nn.Parameter(torch.randn(embedding_dim))

        self.classifier = nn.Linear(embedding_dim, out_features=output_classes, bias = True)

        
    def forward(self, tokens):
        """_summary_

        Args:
            tokens (Tensor): set of tokens, shape TxBxD (n_tokens, batch, embedding_dim)
        """

        #Make cls_token tensor in shape (1,B,D)
        cls_tokens = self.cls_token.repeat(tokens.shape[1],1).unsqueeze(dim=0)

        features = torch.concat([cls_tokens, tokens])

        cls_token_output = self.transformer_enc(features)[0]
        predictions = self.classifier(cls_token_output) 
        
        return predictions


class FullyConnectedClassifier(nn.Module):
    def __init__(self, output_classes, input_dimension, **config):
        super(FullyConnectedClassifier, self).__init__()

        self.classifier = nn.Linear(input_dimension, out_features=output_classes, bias = True)

        
    def forward(self, tokens):
        """_summary_

        Args:
            tokens (Tensor): set of tokens, shape TxBxD (n_tokens, batch, embedding_dim)
        """

        '''
        #TODO: Fix this, i don't know how to make it work with view or reshape 
        n_tokens, batch_size, embedding_dim = tokens.shape
        tokens.view(batch_size, embedding_dim*n_tokens)
        '''
        tokens = list(tokens)
        tokens = torch.cat(tokens, dim=1)

        return self.classifier(tokens)



class TemporalWrapper(nn.Module):
    def __init__(self, model):
        super(TemporalWrapper, self).__init__()

        self.model = model
        
    def forward(self, x, forestloss):
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



class TemporalWrapperV2(nn.Module):
    def __init__(self, model):
        super(TemporalWrapperV2, self).__init__()

        self.model = model
        
    def forward(self, x, forestloss):
        """_summary_

        Args:
            x (input time series): shape BxTxCxWxH
            model (optional)

        Returns:
            Tensor: output features of shape BxTxD (D=output dimension of model passed at init)
        """

        if len(x.shape) == 4: #If single-temporal (ie. no T dimenison), add the T axis
            x = x.unsqueeze(1)

        B, T, C, W, H = x.size()
        # Reshape x to: (B*T)xCxWxH to apply the model on each frame
        x_reshaped = x.view(B * T, C, W, H)
        # Apply the model
        if forestloss is not None:
            out = self.model(x_reshaped, forestloss)
        else:
            out = self.model(x_reshaped)
        
        # Assuming the model's output is a feature vector of shape (B*T)xFeatureDim
        # Reshape output back to: BxTxFeatureDim
        out_reshaped = out.view(B, T, -1)
        
        return out_reshaped


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
    



