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
        tokens = torch.cat([token if len(token.shape) == 3 else token.unsqueeze(dim=0) for token in tokens], dim=0)

        #Make cls_token tensor in shape (1,B,D)
        cls_tokens = self.cls_token.repeat(tokens.shape[1],1).unsqueeze(dim=0)

        features = torch.cat([cls_tokens, tokens])

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





class TabularEncoder(nn.Module):
    def __init__(self, n_classes=12, token_size=10, out_dim=128):
        super(TabularEncoder, self).__init__()
        self.missing_tab_value_token = nn.Parameter(torch.zeros(token_size))
        torch.nn.init.normal_(self.missing_tab_value_token, std=1e-5)

        self.n_classes = n_classes
        self.token_size = token_size

        self.fc_layers = nn.ModuleList([nn.Linear(1, token_size) for _ in range(n_classes)])
        self.shared_fc_layer = nn.Linear(token_size*n_classes, out_dim)

        
    def forward(self, x):
        outputs = []
        
        for i in range(self.n_classes):
            current_feature = x[:, i].unsqueeze(1)  # Select the i-th feature for all batch items
            
            # Create a mask for NaN values
            mask = torch.isnan(current_feature)
            
            # Apply the fully connected layer to non-NaN values
            processed_feature = self.fc_layers[i](current_feature)
            
            # Replace NaN values with the missing value token
            current_feature_with_nan_replaced = torch.where(mask, self.missing_value_token, processed_feature)
            
            outputs.append(current_feature_with_nan_replaced)
        
        concatenated = torch.cat(outputs, dim=1)
        result = self.shared_fc_layer(concatenated)
        
        return result



def load_seco_weights(net, path='../pretrained_models/seco_resnet18_1m.ckpt'):

    #https://github.com/Lightning-AI/pytorch-lightning/issues/15451#issuecomment-1298724879
    from pytorch_lightning.utilities.migration import pl_legacy_patch
    with pl_legacy_patch():
        seco_weights = torch.load(path)
    enc_q = {k:v for k,v in seco_weights['state_dict'].items() if k.startswith('encoder_q')}
    enc_q_v2 = {k.replace("encoder_q.", ""):v for k,v in enc_q.items()}

    enc_q_v2['conv1.weight'] = enc_q_v2.pop('0.weight')
    enc_q_v2['bn.weight'] = enc_q_v2.pop('1.weight')
    enc_q_v2['bn1.bias'] = enc_q_v2.pop('1.bias')
    enc_q_v2['bn1.running_mean'] = enc_q_v2.pop('1.running_mean')
    enc_q_v2['bn1.running_var'] = enc_q_v2.pop('1.running_var')
    enc_q_v2['bn1.num_batches_tracked'] = enc_q_v2.pop('1.num_batches_tracked')
    enc_q_v3 = {k.replace("4.", "layer1."):v for k,v in enc_q_v2.items()}
    enc_q_v3 = {k.replace("5.", "layer2."):v for k,v in enc_q_v3.items()}
    enc_q_v3 = {k.replace("6.", "layer3."):v for k,v in enc_q_v3.items()}
    enc_q_v3 = {k.replace("7.", "layer4."):v for k,v in enc_q_v3.items()}

    net.load_state_dict(enc_q_v3, strict=False)


def load_moco_weights(net):
    from torchgeo.models import ResNet18_Weights
    
    #Get S-2 MoCo weights (13 channels)
    weights = ResNet18_Weights.SENTINEL2_ALL_MOCO
    state_dict = weights.get_state_dict(progress=True)

    #Select R,G,B,NIR channels (indices 1,2,3,7 - not sure how to verify this is correct)
    selected_channels = state_dict['conv1.weight'][:, [1, 2, 3, 7], :, :]
    state_dict['conv1.weight'] = selected_channels

    net.load_state_dict(state_dict, strict=False)



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

    '''try:
        use_seco = kwargs['hyperparams']['image_encoder']['seco_weights']
    except KeyError:
        use_seco = False
    if use_seco:
        load_seco_weights(net)'''

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
    

    try:
        use_moco = kwargs['hyperparams']['image_encoder']['moco_weights']
    except KeyError:
        use_moco = False
    if use_moco:
        load_moco_weights(net)


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
    



