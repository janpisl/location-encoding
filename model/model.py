

import torch.nn as nn
import torch

from model.model_blocks import *
from model.loc_encoder import LocationEncoder
from model.legacy_models import CNN_Att_LSTM

def get_image_encoder(config):
    """Get image encoder that takes in input in shape BxTxCxWxH or BxCxWxH 
    (does not work for cnn-att-lstm, which requires at least 2 temporal steps),
    and produces output of shape BxTxD, where D=embedding_dim, which contains
    features from each image
    
    Args:
        config (dict): _description_

    Returns:
        object inheriting from nn.Module (or None)
    """

    #In case no images+no encoder are to be used
    if config['hyperparams']['image_encoder']['use'] is False:
        return None
    
    encoder_type = config['hyperparams']['image_encoder']['type']

    if encoder_type.lower() == 'cnn-att-lstm':
        encoder = CNN_Att_LSTM(feat_extractor_only=True, **config)
    if encoder_type.lower() == 'cnn':
        encoder = TemporalWrapper(get_resnet(config['hyperparams']['image_encoder']['backbone'], config['hyperparams']['embedding_dim'], **config))

    return encoder

def get_loc_encoder(config):

    return LocationEncoder(positional_embedding=config['hyperparams']['loc_encoder']['positional_embedding'], feat_extractor_only=True, **config)
        



def get_classifier(output_classes, config):
    """Get a module with a forward() that takes in a list of tokens
    and outputs class logits 

    Args:
        output_classes (int): number of classes
        config (dict): 

    Returns:
        object inheriting from nn.Module: classifier of input tokens
    """

    classifier_type = config['hyperparams']['classifier']['type']
    model_version = config['hyperparams']['model_version']

    if classifier_type == 'fc':
        #input_size = _get_feature_vector_size(config)
        input_size = 525
        classifier = FullyConnectedClassifier(output_classes, input_dimension=input_size)
    elif classifier_type == 'transformer':
        if model_version == 'v1':
            classifier = TransformerClassifier(output_classes, **config)
        elif model_version == 'v2':
            classifier = TransformerClassifierV2(output_classes, **config) 
        elif model_version == 'v3':
            classifier = TransformerClassifierV3(output_classes, **config) 

    return classifier


def _get_feature_vector_size(config):
    """Temporary solution; compute the input size of the fully connected layer in a classifier

    Args:
        config (dict)

    Returns:
        int: size of concatenated features (which is the input size of the final fc layer)
    """
    embed_dim = config['hyperparams']['embedding_dim'] 
    feature_vector_dim = 0

    if config['hyperparams']['image_encoder']['use']: 
        if config['hyperparams']['image_encoder']['type'] == 'cnn-att-lstm':
            feature_vector_dim += embed_dim
        elif config['hyperparams']['image_encoder']['type'] == 'cnn':
            img_folder = config['data']['image_folder'].split('/')[-1]
            if 'four_images' in img_folder:
                feature_vector_dim += 4*embed_dim
            elif 'one_image' in img_folder:
                feature_vector_dim += 1*embed_dim
            else:
                raise ValueError(f'Unexpected value: "{img_folder}"')
    
    if config['hyperparams']['loc_encoder']['use']: 
        feature_vector_dim += embed_dim

    if config['hyperparams']['tabular_encoder']['use']: 
        if config['hyperparams']['tabular_encoder']['type'] == 'country_id':
            feature_vector_dim += 1
        elif config['hyperparams']['tabular_encoder']['type'] == 'reports_raw':
            feature_vector_dim += 6
        elif 'token' in config['hyperparams']['tabular_encoder']['type']:
            feature_vector_dim += embed_dim


    return feature_vector_dim


def repeat_to_target_dim(x, target_dim):
    """
    x: tensor of shape (B, D)
    target_dim: int, target size for dim 1
    Returns: tensor of shape (B, T)
    """
    B, D = x.shape
    if target_dim < D:
        return x[:, :target_dim]
    
    repeat_times = (target_dim + D - 1) // D  # ceil division
    x_repeated = x.repeat(1, repeat_times)  # shape (B, D * repeat_times)
    return x_repeated[:, :target_dim]
    

class MultiModalDriverClassifier(nn.Module):

    def __init__(self, output_classes, **kwargs):
        super(MultiModalDriverClassifier, self).__init__()

        try:
            self.model_version = kwargs['hyperparams']['model_version']
            assert self.model_version in ['v1', 'v2', 'v3']
        except KeyError:
            self.model_version = 'v1'

        self.use_forestloss = kwargs['hyperparams']['use_forestloss']
        self.embedding_dim = kwargs['hyperparams']['embedding_dim']
        self.classifier = get_classifier(output_classes, kwargs)
        self.use_image = kwargs['hyperparams']['image_encoder']['use']
        if self.use_image:
            self.image_encoder = get_image_encoder(kwargs)

        self.use_location = kwargs['hyperparams']['loc_encoder']['use']
        if self.use_location:
            self.loc_encoder = nn.Identity()
        self.use_tabular = kwargs['hyperparams']['tabular_encoder']['use']
        self.tabular_encoder = nn.Identity()
        try:
            learn_missing_value = kwargs['hyperparams']['tabular_encoder']['learn_missing_value']
        except KeyError:
            learn_missing_value = True
        if learn_missing_value:
            self.missing_tab_value_token = nn.Parameter(torch.zeros(1))
            torch.nn.init.normal_(self.missing_tab_value_token, std=1e-5)
        else:
            self.missing_tab_value_token = 0.
    
    def forward(self, image_data, location, tabular_data, forestloss_patch=None, dropout=0.):
        
        if self.use_image:
            if self.use_forestloss:
                image_features = self.image_encoder(image_data, forestloss_patch)
            else:
                image_features = self.image_encoder(image_data, None)
            features = [image_features]
        else:
            features = []

        if self.use_location:
            loc_features = self.loc_encoder(location)
            #loc_features = repeat_to_target_dim(loc_features, 128)
            features.append(loc_features)

        if self.use_tabular:
            tabular_data[torch.isnan(tabular_data)] = self.missing_tab_value_token
            tabular_data = self.tabular_encoder(tabular_data)
            #tabular_data = repeat_to_target_dim(tabular_data, 128)
            features.append(tabular_data)
        
        #Get all tokens in shape NxBxD and concat them together into a tensor of NxBxD (num_tokens, batch, embed_dim)
        #tokens = torch.cat([token if len(token.shape) == 3 else token.unsqueeze(dim=0) for token in features], dim=0)
        
        
        tokens = torch.cat(
            [
                image_features[0],
                image_features[1],
                image_features[2],
                image_features[3],
                loc_features,
                tabular_data
            ],
            dim=1
        )
        

        predictions = self.classifier(tokens)


        return predictions
