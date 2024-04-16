

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




def get_classifier(output_classes, config):
    """Get a module with a forward() that takes in a list of tokens
    and outputs class logits 

    Args:
        output_classes (int): number of classes
        config (dict): 

    Returns:
        object inheriting from nn.Module: classifier of input tokens
    """

    model_type = config['hyperparams']['classifier']['type']

    if model_type == 'fc':
        input_size = _get_feature_vector_size(config)
        classifier = FullyConnectedClassifier(output_classes, input_dimension=input_size)
    elif model_type == 'transformer':
        classifier = TransformerClassifier(output_classes, **config)

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
        elif config['hyperparams']['tabular_encoder']['type'] == 'reports_token':
            feature_vector_dim += embed_dim
        elif config['hyperparams']['tabular_encoder']['type'] == 'goldman_token':
            feature_vector_dim += embed_dim

    return feature_vector_dim
    

class MultiModalDriverClassifier(nn.Module):

    def __init__(self, output_classes, **kwargs):
        super(MultiModalDriverClassifier, self).__init__()

        self.embedding_dim = kwargs['hyperparams']['embedding_dim']
        self.classifier = get_classifier(output_classes, kwargs)

        self.use_image = kwargs['hyperparams']['image_encoder']['use']
        if self.use_image:
            self.image_encoder = get_image_encoder(kwargs)
        

        if kwargs['hyperparams']['image_encoder']['freeze']:
            # Freeze all layers except the last fully connected layer
            for name, param in self.image_encoder.named_parameters():
                if 'fc' not in name:
                    param.requires_grad = False

        self.use_location = kwargs['hyperparams']['loc_encoder']['use']
        if self.use_location:
            self.loc_encoder = LocationEncoder(positional_embedding=kwargs['hyperparams']['loc_encoder']['positional_embedding'], feat_extractor_only=True, **kwargs)
        

        self.use_tabular = kwargs['hyperparams']['tabular_encoder']['use']
        if self.use_tabular:
            self.tabular_processing = kwargs['hyperparams']['tabular_encoder']['type']
            if self.tabular_processing == 'reports_token':
                self.tabular_encoder = nn.Linear(6, out_features=self.embedding_dim, bias = True)
            elif self.tabular_processing == 'goldman_token':
                self.tabular_encoder = nn.Linear(5, out_features=self.embedding_dim, bias = True)

    def forward(self, image_data, location, tabular_data, forestloss_patch=None):

        if self.use_image:
            image_features = self.image_encoder(image_data, forestloss_patch)
            features = [image_features]
        else:
            features = []

        if self.use_location:
            loc_features = self.loc_encoder(location)
            features.append(loc_features)

        if self.use_tabular:
            if self.tabular_processing == 'reports_token' or self.tabular_processing == 'goldman_token':
                tabular_data = self.tabular_encoder(tabular_data)
            elif self.tabular_processing == 'country_id' and isinstance(self.classifier, TransformerClassifier):
                tabular_data = tabular_data.unsqueeze(1)
                tabular_data = tabular_data.repeat(1, self.embedding_dim)
            
            features.append(tabular_data)
        
        predictions = self.classifier(features)

        return predictions
