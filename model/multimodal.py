

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

    if config['hyperparams']['image_encoder']['type'] == 'cnn-att-lstm':
        feature_vector_dim = 1*embed_dim
    elif config['hyperparams']['image_encoder']['type'] == 'cnn':
        img_folder = config['data']['image_folder'].split('/')[-1]
        if 'four_images' in img_folder:
            feature_vector_dim = 4*embed_dim
        elif 'one_image' in img_folder:
            feature_vector_dim = 1*embed_dim
        else:
            raise ValueError(f'Unexpected value: "{img_folder}"')
    
    #Location
    feature_vector_dim += embed_dim

    return feature_vector_dim
    



class MultiModalDriverClassifier(nn.Module):


    def __init__(self, output_classes, **kwargs):
        super(MultiModalDriverClassifier, self).__init__()

        self.image_encoder = get_image_encoder(kwargs)
        self.use_location = kwargs['hyperparams']['loc_encoder']['use']
        if self.use_location:
            self.loc_encoder = LocationEncoder(positional_embedding=kwargs['hyperparams']['loc_encoder']['positional_embedding'], feat_extractor_only=True, **kwargs)
        self.classifier = get_classifier(output_classes, kwargs)


    def forward(self, image_data, location, forestloss_patch=None):

        image_features = self.image_encoder(image_data, forestloss_patch)

        if self.use_location:
            loc_features = self.loc_encoder(location)
            if len(image_features.shape) ==  len(loc_features.shape):
                assert len(image_features.shape) == 2, 'unexpected'
                tokens = torch.stack([image_features, loc_features])
            elif len(image_features.shape) == 3 and len(loc_features.shape) == 2:
                tokens = torch.cat([image_features, loc_features.unsqueeze(dim=0)])
        else:
            tokens = image_features if len(image_features.shape) == 3 else image_features.unsqueeze(dim=0)
        
        #Now tokens is a tensor of TxBxD, with T=number of tokens
        predictions = self.classifier(tokens)

        return predictions
