"""
"""

from torch import nn
import torch.nn.functional as F
from torchvision.models import resnet18, resnet34,resnet50
import pdb
import torch.nn as nn
import torch.nn.functional as F
import torch
import copy
from typing import List

from model.model_blocks import *
from model.loc_encoder import LocationDriverClassifier
from model.multimodal import MultiModalClassifier, MultiModalTransformerClassifier
from model.image_encoder import ImageDriverClassifier
        



def get_model(config, output_classes=None):

    if config['hyperparams']['model_type'] == 'monotemporal':
        return get_resnet(config['hyperparams']['backbone'], output_classes, **config)
    elif config['hyperparams']['model_type'] == 'multitemporal':
        return ImageDriverClassifier(config['n_images'], output_classes=output_classes, **config)
    elif config['hyperparams']['model_type'] == 'loc_encoder':
        return LocationDriverClassifier(positional_embedding=config['hyperparams']['positional_embedding'], output_classes=output_classes, **config)
    elif config['hyperparams']['model_type'] == 'multimodal_cls':
        return MultiModalClassifier(output_classes, **config)
    elif config['hyperparams']['model_type'] == 'multimodal_transformer':
        return MultiModalTransformerClassifier(output_classes, **config)

    else:
        raise ValueError('unknown model type')


