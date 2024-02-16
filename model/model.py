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







class DriverClassifier(nn.Module):
    def __init__(self, num_images, output_classes, return_att_scores=False, **kwargs):
        super(DriverClassifier, self).__init__()

        self.output_classes = output_classes

        #TODO: replace this, now no more months before and after, it is simpler
        self.temporal_steps = num_images

        self.feat_extractor = AttentionFeatureExtractor(**kwargs)

        self.classifier = Classifier(self.feat_extractor.lstm_hidden_size, self.output_classes)

        self.return_att_scores = return_att_scores
     
    def forward(self, x):
        
        feat_extractor_outputs = self.feat_extractor(x)
        output = self.classifier(feat_extractor_outputs['features'])

        if self.return_att_scores:
            # Keep the contents from the feature extractor (may include att_scores) except for features which we don't want to keep
            output_dict = {k: v for k, v in feat_extractor_outputs.items() if k != 'features'}
            output_dict['class_logits'] = output
            return output_dict
        else:
            return output
        



def get_model(config, num_images, output_classes):
    #if 'in_channels' not in config.keys():
    #    in_channels = len(config['bands'])
    #    config['in_channels'] = in_channels
    #if not model_name:
    #    model_name = config['model']    

    if config['hyperparams']['model_type'] == 'monotemporal':
        return get_resnet(config['hyperparams']['backbone'], output_classes, **config)
    elif config['hyperparams']['model_type'] == 'multitemporal':
        return DriverClassifier(num_images, output_classes=output_classes, **config)
    elif config['hyperparams']['model_type'] == 'loc_encoder':
        return LocationDriverClassifier(output_classes=output_classes, positional_embedding=config['hyperparams']['positional_embedding'], **config)
        

    else:
        raise ValueError('unknown model type')
