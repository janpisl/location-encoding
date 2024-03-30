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
from model.multimodal import MultiModalDriverClassifier
    

def get_model(config, output_classes=None):

    return MultiModalDriverClassifier(output_classes,  **config)

    #if config['hyperparams']['model_type'] == 'monotemporal':
    #    return get_resnet(config['hyperparams']['backbone'], output_classes, **config)


