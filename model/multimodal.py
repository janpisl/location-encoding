

import torch.nn as nn
import torch

from model.model_blocks import *
from model.loc_encoder import LocationDriverClassifier
from model.image_encoder import ImageDriverClassifier
from model.model_blocks import *


class MultiModalClassifier(nn.Module):


    def __init__(self, output_classes, **kwargs):
        super(MultiModalClassifier, self).__init__()

        
        self.image_encoder = ImageDriverClassifier(kwargs['n_images'], feat_extractor_only=True, **kwargs)

        self.loc_encoder = LocationDriverClassifier(positional_embedding=kwargs['hyperparams']['positional_embedding'], feat_extractor_only=True, **kwargs)

        #self.classifier = Classifier(256+512, output_classes) #this has 2 fc layers
        self.classifier = nn.Linear(256+512, out_features=output_classes, bias = True)

        self.output_classes = output_classes

    def forward(self, image_data, location, forestloss_patch=None):

        image_features = self.image_encoder(image_data, forestloss_patch)
        loc_features = self.loc_encoder(location)
        features = torch.cat([image_features, loc_features], dim=1)

        return self.classifier(features)
    



class MultiModalTransformerClassifier(nn.Module):


    def __init__(self, output_classes, **kwargs):
        super(MultiModalTransformerClassifier, self).__init__()

        embedding_dim = 256
        
        self.image_encoder = get_resnet(kwargs['hyperparams']['backbone'], output_size=embedding_dim, **kwargs)
        
        self.loc_encoder = LocationDriverClassifier(positional_embedding=kwargs['hyperparams']['positional_embedding'], feat_extractor_only=True, **kwargs)

        encoder_layer = nn.TransformerEncoderLayer(d_model=embedding_dim, nhead=8)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=1)
        self.cls_token = nn.Parameter(torch.randn(embedding_dim))
        self.classifier = nn.Linear(embedding_dim, out_features=output_classes, bias = True)

        self.output_classes = output_classes

    def forward(self, image_data, location, forestloss_patch=None):

        model_outputs  = []
        #Length of temporal series
        temporal_steps = image_data.shape[1]

        for i in range(temporal_steps):
            if forestloss_patch is not None:
                model_outputs.append(self.image_encoder(image_data[:,i], forestloss_patch))
            else:
                model_outputs.append(self.image_encoder(image_data[:,i]))

        loc_features = self.loc_encoder(location)
        cls_tokens = self.cls_token.repeat(location.shape[0],1)

        features = torch.stack([cls_tokens, *model_outputs, loc_features])

        outputs = self.transformer(features)
        predictions = self.classifier(outputs[0]) # Use the cls token to get class logits

        return predictions