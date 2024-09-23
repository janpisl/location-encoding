#FCNet adapted from https://github.com/macaodha/geo_prior/tree/master


import torch
import torch.utils.data
import torch.nn as nn
import torch.nn.functional as F
import math

from model.positional_embeddings import *

class LocationEncoder(nn.Module):
    def __init__(self, positional_embedding, output_classes=None, feat_extractor_only=False, **kwargs):
        super(LocationEncoder, self).__init__()

        if positional_embedding == 'direct':
            self.pe = Direct()
        elif positional_embedding == 'cartesian3d':
            self.pe = Cartesian3D()
        elif  positional_embedding == 'wrap':
            self.pe = Wrap()
        elif  positional_embedding == 'theory':
            self.pe = Theory() 
        
        else:
            raise ValueError(f"Unrecognized PE() method: {positional_embedding}")
        
        if not output_classes:
            assert feat_extractor_only
            
        try:
            nn_type = kwargs['hyperparams']['loc_encoder']['nn'].lower()
            assert nn_type in ['fcnet', 'linear']
        except (KeyError, AssertionError):
            nn_type = 'fcnet'
        
        if nn_type == 'fcnet':
            self.nn = FCNet(self.pe.embedding_dim, num_classes=output_classes, dim_hidden=kwargs['hyperparams']['embedding_dim'], feat_extractor_only=feat_extractor_only)
        elif nn_type == 'linear':
            self.nn = nn.Linear(self.pe.embedding_dim, kwargs['hyperparams']['embedding_dim'])


    def forward(self, x):

        embedded_x = self.pe(x)

        return self.nn(embedded_x)



class ResLayer(nn.Module):
    def __init__(self, linear_size):
        super(ResLayer, self).__init__()
        self.l_size = linear_size
        self.nonlin1 = nn.ReLU(inplace=True)
        self.nonlin2 = nn.ReLU(inplace=True)
        self.dropout1 = nn.Dropout()
        self.w1 = nn.Linear(self.l_size, self.l_size)
        self.w2 = nn.Linear(self.l_size, self.l_size)

    def forward(self, x):
        y = self.w1(x)
        y = self.nonlin1(y)
        y = self.dropout1(y)
        y = self.w2(y)
        y = self.nonlin2(y)
        out = x + y

        return out

class FCNet(nn.Module):
    def __init__(self, num_inputs, num_classes, dim_hidden, feat_extractor_only):
        super(FCNet, self).__init__()
        self.inc_bias = False

        self.feat_extractor_only = feat_extractor_only
        if not self.feat_extractor_only:
            self.classifier = nn.Linear(dim_hidden, num_classes, bias=self.inc_bias)

        self.loc_encoder = nn.Sequential(nn.Linear(num_inputs, dim_hidden),
                                    nn.ReLU(inplace=True),
                                    ResLayer(dim_hidden),
                                    ResLayer(dim_hidden),
                                    ResLayer(dim_hidden),
                                    ResLayer(dim_hidden))

    def forward(self, x):
    
        loc_emb = self.loc_encoder(x)
        if  self.feat_extractor_only:
            output = loc_emb
        else:
            output = self.classifier(loc_emb)

        return output
    
