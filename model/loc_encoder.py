#FCNet adapted from https://github.com/macaodha/geo_prior/tree/master


import torch
import torch.utils.data
import torch.nn as nn
import torch.nn.functional as F
import math


from model.positional_embeddings import *

class LocationDriverClassifier(nn.Module):
    def __init__(self, output_classes, positional_embedding, **kwargs):
        super(LocationDriverClassifier, self).__init__()

        if positional_embedding == 'direct':
            self.pe = Direct()
        elif positional_embedding == 'cartesian3d':
            self.pe = Cartesian3D()
        elif  positional_embedding == 'wrap':
            self.pe = Wrap()
        else:
            raise ValueError(f"Unrecognized PE() method: {positional_embedding}")
        
        self.output_classes = output_classes
        
        #TODO: do this properly
        dim_hidden = 256

        self.nn =  FCNet(self.pe.embedding_dim, num_classes=output_classes, dim_hidden=dim_hidden)

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
    def __init__(self, num_inputs, num_classes, dim_hidden):
        super(FCNet, self).__init__()
        self.inc_bias = False
        self.classifier = nn.Linear(dim_hidden, num_classes, bias=self.inc_bias)

        self.loc_encoder = nn.Sequential(nn.Linear(num_inputs, dim_hidden),
                                    nn.ReLU(inplace=True),
                                    ResLayer(dim_hidden),
                                    ResLayer(dim_hidden),
                                    ResLayer(dim_hidden),
                                    ResLayer(dim_hidden))

    def forward(self, x):
        loc_emb = self.loc_encoder(x)
        class_pred = self.classifier(loc_emb)
        
        return class_pred
    
