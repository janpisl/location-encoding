
from model.model_blocks import *
import torch.nn as nn


class ImageDriverClassifier(nn.Module):
    def __init__(self, num_images, output_classes=None, return_att_scores=False, feat_extractor_only=False, **kwargs):
        super(ImageDriverClassifier, self).__init__()

        self.output_classes = output_classes
        if not output_classes:
            assert feat_extractor_only
    
        self.feat_extractor = AttentionFeatureExtractor(**kwargs)
        self.feat_extractor_only = feat_extractor_only
        if not self.feat_extractor_only:
            self.classifier = Classifier(self.feat_extractor.lstm_hidden_size, self.output_classes)

        self.return_att_scores = return_att_scores
        
     
    def forward(self, x, forestloss_patch=None):
        
        feat_extractor_outputs = self.feat_extractor(x, forestloss_patch)
        if self.feat_extractor_only:
            output = feat_extractor_outputs['features']
        else:
            output = self.classifier(feat_extractor_outputs['features'])

        if self.return_att_scores:
            if self.feat_extractor_only:
                raise NotImplementedError() 
            else:
                # Keep the contents from the feature extractor (may include att_scores) except for features which we don't want to keep
                output_dict = {k: v for k, v in feat_extractor_outputs.items() if k != 'features'}
                output_dict['class_logits'] = output
                return output_dict

        else:
            return output