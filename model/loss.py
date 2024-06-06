
import torch.nn as nn
import torch


class TabRegularizationLoss(nn.Module):
    def __init__(self, temperature, regularization_method):
        super(TabRegularizationLoss, self).__init__()
        self.cross_entropy_loss = nn.CrossEntropyLoss(reduction='mean')
        self.temperature = temperature
        if regularization_method == 'l1':
            self.reg_loss = nn.L1Loss(reduction='mean')
        elif regularization_method == 'l2':
            self.reg_loss = nn.MSELoss(reduction='mean')
        elif regularization_method == 'cosine_similarity':
            self.reg_loss = nn.CosineSimilarity()
        else:
            raise NotImplementedError('implemented options are l1, l2, cosine_similarity')


    def forward(self, outputs_aux, output_main, targets, regularization_loss_indices):
        """compute loss as a weighted sum of CrossEntropy and a regularization term,
        that brings together the aux and main predictions, if t

        Args:
            outputs_aux (Tensor): model predictions when tabular input is replaced with no-data
            output_main (Tensor): model predictions with full inputs
            targets (Tensor)
            regularization_loss_indices (Tensor): 
        
        Returns:
            _type_: _description_
        """
        output_main = output_main.clone()
        outputs_aux = outputs_aux.clone()
        ce_loss = self.cross_entropy_loss(output_main, targets)
        #TODO: i should consider that the other classes' logits can be scaled up or down with respect to the
        # preds of the aux model; 

        if isinstance(self.reg_loss, nn.CosineSimilarity):
            #the above TODO i can achieve with cosine similarity;
            #but i need to compute the loss for each item in the batch separately
            #while output_main[regularization_loss_indices] returns a single 1D vector for the entire batch
            #
            #so, i set all predictions where the tabular data was present to 0., because they
            #should in fact differ between the two
            #and if they are all z
            output_main[~regularization_loss_indices] = 0.
            outputs_aux[~regularization_loss_indices] = 0.
            additional_loss = (self.temperature * self.reg_loss(output_main, outputs_aux)).mean()
        else:
            additional_loss = self.temperature * self.reg_loss(output_main[regularization_loss_indices], outputs_aux[regularization_loss_indices])
        
        return ce_loss + additional_loss