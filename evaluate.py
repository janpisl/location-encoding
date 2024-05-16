from torchmetrics.classification import Accuracy, Precision, Recall, F1Score
import torch.nn.functional as F
import wandb
import torch
from sklearn.metrics import classification_report
import numpy as np
import pdb 
import matplotlib.pyplot as plt

from utils import write_confusion_matrix
from model.dataset import LABEL_MAPPING

CLASS_INDICES = [value for key, value in LABEL_MAPPING.items()]
CLASS_NAMES = [key for key in LABEL_MAPPING.keys()]

def evaluate(model, loss_fn, dataloader, device, config, return_preds_and_targets=False):
    """Evaluate the model on `num_steps` batches.

    Args:
        model: (torch.nn.Module) the neural network
        loss_fn: a function that takes batch_output and batch_labels and computes the loss for the batch
        dataloader: (DataLoader) a torch.utils.data.DataLoader object that fetches data
        config: hyperparameters
        return_preds_and_targets (bool): if True, returns a Tensor containing all predictions and another with all targets
    """

    model.eval()

    losses = []
    preds_batches = []
    targets_batches = []

    for data_batch, labels_batch in dataloader:

        image_batch, loc_batch, expansion_rates_batch, forestloss_batch, labels_batch = data_batch[0].to(device), data_batch[1].to(device), data_batch[2].to(device), data_batch[3].to(device), labels_batch.to(device)
        y_hat = model(image_batch, loc_batch, expansion_rates_batch, forestloss_batch)

        loss = loss_fn(y_hat, labels_batch)

        y_hat = F.softmax(y_hat, dim=1)
        y_hat = y_hat.cpu().detach()
        labels_batch = labels_batch.cpu().detach()

        losses.append(loss.item())
        preds_batches.append(y_hat)
        targets_batches.append(labels_batch)

    targets = torch.cat(targets_batches, axis=0).cpu().detach().numpy()
    preds = torch.cat(preds_batches, axis=0).cpu().detach().argmax(dim=1).numpy()

    metrics = compute_and_log_metrics(targets, preds, 'val', config['wandb']['log_to_wandb'], losses=losses)

    if return_preds_and_targets:
        return metrics, preds, targets

    return metrics


def compute_and_log_metrics(targets, preds, dataset_name, log_to_wandb, losses=None):

    #Not logging the confusion matrix atm since I don't use it and it's a lot of data
    #conf_matrix_file_path = write_confusion_matrix(targets, preds, labels=CLASS_INDICES,target_names=CLASS_NAMES)
    #conf_matrix_image = plt.imread(conf_matrix_file_path)

    metrics = classification_report(targets, preds, output_dict=True,labels=CLASS_INDICES,target_names=CLASS_NAMES,zero_division=0)
    log_dict = {}
    try:
        log_dict[f'{dataset_name} accuracy'] = metrics['accuracy']
    except KeyError: 
        # In case of some of the classes not represented in the dataset, the 'accuracy' is not reported
        # and instead they report 'micro avg' 'f1 score', 'precision' and 'recall' which are identical to accuracy (and also identical to each other)
        metrics['accuracy'] = metrics['micro avg']['f1-score'] 
        log_dict[f'{dataset_name} accuracy'] = metrics['accuracy']
    
    log_dict[f'{dataset_name} macro f1-score'] = metrics['macro avg']['f1-score']
    
    #Not logging the confusion matrix atm since I don't use it and it's a lot of data
    #log_dict[f'{dataset_name} conf matrix'] = wandb.Image(conf_matrix_image)
    
    if losses:
        metrics['loss'] = sum(losses)/len(losses)
        log_dict[f'{dataset_name}-loss'] = metrics['loss']
    
    for _cls in CLASS_NAMES:
        log_dict[f"{_cls} {dataset_name} f1-score"] = metrics[_cls]['f1-score']
    
    if log_to_wandb:
        wandb.log(log_dict)
    
    return metrics
