"""Train the model"""

import json
import sys
import argparse
import logging
import os
import itertools
import pdb
import wandb
import numpy as np
import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
#import torch.optim as optim

import utils

import model.dataloader as dataloader
from evaluate import evaluate, compute_and_log_metrics
from model.model import get_model




parser = argparse.ArgumentParser()
parser.add_argument('--data_dir', default='data/64x64_SIGNS',
                    help="Directory containing the dataset")
parser.add_argument('--model_dir', default='experiments/base_model',
                    help="Directory containing params.json")
parser.add_argument('--restore_file', default=None,
                    help="Optional, name of the file in --model_dir containing weights to reload before \
                    training")  # 'best' or 'train'


def train(model, optimizer, loss_fn, dataloader, device, config):
    """Train the model for one epoch

    Args:
        model: (torch.nn.Module) the neural network
        optimizer: (torch.optim) optimizer for parameters of model
        loss_fn: a function that takes batch_output and batch_labels and computes the loss for the batch
        dataloader: (DataLoader) a torch.utils.data.DataLoader object that fetches training data
        params: (Params) hyperparameters
        num_steps: (int) number of batches to train on, each of size params.batch_size
    """

    # set model to training mode
    model.train()

    preds_batches = []
    targets_batches = []

    for i, (train_batch, labels_batch) in enumerate(dataloader):
            
        train_batch, labels_batch = train_batch.to(device), labels_batch.to(device)

        y_hat = model(train_batch)
        loss = loss_fn(y_hat, labels_batch)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if config['wandb']['log_to_wandb']:
            wandb.log({"train-loss": loss.item()})


        y_hat = y_hat.cpu().detach()
        preds = F.softmax(y_hat, dim=1)
        labels_batch = labels_batch.cpu().detach()
        
        preds_batches.append(preds)
        targets_batches.append(labels_batch)

    #Compute metrics over the training set and log to wandb
    _ = compute_and_log_metrics(targets_batches, preds_batches, 'train', config['wandb']['log_to_wandb'])



def train_and_evaluate(model, train_dataloader, val_dataloader, optimizer, loss_fn, config, device, model_dir):
    """Train the model and evaluate every epoch.

    Args:
        model: (torch.nn.Module) the neural network
        train_dataloader: (DataLoader) a torch.utils.data.DataLoader object that fetches training data
        val_dataloader: (DataLoader) a torch.utils.data.DataLoader object that fetches validation data
        optimizer: (torch.optim) optimizer for parameters of model
        loss_fn: a function that takes batch_output and batch_labels and computes the loss for the batch
        params: (Params) hyperparameters
        model_dir: (string) directory containing config, weights and log
    """
    # reload weights from restore_file if specified
    try:
        print(f"Restoring parameters from {config['restore_file_path']}")
        utils.load_checkpoint(config['restore_file_path'], model, optimizer)
    except KeyError:
        pass

    best_val_acc = 0.0

    epochs = config['hyperparams']['epochs']
    epochs_since_last_best = 0
    early_stopping_patience = config['hyperparams']['early_stopping_patience']

    for epoch in range(epochs):

        if epochs_since_last_best > early_stopping_patience:
            print(f'No improvement for {early_stopping_patience} epochs. Early stopping')
            break

        # compute number of batches in one epoch (one full pass over the training set)
        train(model, optimizer, loss_fn, train_dataloader, device, config)


        # Evaluate for one epoch on validation set
        val_metrics = evaluate(model, loss_fn, val_dataloader, device, config, epoch)

        val_acc = val_metrics['accuracy']
        is_best = val_acc >= best_val_acc

        # Save weights
        utils.save_checkpoint({'epoch': epoch + 1,
                               'state_dict': model.state_dict(),
                               'optim_dict': optimizer.state_dict()},
                              is_best=is_best,
                              out_folder=model_dir)
        
        epochs_since_last_best += 1

        if is_best:
            best_val_acc = val_acc
            epochs_since_last_best = 0


def main(config):


    if config['wandb']['log_to_wandb']:
        wandb.init(
            project=config['wandb']['project'], 
            group=config['wandb']['group'],
            reinit=True
        )

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # fetch dataloaders
    dataloaders = dataloader.fetch_dataloader(
        ['train', 'val'], config)
    train_dl = dataloaders['train']
    val_dl = dataloaders['val']
   
    #check how many images there are in one sampleid folder 
    num_images = len(
        os.listdir(
            os.path.join(
                config['data']['image_folder'],
                str(train_dl.dataset.img_labels.iloc[0].sampleid)
                )
        )
    )
    
    # Define the model and optimizer
    model = get_model(config, num_images, output_classes=9) 
    model = model.to(device)

    optimizer = optim.Adam(model.parameters(), lr=config['hyperparams']['learning_rate'])
    loss_fn = nn.CrossEntropyLoss(reduction='mean')

    #TODO: replace with actual wandb_id
    model_dir = os.path.join(config['data']['output_folder'], 'wandb_id')
    
    with open(os.path.join(model_dir, 'config.json'), 'w') as sink:
        json.dump(config, sink)
    
    # Train the model
    train_and_evaluate(model, train_dl, val_dl, optimizer, loss_fn, config, device, model_dir)



if __name__ == '__main__':
   
    with open(sys.argv[1]) as src:
        config = json.load(src)

    #There is a iterable section in the configuration to enable simple tuning
    #The code below generate all combinations and then execute them one by one
    try:
        keys, values = zip(*config['hyperparams'].items())
        permutations_dicts = [dict(zip(keys, v)) for v in itertools.product(*values)]
    except TypeError: #If the config options are not iterable, just run a single experiment with the config as is
        permutations_dicts = [dict(config['hyperparams'])]
    print(f"{len(permutations_dicts)} variant(s) of config")

    for variant in permutations_dicts:
        config['hyperparams'] = variant
        utils.seed_everything(config['hyperparams']['seed'])
        config = utils.parse_boolean_recursion(config)
        main(config)  

 





 
