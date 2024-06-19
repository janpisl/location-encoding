"""Train the model"""

import json
import sys
import argparse
import logging
import os
import itertools
import pdb
from pathlib import Path
import pprint

import pandas as pd
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
from model.model import MultiModalDriverClassifier



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
        config: (dict) configuration duh
    """

    # set model to training mode
    model.train()

    preds_batches = []
    targets_batches = []

    for train_batch, labels_batch in dataloader:

        image_batch, loc_batch, expansion_rates_batch, forestloss_batch, labels_batch = train_batch[0].to(device), train_batch[1].to(device), train_batch[2].to(device), train_batch[3].to(device), labels_batch.to(device)
        y_hat = model(image_batch, loc_batch, expansion_rates_batch, forestloss_batch)

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
        
        if config['wandb']['log_to_wandb']:
            try:
                wandb.log({"missing token value": model.missing_tab_value_token.detach().cpu().item()})
            except AttributeError:
                pass

    targets = torch.cat(targets_batches, axis=0).cpu().detach().numpy()
    preds = torch.cat(preds_batches, axis=0).cpu().detach().argmax(dim=1).numpy()
    
    #Compute metrics over the training set and log to wandb
    _ = compute_and_log_metrics(targets, preds, 'train', config['wandb']['log_to_wandb'])



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

        try:
            if epochs_since_last_best > config['hyperparams']['image_encoder']['freezing_patience']:
                # Freeze all layers except the last fully connected layer
                for name, param in model.image_encoder.named_parameters():
                    if 'fc' not in name:
                        param.requires_grad = False
        except KeyError:
            pass

        if epochs_since_last_best > early_stopping_patience:
            print(f'No improvement for {early_stopping_patience} epochs. Early stopping at epoch {epoch}')
            break

        # compute number of batches in one epoch (one full pass over the training set)
        train(model, optimizer, loss_fn, train_dataloader, device, config)


        # Evaluate for one epoch on validation set
        val_metrics = evaluate(model, loss_fn, val_dataloader, device, config)

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





def initialize_wandb(config, model, train_dl, val_dl):

    try:

        if config['hyperparams']['loc_encoder']['use'] is True:
            loc_enc_name = "yes" #config['hyperparams']['loc_encoder']['positional_embedding']
        else:
            loc_enc_name = "no"

        if config['hyperparams']['tabular_encoder']['use'] is True:
            tab_enc_name = config['hyperparams']['tabular_encoder']['type']
            tab_dataset = config['hyperparams']['tabular_encoder']['dataset']
        else:
            tab_enc_name = "no"

        train_split = [int(key) for key,value in config['hyperparams']['folds_to_split']['mapping'].items() if value == 'train' ]
        val_split = [int(key) for key,value in config['hyperparams']['folds_to_split']['mapping'].items() if value == 'val' ]

        try:
            freeze_after = f"freeze patience {config['hyperparams']['image_encoder']['freezing_patience']}" 
        except KeyError:
            freeze_after = ' ? freeze'

        name = f""" {config['hyperparams']['train_examples_per_class']} ex per class;
            {config['hyperparams']['image_encoder']['use']} ({config['n_images']}) img, {freeze_after}; {loc_enc_name} loc;  {tab_enc_name} {tab_dataset if tab_enc_name != 'no' else ''} tabular; 
            train {train_split}
            val {val_split}"""
    except:
        name = None



    
    wandb.init(
        project=config['wandb']['project'], 
        group=config['wandb']['group'],
        reinit=True,
        name=name
    )

    log_config = config['hyperparams']
    log_config['n_parameters'] = utils.count_parameters(model)

    log_config['train_dst_size'] = len(train_dl.dataset)
    log_config['val_dst_size'] = len(val_dl.dataset)

    wandb.config.update(log_config)

    return wandb.run.id

def main(config):

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    n_classes = 12
    #n_classes = len(np.unique(pd.read_csv(config['data']['annotation_file']).driver ))
    

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
    
    config['n_images'] = num_images
    if 'freeze_after' in config['hyperparams']['image_encoder'].keys():
        print('Old configuration for img freezing')

    # Define the model and optimizer
    model = MultiModalDriverClassifier(n_classes,  **config)
    model = model.to(device)

    '''
    # Separate the parameters
    img_enc = [param for name, param in model.image_encoder.named_parameters() if 'fc' not in name]
    other_params = [param for name, param in model.named_parameters() if "image_encoder" not in name]
    
    # Define the optimizer with two parameter groups
    optimizer = optim.Adam([
        {'params': other_params},
        {'params': img_enc, 'lr': config['hyperparams']['learning_rate']/100}  
    ], lr=config['hyperparams']['learning_rate'], weight_decay=config['hyperparams']['weight_decay']) '''



    optimizer = optim.Adam(model.parameters(), lr=config['hyperparams']['learning_rate'], weight_decay=config['hyperparams']['weight_decay'])

    loss_fn = nn.CrossEntropyLoss(reduction='mean')

    if config['wandb']['log_to_wandb']:
        run_id = initialize_wandb(config, model, train_dl, val_dl)
    else:
        run_id = 'wandb_id'

    model_dir = os.path.join(config['data']['output_folder'], f'{run_id}')
    Path(model_dir).mkdir(parents=True, exist_ok=True)

    with open(os.path.join(model_dir, 'config.json'), 'w') as sink:
        json.dump(config, sink)
    
    train_and_evaluate(model, train_dl, val_dl, optimizer, loss_fn, config, device, model_dir)



if __name__ == '__main__':
   
    with open(sys.argv[1]) as src:
        config = json.load(src)

    #There is an iterable section in the configuration to enable simple hyperparams tuning
    #The code below generate all combinations and then execute them one by one
    try:
        keys, values = zip(*config['hyperparams'].items())
        permutations_dicts = [dict(zip(keys, v)) for v in itertools.product(*values)]
    except TypeError: #If the config options are not iterable, just run a single experiment with the config as is
        permutations_dicts = [dict(config['hyperparams'])]
    print(f"{len(permutations_dicts)} variant(s) of config")

    '''for variant in permutations_dicts:
        config['hyperparams'] = variant
        utils.seed_everything(config['hyperparams']['seed'])
        config = utils.parse_boolean_recursion(config)
        main(config)'''


    for i, variant in enumerate(permutations_dicts):

        try:
            config['hyperparams'] = variant
            utils.seed_everything(config['hyperparams']['seed'])
            config = utils.parse_boolean_recursion(config)
            try:
                wandb.finish()
            except:
                pass
            main(config)
        except Exception as e:
            text =  pprint.pformat(config) + '\nError message:\n' + repr(e)  + '\n\n\n'
            with open("logs_18_6.txt", "a") as sink:
                sink.write(text)





 
