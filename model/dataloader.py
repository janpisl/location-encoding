import os
import pdb
import random

import torch
import numpy as np
import pandas as pd
from torch.nn import Module
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

from model.dataset import DriverDataset, LocationDataset



class NormalizeGEEImages(Module):       
    def __init__(self):
        super(NormalizeGEEImages, self).__init__()

    def forward(self, img):

        return img/10000


class SelectBands(Module):       
    def __init__(self,band_indices=None): 
        super(SelectBands, self).__init__()

        if band_indices:
            self.band_indices = band_indices
        else:
            self.band_indices = [True, True, True, True, False] #The default is RGBNIR (without the QA60 band)

    def forward(self, img):

        #in temporal case, the first dimension is temporal - band indices are 2nd
        if len(img.shape) == 3:
            return img[self.band_indices]
        elif len(img.shape) == 4:
            return img[:, self.band_indices]




class ConditionalGaussianBlur:
    """Applies GaussianBlur if multiple values are present; if only one value, returns input as is
        The condition is added because GaussianBlur to a constant tensor produces an unwanted result
    """
    def __init__(self, transform):
        self.transform = transform

    def __call__(self, x):
        # Check if all elements in x are 1 
        if torch.all(x == 1):
            # Return original tensor without applying the transform
            return x
        else:
            # Apply the transform
            return self.transform(x)


def fetch_dataloader(types, config):
    """
    Fetches the DataLoader object for each type in types from data_dir.

    Args:
        types: (list) has one or more of 'train', 'val', 'test' depending on which data is required
        config: (dict) configuration file

    Returns:
        data: (dict) contains the DataLoader object for each type in types
    """

    if config['hyperparams']['model_type'] == 'loc_encoder':
        pass #No need for transforms
    else:
        try:
            band_indices = config['hyperparams']['band_indices']
        except KeyError:
            band_indices = None

        train_transform = transforms.Compose([
            transforms.CenterCrop(100),
            SelectBands(band_indices),
            NormalizeGEEImages(),
            #transforms.RandomHorizontalFlip(),  # randomly flip image horizontally
            ])  

        eval_transform = transforms.Compose([
            transforms.CenterCrop(100),
            SelectBands(band_indices),
            NormalizeGEEImages(),
            ])
        
        if config['hyperparams']['use_forestloss']:
            layer = config['hyperparams']['forestloss_params']['apply_after_layer']
            assert layer in [1, 2, 3]
            if layer == 3: #Given input images of 100x100, after 3rd conv layer they are downsampled to (13,13)
                patch_size = (13, 13)
            else: #After the first and second conv layer, the size is constant (25, 25) # TODO: understand how this is possible, should be decreasing
                patch_size = (25, 25)

            kernel_size = (config['hyperparams']['forestloss_params']['kernel_size'], config['hyperparams']['forestloss_params']['kernel_size'])
            forestloss_transform = transforms.Compose([
                ConditionalGaussianBlur(
                    transforms.GaussianBlur(
                        kernel_size, 
                        sigma=config['hyperparams']['forestloss_params']['sigma'],
                        )
                    ), 
                transforms.Resize(patch_size, antialias=True),  
                ])       
        else:
            forestloss_transform = None
    

    dataloaders = {}

    def seed_worker(worker_id):
        worker_seed = torch.initial_seed() % 2**32
        np.random.seed(worker_seed)
        random.seed(worker_seed)

    g = torch.Generator()
    g.manual_seed(config['hyperparams']['seed'])


    for split in ['train', 'val', 'test']:
        if split in types:
           
            annotation = pd.read_csv(config['data']['annotation_file'])
            folds_to_split_mapping = {int(key):value for key, value in config['hyperparams']['folds_to_split'].items()} 
            ####temp solution
            '''folds_to_split_mapping = {
                0 : 'train',
                1 : 'train',
                2 : 'val',
                3 : 'train',
                4 : 'test'
            }'''
            annotation['split'] = annotation['split'].map(folds_to_split_mapping) 
            ###


            if split == 'train':
                if config['hyperparams']['model_type'] == 'loc_encoder':
                    dl = DataLoader(
                        LocationDataset(
                            annotation.loc[annotation.split == split].reset_index(drop=True), 
                            ), 
                        batch_size=config['hyperparams']['batch_size'], 
                        shuffle=True,
                        num_workers=32,
                        pin_memory=True,
                        worker_init_fn=seed_worker,
                        generator=g,
                        )  
                else:                  
                    dl = DataLoader(
                        DriverDataset(
                            annotation.loc[annotation.split == split].reset_index(drop=True), 
                            config['data']['image_folder'],
                            transform=train_transform,
                            model_type=config['hyperparams']['model_type'],
                            forestloss_transform=forestloss_transform,
                            **config
                            ), 
                        batch_size=config['hyperparams']['batch_size'], 
                        shuffle=True,
                        num_workers=32,
                        pin_memory=True,
                        worker_init_fn=seed_worker,
                        generator=g,
                        )
            else:
                if config['hyperparams']['model_type'] == 'loc_encoder':
                    dl = DataLoader(
                        LocationDataset(
                            annotation.loc[annotation.split == split].reset_index(drop=True), 
                            ), 
                        batch_size=config['hyperparams']['batch_size'], 
                        shuffle=False,
                        num_workers=32,
                        pin_memory=True,
                        worker_init_fn=seed_worker,
                        generator=g,
                        )                
                else:
                    dl = DataLoader(
                        DriverDataset(
                            annotation.loc[annotation.split == split].reset_index(drop=True), 
                            config['data']['image_folder'],
                            transform=eval_transform,
                            model_type=config['hyperparams']['model_type'],
                            forestloss_transform=forestloss_transform,
                            **config
                            ), 
                        batch_size=config['hyperparams']['batch_size'], 
                        shuffle=False,
                        num_workers=32,
                        pin_memory=True,
                        worker_init_fn=seed_worker,
                        generator=g,
                        )

            dataloaders[split] = dl


    return dataloaders



