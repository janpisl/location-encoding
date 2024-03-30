import os
import pdb
import random

import torch
import numpy as np
import pandas as pd
from torch.nn import Module
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

from model.dataset import DriverDataset



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


def get_forestloss_transform(config):
    """Make a transform consisting of a GaussianBlur (not applied when values across patch are constant)
    and resizing depending on where in the ResNet it will be applied

    Args:
        config (dict): contains configuration of the gaussian blur and which ResNet layer the forestloss patch will be applied *after* 

    Returns:
        transforms.transform
    """
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

    return forestloss_transform



def fetch_dataloader(types, config):
    """
    Fetches the DataLoader object for each type in types from data_dir.

    Args:
        types: (list) has one or more of 'train', 'val', 'test' depending on which data is required
        config: (dict) configuration file

    Returns:
        data: (dict) contains the DataLoader object for each type in types
    """

    #Possible to configure which bands to use, if none, 
    try:
        band_indices = config['hyperparams']['band_indices']
    except KeyError:
        band_indices = None

    train_transform = transforms.Compose([
        transforms.CenterCrop(100),
        SelectBands(band_indices),
        NormalizeGEEImages(),
        #transforms.RandomHorizontalFlip(),  #turned off to to avoid having to link this randomness with forestloss transforms
        ])  

    eval_transform = transforms.Compose([
        transforms.CenterCrop(100),
        SelectBands(band_indices),
        NormalizeGEEImages(),
        ])
    
    forestloss_transform = get_forestloss_transform(config)


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

            annotation['split'] = annotation['split'].map(folds_to_split_mapping) 

            if split == 'train':
                _transform = train_transform
                _shuffle = True
            elif split == 'val' or split == 'test':
                _transform = eval_transform
                _shuffle = False   
            else:
                raise ValueError('Unknown split')

            dl = DataLoader(
                DriverDataset(
                    annotation.loc[annotation.split == split].reset_index(drop=True), 
                    config['data']['image_folder'],
                    transform=_transform,
                    forestloss_transform=forestloss_transform,
                    **config
                    ), 
                batch_size=config['hyperparams']['batch_size'], 
                shuffle=_shuffle,
                num_workers=32,
                pin_memory=True,
                worker_init_fn=seed_worker,
                generator=g,
                )

            dataloaders[split] = dl


    return dataloaders



