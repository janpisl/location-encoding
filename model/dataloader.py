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
    def __init__(self,band_indices=[True, True, True, True, False]): #The default is RGBNIR (without the QA60 band)
        super(SelectBands, self).__init__()

        self.band_indices = band_indices

    def forward(self, img):
        #in temporal case, the first dimension is temporal - band indices are 2nd
        if len(img.shape) == 3:
            return img[self.band_indices]
        elif len(img.shape) == 4:
            return img[:, self.band_indices]


### TODO: Modify transforms to what i need
train_transform = transforms.Compose([
    transforms.CenterCrop(100),
    SelectBands(),
    NormalizeGEEImages(),
    transforms.RandomHorizontalFlip(),  # randomly flip image horizontally
    ])  

# loader for evaluation, no horizontal flip
eval_transform = transforms.Compose([
    transforms.CenterCrop(100),
    SelectBands(),
    NormalizeGEEImages(),
    ])  
 
    #transforms.Resize(64),  # resize the image to 64x64 (remove if images are already 64x64)






def fetch_dataloader(types, config):
    """
    Fetches the DataLoader object for each type in types from data_dir.

    Args:
        types: (list) has one or more of 'train', 'val', 'test' depending on which data is required
        data_dir: (string) directory containing the dataset
        config: (dict) configuration file

    Returns:
        data: (dict) contains the DataLoader object for each type in types
    """
    dataloaders = {}

    def seed_worker(worker_id):
        worker_seed = torch.initial_seed() % 2**32
        np.random.seed(worker_seed)
        random.seed(worker_seed)

    g = torch.Generator()
    g.manual_seed(config['hyperparams']['seed'])


    for split in ['train', 'val', 'test']:
        if split in types:
            #TODO: continue when data split exists
            
            annotation = pd.read_csv(config['data']['annotation_file'])

            # use the train_transformer if training data, else use eval_transformer without random flip
            if split == 'train':
                dl = DataLoader(
                    DriverDataset(
                        annotation.loc[annotation.split == split].reset_index(drop=True), 
                        config['data']['image_folder'],
                        transform=train_transform,
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
                        transform=eval_transform,
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



