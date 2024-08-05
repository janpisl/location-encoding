import os
import pdb
import random
import json

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
            #s2 bands= ['B1', 'B2', 'B3', 'B4', 'B5', 'B6', 'B7', 'B8', 'B8A', 'B9', 'B11', 'B12']
            #self.band_indices = [True, True, True, True, False] #The default is RGBNIR (without the QA60 band)
            self.band_indices = [False, True, True, True, False, False, False, True, False, False, False, False] #i have all 12 bands

    def forward(self, img):
        #try:
        #in temporal case, the first dimension is temporal - band indices are 2nd
        if len(img.shape) == 3:
            return img[self.band_indices]
        elif len(img.shape) == 4:
            return img[:, self.band_indices]
        #except:
        #    self.band_indices = [True, True, True, True]  #for more recent data i haven't been using the QA60 band
        #    return self.forward(img)



class ConditionalTransform(Module):
    """Applies transform if multiple values are present in input; if only one value, returns input as is
        The condition is added because GaussianBlur to a constant tensor produces an unwanted result
    """
    def __init__(self, transform):
        super(ConditionalTransform, self).__init__()
        self.transform = transform

    def __call__(self, x):
        # Check if all elements in x are 1 
        if torch.all(x == 1):
            # Return original tensor without applying the transform
            return x
        else:
            # Apply the transform
            return self.transform(x)



class BinarizeTensor(Module):
    """Returns a Tensor with value 1.0 where original value was between
    min and max value (inclusive), and 0.0 elsewhere
    """  
    def __init__(self, min_val=2000, max_val=2020): 
        super(BinarizeTensor, self).__init__()

        self.min_val = min_val
        self.max_val = max_val

    def forward(self, _input):

        _input[(_input < self.min_val) | (_input > self.max_val)] = 0.
        _input[(_input >= self.min_val) & (_input <= self.max_val)] = 1.

        return _input


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
            BinarizeTensor(),
            ConditionalTransform(
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



def sample_rows(df, column, n_rows):
    """
    Return a DataFrame which contains n_rows for each unique value in 'column', 
    randomly sampled. If there are less rows than n_rows for some value in 'column', 
    return all rows that have that value.

    Arguments:
        df: pandas.DataFrame
        column: str
        n_rows: int
        
    Returns:
        pandas.DataFrame
    """
    return df.groupby(column).apply(lambda x: x.sample(min(len(x), n_rows))).reset_index(drop=True)



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
            
            split_column = config['hyperparams']['folds_to_split']['column']
            
            try:
                folds_to_split_mapping = {int(key):value for key, value in config['hyperparams']['folds_to_split']['mapping'].items()} 
            except AttributeError: #in case not a dict is passed but a string
                with open(config['hyperparams']['folds_to_split']['mapping']) as src:
                    folds_to_split_mapping = json.loads(src.read())
                
            annotation['split'] = annotation[split_column].map(folds_to_split_mapping) 
            if split == 'train':
                _transform = train_transform
                _shuffle = True
                try:
                    # Select only a subset of the training data - in such a way that the number of examples per class is uniform
                    examples_per_class = config['hyperparams']['train_examples_per_class']
                    annotation = sample_rows(annotation.loc[annotation.split == split].reset_index(drop=True), 'driver', examples_per_class)
                except KeyError:
                    pass
                #Do not sample by examples_per_class, but just by the size of the train dataset - for figures
                try:
                    total_size = config['hyperparams']['train_dataset_total_size']
                    examples_per_class = int(total_size/len(annotation.loc[annotation.split == split].driver.unique()))
                    train_df = annotation.loc[annotation.split == split].reset_index(drop=True)
                    annotation = sample_rows(train_df, 'driver', examples_per_class)
                    if len(annotation) < total_size:
                        random_samples = train_df.loc[~train_df.sampleid.isin(annotation.sampleid)].sample((total_size - len(annotation)))
                        annotation = pd.concat([annotation, random_samples])
                except KeyError:
                    pass                


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
                num_workers=int(config['num_workers']),
                pin_memory=True,
                worker_init_fn=seed_worker,
                generator=g,
                )

            dataloaders[split] = dl


    return dataloaders



