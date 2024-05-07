import os
import pdb
import json

import torch
import rasterio
import numpy as np
import pandas as pd
from torch import Tensor
from torch.utils.data import Dataset
import geopandas as gpd
from scipy.ndimage import gaussian_filter



LABEL_MAPPING = {
    'pasture' : 0, 
    'forest plantation' : 1,
    'soy' : 2,
    'coffee' : 3,
    'palm oil': 4,
    'mining': 5,
    'urban': 6,
    'cocoa': 7,
    'rubber': 8,
    'forest regrowth': 9,
    'shifting agriculture': 10,
    'other': 11 
}



class DriverDataset(Dataset):
    def __init__(self, 
                 annotations_file, 
                 img_dir, 
                 transform, 
                 target_transform=None, 
                 **kwargs):
    
        self.img_dir = img_dir
        self.img_labels = annotations_file if isinstance(annotations_file, pd.DataFrame) else pd.read_csv(annotations_file)
        self.img_labels['geometry'] = gpd.GeoSeries.from_wkt(self.img_labels['geometry'])

        
        #self.min_year = 2000

        #Randomly swapping filenames 
        #print(" ----  Randomizing labels! ---- ")
        #self.img_labels['driver'] = self.img_labels[['driver']].sample(frac=1, random_state=1).reset_index(drop=True)

        self.img_dir = img_dir
        self.transform = transform
        self.target_transform = target_transform

        self.use_image = kwargs['hyperparams']['image_encoder']['use']

        self.use_forestloss = kwargs['hyperparams']['use_forestloss']
        if self.use_forestloss:
            self.forestloss_dir = kwargs['data']['forestloss_folder']
            self.forestloss_transform = kwargs['forestloss_transform']


        self.use_tabular = kwargs['hyperparams']['tabular_encoder']['use']
        if self.use_tabular:
            self.tabular_processing = kwargs['hyperparams']['tabular_encoder']['type'] #Should be 'country_id' or 'token'
            try:
                self.tabular_temporal_processing = kwargs['hyperparams']['tabular_encoder']['temporal_processing'] #Should be 'dynamic' or 'static'
            except KeyError:
                self.tabular_temporal_processing = 'static'
            self.tabular_dataset = kwargs['hyperparams']['tabular_encoder']['dataset'].lower()
            if self.tabular_dataset == 'faostat':
                self.tabular_data = pd.read_csv(kwargs['data']['faostat_data']) 
                #self.tabular_column_names = ['Cattle', 'Palm oil', 'Soya beans', 'Cocoa beans', 'Coffee, green', 'Natural rubber in primary forms']
                #self.tabular_cls_indices = torch.tensor([0, 4, 2, 7, 3, 8]) 

                self.tabular_column_names = ['Cattle', 'Soya beans', 'Cocoa beans', 'Coffee, green', 'Oil palm fruit', 'Natural rubber in primary forms']
                self.tabular_cls_indices = torch.tensor([0, 2, 7, 3, 4, 8]) 
            elif self.tabular_dataset == 'goldman':
                self.tabular_data = pd.read_csv(kwargs['data']['goldman_data'])
                self.tabular_column_names = ['oil_palm_loss', 'soy_loss', 'cocoa_loss', 'pasture_loss', 'coffee_loss']
                self.tabular_cls_indices = torch.tensor([4, 2, 7, 0, 3]) 
            elif self.tabular_dataset == 'deduce':
                self.tabular_data = pd.read_csv(kwargs['data']['deduce_data'])
                self.tabular_column_names = ['Pasture', 'Forest plantation',  
                                    'Soya beans', 'Coffee, green', 'Oil palm fruit',
                                    'Cocoa beans', 'Natural rubber in primary forms']
                self.tabular_cls_indices = torch.tensor([0,1,2,3,4,7,8]) 


    def __len__(self):
        return len(self.img_labels)


    def load_images(self, idx):

        if not self.use_image: #Avoid loading images if not using them; just return something so that the dataloader does not braeak
            return Tensor([0])
        
        sampleid = str(self.img_labels.iloc[idx]['sampleid'])
        paths =  [os.path.join(self.img_dir, sampleid, img_path) for img_path in os.listdir(os.path.join(self.img_dir, sampleid)) ] 

        if len(paths) == 1: #For monotemporal model, just grab the only image in folder
            with rasterio.open(paths[0]) as source:
                loaded_data = Tensor(source.read().astype('int16')) #The transforms expect this to be a tensor
        else:
            images = []
            for img_path in paths:
                with rasterio.open(img_path) as source:
                    data = Tensor(source.read().astype('int16')) #The transforms expect this to be a tensor
                images.append(data)
            loaded_data = torch.stack(images, dim=0)
        return loaded_data


    def _get_commodity_data(self, idx, forestloss_patch=None):
        #TODO: include also data quality flag - there are 5 possibilities (official, estimated, missing, imputed, value from int'l org)        
        
        if not self.use_tabular:
            return Tensor([0]) #Just return something so that the dataloader doesn't crash
        if self.tabular_temporal_processing == 'static':
            rates = self._get_static_commodity_data(idx)
        elif self.tabular_temporal_processing == 'dynamic':    
            assert forestloss_patch is not None, 'forest loss data needed for dynamic commodity processing'
            rates = self._get_dynamic_commodity_data(idx, forestloss_patch)
        else:
            raise ValueError('commodity data processing can be static or dynamic')

        #np.random.shuffle(rates[0]) #as a sanity check, random shuffle of the values 
        # Create a tensor of size 12, fill with values at the corresponding indices
        #TODO: learn a 'missing data' token instead
        rates = np.nan_to_num(rates.astype('float32'), nan=0.0)
        tabular_data = torch.zeros((12), dtype=torch.float)
        tabular_data[self.tabular_cls_indices] = torch.Tensor(rates.squeeze())
        tabular_data = torch.nan_to_num(tabular_data, nan=0.0)
        tabular_data = tabular_data.squeeze()

        #tabular_data = add_noise(tabular_data, noise_level=0.05) #Optionally add noise

        return tabular_data


    def _get_dynamic_commodity_data(self, idx, forestloss_patch): #TODO: Max year should be 2022, this is temporary

        years = forestloss_patch[(forestloss_patch >= 2000) & (forestloss_patch <= 2020)]
        try:
            #min_year, max_year = years.min(), years.max()
            min_year, max_year = np.percentile(years.flatten(), 10), np.percentile(years.flatten(), 90)
            min_year = max(min_year - 1, 2000) #this is because I want the expansion rates with respect to before the deforestation event
        except:
            min_year, max_year = 2000, 2020
       
        country_data = self.tabular_data.loc[self.tabular_data['country'] == self.img_labels.iloc[idx].country]
        rates = []
        for _class in self.tabular_column_names:
            try:
                production_at_the_start = country_data.loc[(country_data.Item == _class) & (country_data.Year == int(min_year)), 'Value' ].item()
                production_at_the_end = country_data.loc[(country_data.Item == _class) & (country_data.Year == int(max_year)), 'Value' ].item()
                rate = production_at_the_end - production_at_the_start
            except ValueError: #Signifies that one (or both) of the values are missing
                rate = np.nan
            
            rates.append(rate)
        
        return np.array(rates)


    def _get_static_commodity_data(self, idx):
        
        if self.tabular_processing == 'country_id':
            #TODO: this should not require any dataset; just a one-hot encoding of countries
            return torch.Tensor([self.tabular_production_data.loc[self.tabular_production_data.country == self.img_labels.iloc[idx].country].index.values[0]])

        assert self.tabular_processing == 'token', 'Only token processing implemented at the moment'
        if self.tabular_dataset == 'goldman':
 
            # This dataset is produced at a county-level, not country
            county_indices = self.img_labels.goldman_index.iloc[idx]
            county_indices = json.loads(county_indices)
            if len(county_indices) == 0:
                rates = np.array([np.nan, np.nan, np.nan, np.nan, np.nan], dtype='float32')
            else:
                rates = self.tabular_data.loc[county_indices[0], self.tabular_column_names].values
                if len(rates.shape) == 2: #If the examples intersects with multiple counties, take the average
                    rates = np.nanmean(rates, axis=0)
        else:
            country_data = self.tabular_data.loc[self.tabular_data['country'] == self.img_labels.iloc[idx].country]
            rates = country_data[self.tabular_column_names].values
            
        return rates
        

    def __getitem__(self, idx):

        image_data = self.load_images(idx)  

        class_name = self.img_labels.driver.iloc[idx]
        label = LABEL_MAPPING[class_name]

        #Get location as a tensor
        lat, lon = self.img_labels.iloc[idx].geometry.y, self.img_labels.iloc[idx].geometry.x
        loc = torch.Tensor([lat, lon])

        if self.transform and self.use_image:
            image_data = self.transform(image_data)
        if self.target_transform:
            label = self.target_transform(label)    
        
        forestloss_data = Tensor([0]) # In case it is not used, return something valid to avoid dataloader crashing
        #Load forestloss patch if applicable
        if self.use_forestloss:
            sampleid = str(self.img_labels.iloc[idx]['sampleid'])
            path = os.path.join(self.forestloss_dir, f'{sampleid}.tif')
            with rasterio.open(path) as src:
                forestloss_data = Tensor(src.read())
                forestloss_original  = forestloss_data.detach().clone() #I need the forestloss data pre-transform to get the minimum and maximum forest loss years
                #TODO: find bug - forestloss_data should be just integer numbers but they are not
                #patch_min_year = forestloss_data[forestloss_data >= self.min_year].min()
                forestloss_data = self.forestloss_transform(forestloss_data)
        
        tabular_data = self._get_commodity_data(idx, forestloss_original)
        inputs  = [image_data, loc, tabular_data, forestloss_data]

        return inputs, label



def add_noise(tensor, noise_level=0.01):
    # Compute the range of the tensor
    value_range = tensor.max() - tensor.min()
    
    # Generate random noise scaled by the range of the tensor and the desired noise level
    noise = noise_level * value_range * torch.randn_like(tensor)
    
    # Create a mask for non-zero elements
    mask = tensor != 0
    
    # Apply noise only to non-zero elements
    tensor = tensor + noise * mask
    
    return tensor