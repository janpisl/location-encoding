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

FOREST_LOSS_YEAR_RANGE = 2000, 2020



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
        #Randomly swapping filenames 
        #print(" ----  Randomizing labels! ---- ")
        #self.img_labels['driver'] = self.img_labels[['driver']].sample(frac=1, random_state=1).reset_index(drop=True)

        self.img_dir = img_dir
        self.transform = transform
        self.target_transform = target_transform

        self.use_image = kwargs['hyperparams']['image_encoder']['use']

        self.use_forestloss = kwargs['hyperparams']['use_forestloss']
        self.forestloss_dir = kwargs['data']['forestloss_folder']
        self.forestloss_transform = kwargs['forestloss_transform']


        self.use_tabular = kwargs['hyperparams']['tabular_encoder']['use']
        if self.use_tabular:
            self.tabular_processing = kwargs['hyperparams']['tabular_encoder']['type'] #Should be 'country_id' or 'token'
            try:
                self.tabular_dropout = kwargs['hyperparams']['tabular_encoder']['dropout']
                assert self.img_labels.split.unique()[0] == 'train' #For validation/test, no dropout should be used
            except (KeyError, AssertionError):
                self.tabular_dropout = 0.
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
        
        if self.tabular_processing == 'country_id':
            #TODO: this should not require any dataset; just a one-hot encoding of countries
            return torch.Tensor([self.tabular_production_data.loc[self.tabular_production_data.country == self.img_labels.iloc[idx].country].index.values[0]])

        if self.tabular_temporal_processing == 'static':
            min_year, max_year = FOREST_LOSS_YEAR_RANGE
        elif self.tabular_temporal_processing == 'dynamic':
            min_year, max_year = self._get_range(forestloss_patch, 10, 90)
        
        # Using the tabular dataset, grab rows corresponding to the country of given example
        country_data = self.tabular_data.loc[self.tabular_data['country'] == self.img_labels.iloc[idx].country]

        if self.tabular_dataset == 'deduce':
            # Grab rows containing data for target time period
            data_in_period = country_data.loc[(country_data['Year'] >= min_year) & (country_data['Year'] <= max_year) ]
            rates = []
            # For each class, sum the deforestation attributed to that class over the target period
            for _cls in self.tabular_column_names:
                target_column = 'Commodity group' if _cls in ['Pasture', 'Forest plantation'] else 'Commodity'
                values_per_year = data_in_period.loc[data_in_period[target_column] == _cls]['commodity attributed deforestation']
                target_value = values_per_year.sum(skipna=False, min_count=1)/values_per_year.count()
                rates.append(target_value)
            rates = np.array(rates)
        elif self.tabular_dataset == 'faostat':
            rates = self._get_faostat_commodity_expansion(country_data, min_year, max_year)
        elif self.tabular_dataset == 'goldman':
            raise NotImplementedError("Goldman not yet implemented")

            # This dataset is produced at a county-level, not country
            '''county_indices = self.img_labels.goldman_index.iloc[idx]
            county_indices = json.loads(county_indices)
            if len(county_indices) == 0:
                rates = np.array([np.nan, np.nan, np.nan, np.nan, np.nan], dtype='float32')
            else:
                rates = self.tabular_data.loc[county_indices[0], self.tabular_column_names].values
                if len(rates.shape) == 2: #If the examples intersects with multiple counties, take the average
                    rates = np.nanmean(rates, axis=0)'''


        # Create a tensor of size 12, fill with values at the corresponding indices
        tabular_data = torch.full((12,), fill_value=float('nan')) 
        tabular_data[self.tabular_cls_indices] = torch.Tensor(rates.squeeze())
        #tabular_data = torch.nan_to_num(tabular_data, nan=0.0)
        tabular_data = tabular_data.squeeze()
        if self.tabular_dropout > 0:
            tabular_data = set_nan_with_probability(tabular_data, self.tabular_dropout)

        return tabular_data


    def _get_range(self, array, min_perc=0, max_perc=100):
        """Find minimum and maximum values from input array that are
        within the FOREST_LOSS_YEAR_RANGE; if min_perc and/or max_perc 
        are specified, return the corresponding percentiles 

        Args:
            array (np.array): input array (forestloss patch)
            min_perc (int, optional): minimum percentile to be used. Defaults to 0.
            max_perc (int, optional): maximum percentile. Defaults to 100.

        Returns:
            tuple(int, int): the minimum and maximum value
        """

        years = array[(array >= FOREST_LOSS_YEAR_RANGE[0]) & (array <= FOREST_LOSS_YEAR_RANGE[1])]
        try:
            min_year, max_year = np.percentile(years.flatten(), min_perc), np.percentile(years.flatten(), max_perc)
        except:
            min_year, max_year = FOREST_LOSS_YEAR_RANGE #This is just in case the array only contains 0s, which should not happen but does happen for some corner cases

        return min_year, max_year


    def _get_faostat_commodity_expansion(self, country_data, min_year, max_year):

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



    def __getitem__(self, idx):
        try:
            image_data = self.load_images(idx)  
        except:
            print(self.img_labels.iloc[idx])
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
        #Load forestloss patch  
        sampleid = str(self.img_labels.iloc[idx]['sampleid'])
        path = os.path.join(self.forestloss_dir, f'{sampleid}.tif')
        with rasterio.open(path) as src:
            forestloss_data = Tensor(src.read())
            forestloss_original  = forestloss_data.detach().clone() #I need the forestloss data pre-transform to get the minimum and maximum forest loss years
        if self.use_forestloss:
            forestloss_data = self.forestloss_transform(forestloss_data)
        
        tabular_data = self._get_commodity_data(idx, forestloss_original)
        inputs  = [image_data, loc, tabular_data, forestloss_data]

        return inputs, label








def set_nan_with_probability(tensor, probability):
    mask = torch.rand(tensor.shape) < probability
    tensor[mask] = float('nan')
    return tensor