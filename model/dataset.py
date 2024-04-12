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
            self.tabular_production_data = pd.read_csv(kwargs['data']['tabular_production_data'])
            self.tabular_processing = kwargs['hyperparams']['tabular_encoder']['type']
            self.FAOSTAT_classes = ['Cattle', 'Palm oil', 'Soya beans', 'Cocoa beans', 'Coffee, green', 'Natural rubber in primary forms']
        else:
            self.tabular_processing = 'country_id' #This is just so that when I don't use tabular data, it doesn't crash while keeping the inputs constant


        #self.use_goldman_data = kwargs['hyperparams']['tabular_encoder']['use_goldman_data22']
        self.use_goldman_data = True
        if self.use_goldman_data:
            #self.goldman_data = pd.read_csv(kwargs['data']['goldman_data'])
            self.goldman_data = pd.read_csv('data/tmp/goldman.csv')
            self.goldman_cols = ['oil_palm_loss', 'soy_loss', 'cocoa_loss', 'pasture_loss', 'coffee_loss']


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
    

    def _get_static_commodity_expansion_rates(self, country): #TODO: Max year should be 2022, this is temporary

        country_data = self.tabular_production_data.loc[self.tabular_production_data['country'] == country]
        values = country_data[self.FAOSTAT_classes].values
        #np.random.shuffle(values[0]) #as a sanity check, random shuffle of the values 
        
        return torch.Tensor(values)
    
    def _get_goldman_commodity_rates(self, county_indices):
        """_summary_

        Args:
            county_indices (list): 
        """
        
        county_indices = json.loads(county_indices)
        if len(county_indices) == 0:
            rates = np.array([0,0,0,0,0]) #This is also temporary
        else:
            rates = self.goldman_data.loc[county_indices[0], self.goldman_cols].values

            if len(rates.shape) == 2: #If the examples intersects with multiple counties, take the average
                rates = np.nanmean(rates, axis=0)

        #TEMPORARY - the real solution would be to learn a missing-data-token
        rates = np.nan_to_num(rates.astype('float'))
        
        return torch.Tensor(rates)


    def _get_dynamic_commodity_expansion_rates(self, country, min_year=2000, max_year=2020): #TODO: Max year should be 2022, this is temporary
        assert False, 'for now, the values are precomputed per-country'

        #TODO: include also data quality flag - there are 5 possibilities (official, estimated, missing, imputed, value from int'l org)        

        FAOSTAT_classes = ['Cattle', 'Palm oil', 'Soya beans', 'Cocoa beans', 'Coffee, green',
       'Natural rubber in primary forms']

        country_data = self.tabular_production_data.loc[self.tabular_production_data['country'] == country]
        pdb.set_trace()
        values = []

        for _class in FAOSTAT_classes:
            try:
                production_at_the_start = country_data.loc[(country_data.Item == _class) & (country_data.Year == int(min_year)), 'Value' ].item()
                production_at_the_end = country_data.loc[(country_data.Item == _class) & (country_data.Year == int(max_year)), 'Value' ].item()
                expansion = production_at_the_end - production_at_the_start
            except ValueError: #Signifies that one (or both) of the values are missing
                expansion = 0
            values.append(expansion)

        return torch.Tensor(values)


    def __getitem__(self, idx):

        #Load image(s)
        image_data = self.load_images(idx)  

        #Get label
        class_name = self.img_labels.driver.iloc[idx]
        label = LABEL_MAPPING[class_name]

        #Get location as a tensor
        lat, lon = self.img_labels.iloc[idx].geometry.y, self.img_labels.iloc[idx].geometry.x
        loc = torch.Tensor([lat, lon])

        if self.transform and self.use_image:
            image_data = self.transform(image_data)
        if self.target_transform:
            label = self.target_transform(label)    
        
        #TODO: test if this works when forestloss=False (maybe returning None won't work with the default collate fn)
        forestloss_data = None
        #Load forestloss patch if applicable
        if self.use_forestloss:
            sampleid = str(self.img_labels.iloc[idx]['sampleid'])
            path = os.path.join(self.forestloss_dir, f'{sampleid}.tif')
            with rasterio.open(path) as source:
                forestloss_data = Tensor(source.read())
                #TODO: find bug - forestloss_data should be just integer numbers but they are not
                #patch_min_year = forestloss_data[forestloss_data >= self.min_year].min()
                forestloss_data = self.forestloss_transform(forestloss_data)


        ### Adding tabular data - I can always add this even if they are not used - does not take a lot of time
        country = self.img_labels.iloc[idx].country
        if self.tabular_processing == 'country_id':
            tabular_data = torch.Tensor([self.tabular_production_data.loc[self.tabular_production_data['country'] == country].index.values[0]])
        elif self.tabular_processing == 'goldman_token':
            tabular_data = self._get_goldman_commodity_rates(self.img_labels.goldman_index.iloc[idx])
        else:
            tabular_data = self._get_static_commodity_expansion_rates(country)
        tabular_data = tabular_data.squeeze()

        inputs  = [image_data, loc, tabular_data, forestloss_data]
        #inputs  = [image_data, loc, commodity_production, forestloss_data]

        return inputs, label
    


