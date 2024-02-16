import os
import pdb
 
import torch
import rasterio
import pandas as pd
from torch import Tensor
from torch.utils.data import Dataset


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
    'shifting agriculture': 10
}



class DriverDataset(Dataset):
    def __init__(self, 
                 annotations_file, 
                 img_dir, 
                 transform, 
                 target_transform=None, 
                 #TODO: include loading forest loss patch for given sample
                 **kwargs):
    
        self.img_dir = img_dir
        self.img_labels = annotations_file if isinstance(annotations_file, pd.DataFrame) else pd.read_csv(annotations_file)

        #Randomly swapping filenames 
        #print(" ----  Randomizing labels! ---- ")
        #self.img_labels['driver'] = self.img_labels[['driver']].sample(frac=1, random_state=1).reset_index(drop=True)

        self.img_dir = img_dir
        self.transform = transform
        self.target_transform = target_transform

        try:
            self.model_type = kwargs['model_type']
        except KeyError:
            self.model_type = None


    def __len__(self):
        return len(self.img_labels)


    def load_images(self, idx):
        
        sampleid = str(self.img_labels.iloc[idx]['sampleid'])
        paths =  [os.path.join(self.img_dir, sampleid, img_path) for img_path in os.listdir(os.path.join(self.img_dir, sampleid)) ] 

        if len(paths) == 1 or self.model_type == 'monotemporal': #For monotemporal model, just grab the first image in folder
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


    def __getitem__(self, idx):
        image_data = self.load_images(idx)  

        class_name = self.img_labels.driver.iloc[idx]
        label = LABEL_MAPPING[class_name]

        if self.transform:
            image_data = self.transform(image_data)
        if self.target_transform:
            label = self.target_transform(label)    

        return image_data, label