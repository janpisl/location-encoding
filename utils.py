
import io
import json
import logging
import shutil
import pdb
import os
from pathlib import Path 
import matplotlib.pyplot as plt
import random

import torch
import rasterio
import numpy as np
import pandas as pd
import geopandas as gpd
from torchmetrics import ConfusionMatrix
from matplotlib.colors import LogNorm
from shapely.geometry import Polygon, box
from sklearn.metrics import confusion_matrix
from rasterio.warp import calculate_default_transform, reproject, Resampling

import seaborn as sns




BANDS = {
    "S2" : ['B1', 'B2', 'B3', 'B4', 'B5', 'B6', 'B7', 'B8', 'B8A', 'B9', 'B11', 'B12', 'QA60'],
    "S2_RGBNIR": ['B2', 'B3', 'B4', 'B8', 'QA60'], #These are the bands I am currently downloading 
    #"L8" : ['B1', 'B2', 'B3', 'B4', 'B5', 'B6', 'B7', 'B10', "B11", "QA_PIXEL"] # this is TOA
    "L8" : ['SR_B1', 'SR_B2', 'SR_B3', 'SR_B4', 'SR_B5', 'SR_B6', 'SR_B7', 'QA_PIXEL'] # this is SR
}

def seed_everything(seed: int):

    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def get_class_counts(df, cols):

    labels = df[cols]
    classes, counts = np.unique(labels.idxmax(axis=1), return_counts=True)

    return classes, counts


def parse_boolean(value):
    assert value.lower() in ['true', 'false'], f'Must be True or False, got {value}'
    return True if value.lower() == 'true' else False


def parse_boolean_recursion(value):
    if (isinstance(value, str) and (value.lower() in ['true', 'false'])):
        return True if value.lower() == 'true' else False
    elif isinstance(value, dict):
        for key in value.keys():
            value[key] = parse_boolean_recursion(value[key])
        return value
    else:
        return value

 

def compute_weights(labels):
    """Compute weights to be used in loss function.
    If x votes were cast for one answer,
    assign it x-times higher weight.
    Negative labels have base weight.

    Args:
        labels (Tensor): labels from geowiki dataset

    Returns:
        Tensor: weights
    """
    labels[labels == 0] = 1
    weights = labels/labels.mean()

    return weights




def get_histogram_matrix(array, bins=[x/20 for x in range(0,21)], ylim=(0,30000)):

    fig = plt.figure(figsize = (12,9))
    plt.hist(array, bins=bins)
    plt.ylim(ylim)
    io_buf = io.BytesIO()
    #fig.savefig(io_buf, format='raw', bbox_inches="tight")
    fig.savefig(io_buf, format='raw')
    io_buf.seek(0)

    img_arr = np.reshape(np.frombuffer(io_buf.getvalue(), dtype=np.uint8),
                    newshape=(int(fig.bbox.bounds[3]), int(fig.bbox.bounds[2]), -1))

    io_buf.close()
    plt.close()

    return img_arr






def write_confusion_matrix(y_true, y_pred, labels, filename='/home/jan/Documents/EPFL/project_2/data/tmp/confmatrix.png', target_names=None, figsize=(10,10)):
    """
    Generate matrix plot of confusion matrix with pretty annotations.
    The plot image is saved to disk.
    args: 
      y_true:    true label of the data, with shape (nsamples,)
      y_pred:    prediction of the data, with shape (nsamples,)
      filename:  filename of figure file to save; 
      classes:    string array, name the order of class labels in the confusion matrix.
                 use `clf.classes_` if using scikit-learn models.
                 with shape (nclass,).
      class_names:      dict: any -> string, length == nclass.
                 if not None, map the labels & ys to more understandable strings.
                 Caution: original y_true, y_pred and labels must align.
      figsize:   the size of the figure plotted.
    """
    if target_names is not None:
        y_pred = [target_names[yi] for yi in y_pred]
        y_true = [target_names[yi] for yi in y_true]
        labels = [target_names[yi] for yi in labels]
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    cm_sum = np.sum(cm, axis=1, keepdims=True)
    cm_perc = cm / cm_sum.astype(float) * 100
    annot = np.empty_like(cm).astype(str)
    nrows, ncols = cm.shape
    for i in range(nrows):
        for j in range(ncols):
            c = cm[i, j]
            p = cm_perc[i, j]
            if i == j:
                s = cm_sum[i]
                annot[i, j] = '%.1f%%\n%d/%d' % (p, c, s)
            elif c == 0:
                annot[i, j] = ''
            else:
                annot[i, j] = '%.1f%%\n%d' % (p, c)
    cm = pd.DataFrame(cm, index=labels, columns=labels)
    cm.index.name = 'Actual'
    cm.columns.name = 'Predicted'
    fig, ax = plt.subplots(figsize=figsize)
    sns.heatmap(cm, annot=annot, fmt='', ax=ax)
    plt.savefig(filename)
    plt.tight_layout()
    plt.close()

    return filename




    


def parse_config_option(option, option_name):
    params = None
    if isinstance(option, dict):
        option = option[option_name]
        params = {k:v for k,v in option if k != option_name}

    return option, params



def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)



### Hansen + Vancutsem filename parsing utils

def parse_lat(latitude: str) -> tuple:
    assert ("S" in latitude or "N" in latitude), 'Unexpected name format'
    #Handle naming differences between GFC and TMF
    if latitude[0].isdigit():
        idx = 0
    else:
        idx = 1

    if "S" in latitude:
        max_lat = int(latitude.split("S")[idx]) * -1
    else:
        max_lat = int(latitude.split("N")[idx])
    min_lat = max_lat - 10

    return min_lat, max_lat

def parse_lon(longitude: str) -> tuple:
    assert ("W" in longitude or "E" in longitude), 'Unexpected name format'
    #Handle naming differences between GFC and TMF
    if longitude[0].isdigit():
        idx = 0
    else:
        idx = 1
    
    if "W" in longitude:
        max_lon = int(longitude.split("W")[idx]) * -1       
    else:
        max_lon = int(longitude.split("E")[idx])    
    min_lon = max_lon + 10

    return min_lon, max_lon

def get_extent_from_filename(filename):
    """Extract bounding box lat,lon from filename,
    requires file to conform to the naming convention
    of Global Forest Change or Tropical Moist Forest layers,
    e.g., some_name_N10_E30.tif

    Args:
        filename (str):

    Returns:
        List[float]: coords of corners of bbox
    """
    lat, lon = filename.split('_')[-2:]

    min_lat, max_lat = parse_lat(lat)
    min_lon, max_lon = parse_lon(lon)

    return min_lat, max_lat, min_lon, max_lon



def get_tiles_as_gdf(dir):
    """Iterate over files in folder, extract the lat,lon 
    extent of each tile, return them as a GeoDataFrame

    Args:
        dir (str): path to folder with either Hansen GFC or Vancutsem TMF

    Returns:
        geopandas.GeoDataFrame: tiles with geometries
    """
    shapes = []
    filenames = []
    for filename in os.listdir(dir):
        try:
        
            if not filename.endswith('.tif'):
                continue

            try:
                min_lat, max_lat, min_lon, max_lon = get_extent_from_filename(filename.split(".tif")[0])
                polygon = Polygon([(min_lon, min_lat), (min_lon, max_lat), (max_lon, max_lat, ), (max_lon, min_lat )])
            except (AssertionError, ValueError) as e: #Unexpected name format - in that case, get the extent by reading the raster
                path = os.path.join(dir, filename)
                with rasterio.open(path) as src:
                        bounds = src.bounds
                polygon = box(*bounds)
            
            shapes.append(polygon)
            filenames.append(filename)
        except Exception as e:
            print(filename,  "  ", e)

    gdf = gpd.GeoDataFrame(geometry=shapes)
    gdf['filename'] = [os.path.join(dir, filename) for filename in filenames]

    return gdf



########

        
    
def set_logger(log_path):
    """Set the logger to log info in terminal and file `log_path`.

    In general, it is useful to have a logger so that every output to the terminal is saved
    in a permanent file. Here we save it to `model_dir/train.log`.

    Example:
    ```
    logging.info("Starting training...")
    ```

    Args:
        log_path: (string) where to log
    """
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    if not logger.handlers:
        # Logging to a file
        file_handler = logging.FileHandler(log_path)
        file_handler.setFormatter(logging.Formatter('%(asctime)s:%(levelname)s: %(message)s'))
        logger.addHandler(file_handler)

        # Logging to console
        stream_handler = logging.StreamHandler()
        stream_handler.setFormatter(logging.Formatter('%(message)s'))
        logger.addHandler(stream_handler)


def save_dict_to_json(d, json_path):
    """Saves dict of floats in json file

    Args:
        d: (dict) of float-castable values (np.float, int, float, etc.)
        json_path: (string) path to json file
    """
    with open(json_path, 'w') as f:
        # We need to convert the values to float for json (it doesn't accept np.array, np.float, )
        d = {k: float(v) for k, v in d.items()}
        json.dump(d, f, indent=4)


def save_checkpoint(state, is_best, out_folder):
    """Saves model and training parameters at checkpoint + 'last.pt'. If is_best==True, also saves
    checkpoint + 'best.pt'

    Args:
        state: (dict) contains model's state_dict, may contain other keys such as epoch, optimizer state_dict
        is_best: (bool) True if it is the best model seen till now
        checkpoint: (string) folder where parameters are to be saved
    """
    filepath = os.path.join(out_folder, 'last.pt')
    if not os.path.exists(out_folder):
        print("Checkpoint directory does not exist! Creating directory {}".format(out_folder))
        os.mkdir(out_folder)

    torch.save(state, filepath)
    if is_best:
        shutil.copyfile(filepath, os.path.join(out_folder, 'best.pt'))


def load_checkpoint(checkpoint, model, optimizer=None):
    """Loads model parameters (state_dict) from file_path. If optimizer is provided, loads state_dict of
    optimizer assuming it is present in checkpoint.

    Args:
        checkpoint: (string) filename which needs to be loaded
        model: (torch.nn.Module) model for which the parameters are loaded
        optimizer: (torch.optim) optional: resume optimizer from checkpoint
    """
    if not os.path.exists(checkpoint):
        raise("File doesn't exist {}".format(checkpoint))
    checkpoint = torch.load(checkpoint)
    model.load_state_dict(checkpoint['state_dict'])

    if optimizer:
        optimizer.load_state_dict(checkpoint['optim_dict'])

    return checkpoint





### Utils for dataset curation


def get_window(geometry, raster_path):
    #Get window for windowed read of a raster
    with rasterio.open(raster_path) as src:
        src_profile = src.profile
    
    left, bottom, right, top = geometry.bounds
    

    return  rasterio.windows.from_bounds(
        left=left, 
        bottom=bottom, 
        right=right, 
        top=top,
        transform=src_profile['transform']
    )


def get_patch_geometry(i, j, transform, row_off, col_off, patch_rows, patch_cols):
    #Create a patch polygon from array positions, offset, transform and size of patch
    upper_left_coords = rasterio.transform.xy(
        transform, 
        i + row_off,
        j + col_off,
        offset='ul'
    )

    lower_right_coords = rasterio.transform.xy(
        transform,
        i + row_off + patch_rows,
        j + col_off + patch_cols,
        offset='ul'
    )
    return box(upper_left_coords[0], lower_right_coords[1], lower_right_coords[0], upper_left_coords[1])


def get_majority_value(input_array, target_values, min_ratio=0.5):
    """Return True if any of the target_values has at least min_ratio 
    values in the input array
    Example:
        input_array = np.array([0,0,1,2,2,2,2,2,2])
        target_values = [1,2]
        min_ratio = 0.5
        - returns True because out of 9 values, 6 are of one of the target values (2); that's a ratio 0.67 and 0.67 > min_ratio

    Args:
        input_array (np.array): 
        target_values (List): 
        min_ratio (float): must be at least 0.5
        
    Returns:
        int or None
    """
    values, counts = np.unique(input_array, return_counts=True)
    total_counts = np.sum(counts)
    assert min_ratio >= 0.5

    # Get counts of target values
    target_counts = counts[np.isin(values, target_values)]

    # Check each target value and find if any fulfills the condition
    majority_value = None
    for value, count in zip(values[np.isin(values, target_values)], target_counts):
        if count / total_counts >= min_ratio:
            majority_value = value
    
    return majority_value



def extract_sample(
        forestloss_data,
        forestloss_profile,
        label_data,
        i,
        j,
        patch_rows,
        patch_cols,
        target_years,
        target_values,
        min_ratio_for_driver,
        forest_loss_window,
        min_px_forest_loss):
        
    
    forestloss_patch = forestloss_data[i:i+patch_rows, j:j+patch_cols]

    forest_loss_mask = (np.isin(forestloss_patch, target_years))

    # Check if there is enough forest loss
    if not forest_loss_mask.sum() >= min_px_forest_loss:
        return None

    label_patch = label_data[i:i+patch_rows, j:j+patch_cols]

    # Check for if there is a single class that has at least min_ratio proportion
    _driver = get_majority_value(label_patch[forest_loss_mask], target_values, min_ratio_for_driver)
    if not _driver:
        return None

    # Compute bbox of patch
    _geom = get_patch_geometry(i, j, 
                               forestloss_profile['transform'], 
                               forest_loss_window.row_off, 
                               forest_loss_window.col_off,
                               patch_rows, 
                               patch_cols
                              )

    return _driver, forestloss_patch, _geom

    


    #### 


import wandb

def get_wandb_runs(filters, path="janpisl/project_2", skip_runs_with_no_data=True):
    """Fetch runs from wandb, return them as a list of dicts

    Args:
        filters (dictionary): must adhere to wandb filtering structure
        path (str, optional): path to project. Defaults to "janpisl/project_2".
        skip_runs_with_no_data (bool, optional): don't include runs that contain no training/other logs. Defaults to True.

    Returns:
        List[dict]: contains 1 dictionary for each run
    """
    

    api = wandb.Api()

    runs = api.runs(path=path, filters=filters)

    models = []

    #Iterate over all runs fitting the filters, for each create a dictionary and add to a list of models
    for i, run in enumerate(iter(runs)):
        model_dict = {}

        model_dict['config'] = run.config
        model_dict['wandb_id'] = run.id
        
        if skip_runs_with_no_data and len(run.summary.keys()) == 1: #This means there is no metrics -> killed or failed run
            continue
            
        model_dict['summary'] = run.summary
        
        name= " ".join(run.name.split())
        model_dict['name'] = name
        models.append(model_dict)


    return models



import model.dataloader as dataloader
from model.model import MultiModalDriverClassifier
from evaluate import evaluate, compute_and_log_metrics

import torch.nn as nn


def load_model_and_predict(model_dict, n_classes, device, dataset='val'):
    
    wandb_id = model_dict['wandb_id']
    model_dir = os.path.join('../models_trained', wandb_id)
    config_path = os.path.join(model_dir, 'config.json')
    with open(config_path) as source:
        config = json.loads(source.read())

    dl = dataloader.fetch_dataloader(
        [dataset], config)[dataset]

    #Load trained model
    model = MultiModalDriverClassifier(n_classes,  **config)
    model = model.to(device)

    trained_weights = f'../models_trained/{wandb_id}/best.pt'
    _ = model.load_state_dict(torch.load(trained_weights)['state_dict'])
    _ = model.eval()

    loss_fn = nn.CrossEntropyLoss(reduction='mean')

    config['wandb']['log_to_wandb'] = False

    metrics, preds, _ = evaluate(
        model=model, 
        loss_fn=loss_fn, 
        dataloader=dl, 
        device=device, 
        config=config,
        return_preds_and_targets=True
    )

    return metrics, preds





from collections import defaultdict
from typing import List, Dict, Any


def find_keys_with_multiple_values(configs: List[Dict[str, Any]], exceptions: List[str] = ['# parameters', 'val dst size', 'train dst size', 'n_parameters', 'val_dst_size', 'train_dst_size']) -> Dict[str, Any]:
    """return hyperparams from configs where the value is not identical for all configs

    Args:
        configs (List[dict]): contains all model configurations for which I want to find unique combinations
        exceptions (list, optional): items in configuration that should not be taken into account

    Returns:
        dict: contains key:value pairs, where
            key:  parameter (key in input config files) where the config files have different values 
            value: all the unique values in the input config files
    """

    def extract_values(d, prefix=''):
        for key, value in d.items():
            full_key = f"{prefix}.{key}" if prefix else key
            if isinstance(value, dict) and full_key != 'folds_to_split.mapping':
                yield from extract_values(value, full_key)
            else:
                yield full_key, value

    # Dictionary to store all values for each key
    all_values = defaultdict(set)

    # Extract values from each dictionary and populate all_values
    for item in configs:
        for key, value in extract_values(item):
            if isinstance(value, dict):
                value = tuple(sorted(value.items()))  # Convert dict to a hashable type
            all_values[key].add(value)

    # Filter keys with more than one unique value
    result = {}
    for key, values in all_values.items():
        if len(values) > 1:
            parts = key.split('.')
            current = result
            for part in parts[:-1]:
                if part not in current:
                    current[part] = {}
                current = current[part]
            current[parts[-1]] = list(values)

    # Remove exceptions
    filtered_result = {key: value for key, value in result.items() if key not in exceptions}

    return filtered_result




import itertools

def generate_combinations(config, exclude_keys):
    def flatten_dict(d, parent_key='', sep='.'):
        items = []
        for k, v in d.items():
            new_key = f"{parent_key}{sep}{k}" if parent_key else k
            if isinstance(v, dict):
                items.extend(flatten_dict(v, new_key, sep=sep).items())
            else:
                items.append((new_key, v))
        return dict(items)

    # Flatten the dictionary, excluding specified keys
    flat_config = {}
    for key, value in config.items():
        if key in exclude_keys:
            continue
        if isinstance(value, list):
            flat_config[key] = value
        else:
            nested_flat_config = flatten_dict(value, key)
            flat_config.update(nested_flat_config)

    # Generate all combinations
    keys, values = zip(*flat_config.items())
    product_combinations = itertools.product(*values)

    # Reconstruct nested dictionaries from the combinations
    def unflatten_dict(d):
        result = {}
        for key, value in d.items():
            parts = key.split('.')
            d = result
            for part in parts[:-1]:
                d = d.setdefault(part, {})
            d[parts[-1]] = value
        return result

    combinations = [unflatten_dict(dict(zip(keys, combination))) for combination in product_combinations]

    return combinations




def find_key_for_value(data, target_value):
    #by ChatGPT
    matching_keys = [key for key, value in data.items() if value == target_value]
    
    if len(matching_keys) != 1:
        raise ValueError("There is not exactly one key corresponding to the value")
    
    return matching_keys[0]



def reproject_raster(input_raster_path, output_raster_path,dst_crs='EPSG:4326'):
    """
    Reprojects a raster 

    Parameters:
    input_raster_path (str): Path to the input raster file.
    output_raster_path (str): Path to the output raster file.
    """
    # Open the input raster
    with rasterio.open(input_raster_path) as src:
        # Get the original CRS
        src_crs = src.crs

        
        # Calculate the transform and new dimensions
        transform, width, height = calculate_default_transform(
            src_crs, dst_crs, src.width, src.height, *src.bounds)
        
        # Define the metadata for the output raster
        kwargs = src.meta.copy()
        kwargs.update({
            'crs': dst_crs,
            'transform': transform,
            'width': width,
            'height': height
        })
        
        # Create the output raster and reproject the data
        with rasterio.open(output_raster_path, 'w', **kwargs) as dst:
            for i in range(1, src.count + 1):
                reproject(
                    source=rasterio.band(src, i),
                    destination=rasterio.band(dst, i),
                    src_transform=src.transform,
                    src_crs=src_crs,
                    dst_transform=transform,
                    dst_crs=dst_crs,
                    compress='lzw',
                    resampling=Resampling.nearest)

