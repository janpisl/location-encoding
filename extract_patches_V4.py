#!/usr/bin/env python
# coding: utf-8

import os
import copy
from pathlib import Path

import rasterio
import numpy as np
import pandas as pd
import geopandas as gpd
from shapely.geometry import box
from rasterio.enums import Resampling

from utils import get_tiles_as_gdf, extract_sample, get_window



def load_dataset(dataset_name):
    if dataset_name == 'rubber':
        rubber_data_path = '/home/jan/Documents/EPFL/project_2/data/input_datasets/rubber/Rubber_10m'
        label_gdf = get_tiles_as_gdf(rubber_data_path)
        dataset_extent = box(*label_gdf.total_bounds)
        class_mapping = {
            1 : 'rubber'
        }
        dataset_prefix = 20000000

        

    elif dataset_name == 'mapbiomas':
        mapbiomas_path = '/home/jan/Documents/EPFL/drivers/data/mapbiomas/brasil_coverage_2022.tif'
        with rasterio.open(mapbiomas_path) as src:
            dataset_extent = box(*src.bounds)
        label_gdf = gpd.GeoDataFrame({'geometry' : [dataset_extent], 'filename': [mapbiomas_path]})
        
        class_mapping = {
            #1 : 'forest', # superclass
            #3 : 'forest',  # "Forest Formation"
            #10 : 'non-forest natural formation', #incl. wetland, grassland etc. - I think this is the 'natural regrowth'

            # Commodities & drivers
            9 : 'forest plantation',
            15 : 'pasture', 
            39 : 'soy',
            46 : 'coffee',
            35 : 'palm oil',
            30 : 'mining',
            24 : 'urban',
            
            32 : 'hypersaline tidal flat',
            29 : 'rocky outcrop', 
            20 : 'sugar cane', 
            62 : 'cotton',
            40 : 'rice', 
            41 : 'other temporary crops',
            47 : 'citrus', 
            23 : 'beach, dune, snad', 
            33 : 'river, lake, ocean',
            31 : 'aquaculture' 
            
        }
        
        dataset_prefix = 10000000
        

    elif dataset_name == 'cocoa':
        cocoa_file_path = '/home/jan/Documents/EPFL/project_2/data/input_datasets/cocoa_map_th_090.tif'
        with rasterio.open(cocoa_file_path) as src:
            dataset_extent = box(*src.bounds)
        label_gdf = gpd.GeoDataFrame({'geometry' : [dataset_extent], 'filename': [cocoa_file_path]})
        
        class_mapping = {
            1 : 'cocoa'
        }
        dataset_prefix = 30000000


    elif dataset_name == 'vancutsem_regrowth':
        regrowth_data_path = 'data/input_datasets/vancutsem_regrowth/'
        label_gdf = get_tiles_as_gdf(regrowth_data_path)
        dataset_extent = box(*label_gdf.total_bounds)
        class_mapping = {
            30 : 'forest regrowth'
        }
        dataset_prefix = 40000000
        
        
        
    elif dataset_name == 'laos_shifting_agriculture':
        dataset_prefix = 50000000
        
        laos_shifting_ag_data_path = 'data/input_datasets/laos_dis_2020.tif'
        with rasterio.open(laos_shifting_ag_data_path) as src:
            dataset_extent = box(*src.bounds)
        label_gdf = gpd.GeoDataFrame({'geometry' : [dataset_extent], 'filename': [laos_shifting_ag_data_path]})
        
        class_mapping = {
            8 : 'shifting agriculture'
        }
        

    elif dataset_name == 'coffee_maskell':
        dataset_prefix = 70000000
        
        vietnam_coffee_data_path = 'data/input_datasets/coffee_MasketEtAl/classified_binary_2020Oct.tif'
        with rasterio.open(vietnam_coffee_data_path) as src:
            dataset_extent = box(*src.bounds)
        label_gdf = gpd.GeoDataFrame({'geometry' : [dataset_extent], 'filename': [vietnam_coffee_data_path]})
         
        class_mapping = {
            1 : 'coffee'
        }

    elif dataset_name == 'mapbiomas_indonesia':
        dataset_prefix = 80000000
        
        mapb_indo = 'data/input_datasets/mapbiomas_indonesia/indonesia_coverage_2022.tif'
        with rasterio.open(mapb_indo) as src:
            dataset_extent = box(*src.bounds)
        label_gdf = gpd.GeoDataFrame({'geometry' : [dataset_extent], 'filename': [mapb_indo]})
        
        class_mapping = {
            9 : 'forest plantation',
            35 : 'palm oil',
            30 : 'mining',
            26 : 'water', #water
            40 : 'rice', #rice
            #21 : 'other' #other agriculture #this possibly contains cocoa so not included
        }    

        
        
    elif dataset == 'palm_oil':
        dataset_prefix = 90000000
        palm_oil_folder = 'data/input_datasets/oil_plantation/High_resolution_global_industrial_and_smallholder_oil_palm_map_for_2019/oil_palm_map'
        
        label_gdf = get_tiles_as_gdf(palm_oil_folder)
        dataset_extent = box(*label_gdf.total_bounds)
        class_mapping = {
            1 : 'palm oil',
            2 : 'palm oil'
        }



    elif dataset == 'worldcover':
        dataset_prefix = 100000000
        worldcover_folder = 'data/input_datasets/worldcover/'
        
        label_gdf = get_tiles_as_gdf(worldcover_folder)
        dataset_extent = box(*label_gdf.total_bounds)
        class_mapping = {
            50 : 'urban',
        }


    elif dataset == 'cocoa_ecuador':
        dataset_prefix = 110000000
        cocoa_ecuador_folder = 'data/input_datasets/cocoa_ecuador'
        
        label_gdf = get_tiles_as_gdf(cocoa_ecuador_folder)
        dataset_extent = box(*label_gdf.total_bounds)
        class_mapping = {
            1 : 'cocoa',
        }

    elif dataset == 'cofee_sdpt_brazil':
        dataset_prefix = 120000000
        coffee_sdpt_folder = 'data/input_datasets/sdpt_coffee_brazil'
        
        label_gdf = get_tiles_as_gdf(coffee_sdpt_folder)
        dataset_extent = box(*label_gdf.total_bounds)
        class_mapping = {
            1 : 'coffee',
        }

    elif dataset == 'tyukavina_congo_basin_agri':
        dataset_prefix = 130000000
        tyukavina_congo_folder = 'data/input_datasets/tyukavina_congo_basin_agri'
        
        label_gdf = get_tiles_as_gdf(tyukavina_congo_folder)
        dataset_extent = box(*label_gdf.total_bounds)
        class_mapping = {
            1 : 'shifting agriculture',
        }


    elif dataset == 'mines':
        dataset_prefix = 140000000
        mines_folder = 'data/input_datasets/mines'
        
        label_gdf = get_tiles_as_gdf(mines_folder)
        dataset_extent = box(*label_gdf.total_bounds)
        class_mapping = {
            1 : 'mining',
        }


    elif dataset == 'soy_south_america':
        dataset_prefix = 150000000
        soy_folder = 'data/input_datasets/soy_south_america'
        
        label_gdf = get_tiles_as_gdf(soy_folder)
        dataset_extent = box(*label_gdf.total_bounds)
        class_mapping = {
            1 : 'soy',
        }


    elif dataset == 'mapbiomas_peru':
        dataset_prefix = 160000000
        mapbiomas_peru_path = 'data/input_datasets/peru_coverage_2022.tif'

        with rasterio.open(mapbiomas_peru_path) as src:
            dataset_extent = box(*src.bounds)
        label_gdf = gpd.GeoDataFrame({'geometry' : [dataset_extent], 'filename': [mapbiomas_peru_path]})
        
        class_mapping = {
            9 : 'forest plantation',
            15 : 'pasture', 
            39 : 'soy',
            35 : 'palm oil',
            30 : 'mining',
            24 : 'urban',
            32 : 'hypersaline tidal flat',
            33 : 'river, lake, ocean',
            31 : 'aquaculture' 
            
        }
        
    elif dataset == 'cocoa_peru_points':
        dataset_prefix = 170000000
        cocoa_peru_path = 'data/input_datasets/cocoa_peru.tif.tif'


        with rasterio.open(cocoa_peru_path) as src:
            dataset_extent = box(*src.bounds)
        label_gdf = gpd.GeoDataFrame({'geometry' : [dataset_extent], 'filename': [cocoa_peru_path]})
        
        class_mapping = {
            1 : 'cocoa',           
        }


    elif dataset == 'mapbiomas_bolivia':
        dataset_prefix = 180000000
        mapbiomas_peru_path = 'data/input_datasets/bolivia_coverage_2021.tif'

        with rasterio.open(mapbiomas_peru_path) as src:
            dataset_extent = box(*src.bounds)
        label_gdf = gpd.GeoDataFrame({'geometry' : [dataset_extent], 'filename': [mapbiomas_peru_path]})
        
        class_mapping = {
            15 : 'pasture', 
            30 : 'mining',
            24 : 'urban',
            33 : 'river, lake, ocean',
            
        }


    elif dataset == 'manual_annotation':
        dataset_prefix = 190000000
        manual_annotation = 'data/input_datasets/manual_annotation/rasters'
        
        label_gdf = get_tiles_as_gdf(manual_annotation)
        dataset_extent = box(*label_gdf.total_bounds)
        class_mapping = {
            1 : 'coffee',
            2 : 'cocoa',
            3 : 'rubber'
        }

    elif dataset == 'africa_masolele':
        dataset_prefix = 210000000
        folders_masolele_rasters = 'data/input_datasets/Africa_Masolele/rasters'
        
        label_gdf = get_tiles_as_gdf(folders_masolele_rasters)
        dataset_extent = box(*label_gdf.total_bounds)
        class_mapping = {
            2 : 'pasture',
            4 : 'shifting agriculture',
            5 : 'forest plantation',
            9 : 'urban',
            11 : 'other',
            12 : 'oil palm',
            13 : 'rubber',
            16 : 'urban'

        }

    return dataset_prefix, class_mapping, label_gdf, dataset_extent





def write_cocoa_map_with_threshold(_cocoa_file_path, _cocoa_file_path_65, _threshold, _output_path):
    '''
    cocoa_file_path_65 = '/home/jan/Documents/EPFL/project_2/data/input_datasets/cocoa_map_th_065.tif'
    cocoa_file_path = '/home/jan/Documents/EPFL/project_2/data/input_datasets/cocoa_map.tif'
    out_path = cocoa_file_path_65.replace('065', '090')
    '''
    
    with rasterio.open(_cocoa_file_path) as src:
        data = src.read()   

        data[np.where((data > 0) & (data < _threshold))] = 0.

        data[data >= _threshold] = 1.

        data[data < 0] = np.nan
    
    with rasterio.open(_cocoa_file_path_65) as src:
        profile = src.profile
        
    with rasterio.open(_output_path, "w", **profile) as sink:
        sink.write(data)



def write_numpy_array_to_tif(array, geometry, profile, patch_size, output_path):
    
    transform = rasterio.transform.from_bounds(
        geometry.bounds[0], 
        geometry.bounds[1], 
        geometry.bounds[2], 
        geometry.bounds[3], 
        patch_size[0], 
        patch_size[1])
    
    out_profile = copy.deepcopy(profile)

    out_profile.update({'transform': transform,
                    'width' : patch_size[0],
                    'height' : patch_size[1]
                })
    
    with rasterio.open(output_path, "w", **out_profile) as sink:
        sink.write(array, 1)


def extract_patches(
        label_gdf, 
        intersecting_forestloss, 
        dataset_prefix,
        target_values,
        target_years,
        patch_rows,
        patch_cols,
        min_px_forest_loss,
        min_ratio_for_driver
        ):


    geoms = []
    drivers = []
    sampleids = [] 
        
        
    sampleid = dataset_prefix

    for idx1, label_tile in label_gdf.iterrows(): #Can be a single one (cocoa, mapb) or multiple (rubber)
        print(idx1)
        forestloss_intersection = intersecting_forestloss.loc[intersecting_forestloss.intersects(label_tile.geometry)]
        
        for idx2, forest_loss_tile in forestloss_intersection.iterrows():
            #print(idx2)
            intersection = label_tile.geometry.intersection(forest_loss_tile.geometry)
            label_window = get_window(intersection, label_tile.filename)
            forest_loss_window = get_window(intersection, forest_loss_tile.filename)

            
            with rasterio.open(forest_loss_tile.filename) as src:
                forestloss_data = src.read(1, window=forest_loss_window)
                forestloss_profile = src.profile
            
            with rasterio.open(label_tile.filename) as src:
                label_data = src.read(1,           
                    out_shape=(
                        forestloss_data.shape[0],
                        forestloss_data.shape[1]  
                    ),
                resampling=Resampling.bilinear,
                    window=label_window)

                
            assert forestloss_data.shape == label_data.shape
            
            for i in range(0, forestloss_data.shape[0], patch_rows):
                for j in range(0, forestloss_data.shape[1], patch_cols):
                    

                    sample = extract_sample(
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
                                min_px_forest_loss)
                    
                    
                    if sample: #If conditions are not met, it returns None; if not, it contains the three items
                        driver_class, forestloss, patch_geometry = sample
                        drivers.append(int(driver_class))

                        geoms.append(patch_geometry)
                        sampleids.append(sampleid)
                        sampleid += 1

            if len(label_gdf) == 1: #For e.g., MapBiomas, i want prints per TMF tile 
                print(f"Items: {len(geoms)}")
        
        print(f"Items: {len(geoms)}")

    return geoms, drivers, sampleids




def main(
        dataset, 
        target_years, 
        forestloss_tiles, 
        output_dir,
        patch_rows,
        patch_cols,
        min_px_forest_loss,
        min_ratio_for_driver
        ):
    
    print(f'Processing dataset: {dataset}')

    dataset_prefix, class_mapping, label_gdf, dataset_extent = load_dataset(dataset_name=dataset)

    target_values = [key for key in class_mapping.keys()]

    filtered_forestloss = forestloss_tiles.loc[forestloss_tiles.intersects(dataset_extent)]

    geoms, drivers, sampleids = extract_patches(
        label_gdf,
        filtered_forestloss,
        dataset_prefix,
        target_values,
        target_years,
        patch_rows=patch_rows,
        patch_cols=patch_cols,
        min_px_forest_loss=min_px_forest_loss,
        min_ratio_for_driver=min_ratio_for_driver
    )

    gdf = gpd.GeoDataFrame(geometry=geoms)


    gdf['driver'] = [class_mapping[val] for val in drivers]
    gdf['sampleid'] = sampleids

    #forest_loss_years_list = [ (item_tuple[0].tolist(),item_tuple[1].tolist()) for item_tuple in forest_loss_years]
    #gdf['forest_loss_years'] = forest_loss_years_list

    gdf['geometry'] = gdf['geometry'].centroid

    df = pd.DataFrame(gdf)

    df.to_csv(os.path.join(output_dir, f'{dataset}.csv'), index=False)

    gdf.to_file(f'data/tmp/dst_v5/{dataset}.gpkg')






if __name__ == '__main__':

    print('starting')
    target_years = list(range(2010, 2020))

    patch_rows, patch_cols = 33, 33 #as close as possible to 1km square
    min_px_forest_loss = 15 #Pixel size is 30x30m;
    min_ratio_for_driver = 0.5 # E.g., for patch with 100 pixels of forest loss, at least min_ratio_for_driver*100 must be of the main driver class


    forestloss_dir = '../Vancutsem_yearloss'
    forestloss_tiles = get_tiles_as_gdf(forestloss_dir)
    output_dir = 'data/labels/dataset_v5'
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    datasets = [
        'rubber',
        'cocoa',
        'mapbiomas',
        'vancutsem_regrowth',
        #'laos_shifting_agriculture',
        'coffee_maskell',
        'mapbiomas_indonesia',
        'palm_oil', 
        'worldcover',
        'cofee_sdpt_brazil', 
        'cocoa_ecuador',
        'tyukavina_congo_basin_agri',
        'mines',
        'soy_south_america',
        'mapbiomas_peru',
        'cocoa_peru_points',
        'manual_annotation',
        'mapbiomas_bolivia',
        'africa_masolele'
        ]


    for dataset in datasets:
        main(dataset, target_years, forestloss_tiles, output_dir, patch_rows, patch_cols, min_px_forest_loss, min_ratio_for_driver)