#!/usr/bin/env python
# coding: utf-8



import fiona 
import geopandas as gpd
import os 
import pandas as pd
import numpy as np


gdb_file = '/home/jan/Documents/EPFL/project_2/data/input_datasets/sdpt_v2_v11282023.gdb/sdpt_v2_v11282023.gdb'
# Get all the layers from the .gdb file 
layers = fiona.listlayers(gdb_file)

gdfs = {}

cocoa_gdfs = []
coffee_gdfs = []


i = 0


while i < len(layers):
    
    layer = layers[i]
    i += 1
    
    gdf = gpd.read_file(gdb_file,layer=layer)
    gdfs[layer] = gdf
    print(layer)
    cocoa = gdf[gdf['originalName'].str.lower().str.contains('caca', na=False)]
    coffee = gdf[gdf['originalName'].str.lower().str.contains('coffe', na=False)]

    if not cocoa.empty:
        cocoa_gdfs.append(cocoa)
    if not coffee.empty:
        coffee_gdfs.append(coffee)
    

try:
    cocoa = pd.concat(cocoa_gdfs)
    cocoa = gpd.GeoDataFrame(cocoa)
    cocoa.to_file('data/tmp/sdpt/cocoa.gpkg')
except ValueError:
    pass

try:
    coffee = pd.concat(coffee_gdfs)
    coffee = gpd.GeoDataFrame(coffee)
    coffee.to_file('data/tmp/sdpt/coffee.gpkg')
except ValueError:
    pass
out_dir = '/home/jan/Documents/EPFL/project_2/data/tmp/sdpt'

