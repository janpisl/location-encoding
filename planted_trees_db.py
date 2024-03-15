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

#gdfs = {}

cocoa_gdfs = []
coffee_gdfs = []
rubber_gdfs = []


i = 0


while i < len(layers):
    
    layer = layers[i]
    i += 1
    
    gdf = gpd.read_file(gdb_file,layer=layer)
    print(layer)
    cocoa = gdf[gdf['originalName'].str.lower().str.contains('caca', na=False)]
    coffee = gdf[gdf['originalName'].str.lower().str.contains('coffe', na=False)]
    
    if layer in ['idn_plant_v2', 'tha_plant_v2', 'vnm_plant_v2', 'chn_plant_v2', 'mli_plant_v2', 'mmr_plant_v2', 'lao_plant_v2']: #For southeast Asia I already have data
        #Indonesia  Thailand  Vietnam China Malaysia  Myanmar ??Cambodia?? - didnt find cambodia  Laos 
        rubber = pd.DataFrame()
    else:
        rubber = gdf[gdf['originalName'].str.lower().str.contains('rubber', na=False)]

    if not cocoa.empty:
        cocoa_gdfs.append(cocoa)
    if not coffee.empty:
        coffee_gdfs.append(coffee)
    if not rubber.empty:
        rubber_gdfs.append(rubber)

    if i % 10 == 0:
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

        try:
            rubber = pd.concat(rubber_gdfs)
            rubber = gpd.GeoDataFrame(rubber)
            rubber.to_file('data/tmp/sdpt/rubber.gpkg')
        except ValueError:
            pass