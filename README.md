# Mapping Post-Deforestation Land Uses with Location-Aware Deep Learning

This repository contains the implementation of a deep learning model designed for mapping post-deforestation land uses across tropical regions. It recognizes eleven classes, including seven major forest-risk commodities (beef, soy, palm oil, rubber, cocoa, coffee, and wood fiber), as well as mining, subsistence agriculture, urban expansion; other land uses are classified as 'other'.


## Model inputs

- **Satellite image time series:** Four Sentinel-2 images (RGB + Near-Infrared).
- **Location:** Geographic coordinates of the deforestation event.
- **Tabular Data:** Country-level statistics estimating the attribution of deforestation to agricultural commodities

## Post-deforestation Land Use in the Tropics (PLUTo) dataset

- Compiled from 19 publicly available datasets.
- 11 land-use classes across the tropics
- Public access [here](https://zenodo.org/records/15629667).

## Key Results

- Including location-related modalities increases classification accuracy by approximately 10%, depending on the size of the training dataset.

For any questions or additional information, please contact:

Jan Pišl
jan.pisl@epfl.ch
