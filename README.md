# Mapping Post-Deforestation Land Uses with Location-Aware Deep Learning

This repository contains the implementation of a deep learning model designed for mapping post-deforestation land uses across tropical regions. It recognizes eleven classes, including seven major forest-risk commodities (beef, soy, palm oil, rubber, cocoa, coffee, and wood fiber), as well as mining, subsistence agriculture, urban expansion; other land uses are classified as 'other'.


## Model Architecture

- **Satellite Imagery:** Sentinel-2 time series data (RGB + Near-Infrared, 10m resolution).
- **Location Encoding:** Geographic coordinates integrated through positional embeddings.
- **Tabular Data:** Country-level statistics detailing agricultural commodity-driven deforestation.
- **Multi-modal Fusion:** Transformer-based architecture integrating image, location, and tabular data.

## PLUTo Dataset

- Compiled from 19 publicly available datasets.
- Covers 11 land-use classes across over 60 tropical countries.
- Accessible publicly [here](https://zenodo.org/records/15629667).

## Key Results

- Inclusion of location-related modalities increases classification accuracy by approximately 10%, depending on the size of the training dataset.

For any questions or additional information, please contact:

Jan Pišl
jan.pisl@epfl.ch
