# Mapping Post-Deforestation Land Uses with Location-Aware Deep Learning

Code for the paper [*Mapping land uses following tropical deforestation with
location-aware deep learning*](https://www.sciencedirect.com/science/article/pii/S0924271625004915)
(Pišl et al., ISPRS Journal of Photogrammetry and Remote Sensing, 2026).

A multi-modal Transformer that classifies post-deforestation land use from a Sentinel-2
image time series, geographic coordinates, and country-level deforestation statistics.
See the paper for
the method and results.

## Contents

- [Repository structure](#repository-structure)
- [Installation](#installation)
- [The PLUTo dataset](#the-pluto-dataset)
- [Data setup](#data-setup)
- [Training](#training)
- [Evaluation](#evaluation)
- [Model and configuration](#model-and-configuration)
- [Citation](#citation)
- [Contact](#contact)

## Repository structure

```
.
├── train.py                 Training entry point; reads a JSON config
├── evaluate.py              Evaluation loop and metrics
├── utils.py                 Checkpointing, seeding, config parsing, confusion matrices
├── extract_patches.py       Extracts deforested patches from TMF and assigns labels
├── download_images.py       Downloads the Sentinel-2 time series per patch
├── requirements.txt
├── config/
│   └── example.json         Example config (the final model in the paper)
└── model/
    ├── model.py             The multi-modal model
    ├── model_blocks.py      Image encoder, tabular encoder, Transformer classifier
    ├── loc_encoder.py       Location encoder (positional embedding + network)
    ├── positional_embeddings.py
    ├── siren.py
    ├── dataset.py           Loads images, location, tabular data, and deforestation mask
    ├── dataloader.py        Dataloaders, transforms, and train/val/test splitting
    └── loss.py              Cross-entropy and tabular regularization loss
```

## Installation

Requires Python 3 and PyTorch.

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

Training runs on GPU if available and falls back to CPU otherwise.

## The PLUTo dataset

Training and evaluation use the Post-deforestation Land Use in the Tropics (PLUTo)
dataset, released separately at:

**https://zenodo.org/records/17831353**

Each sample covers a 1 km² deforested patch and consists of:

- a time series of four Sentinel-2 images (RGB + near-infrared, 10 m resolution),
- a deforestation mask from the TMF dataset,
- the latitude/longitude of the patch centre,
- 11 values from the DeDuCE dataset (one per class), the deforestation attributed to each
  class for the patch's country,
- a land-use class label (one of eleven classes).

The split into train, validation, and test folds is geographic. See the paper for the
dataset construction, class list, and statistics.

## Data setup

Dataset paths are read from the `data` section of the config. After downloading PLUTo,
edit them to point to your local copy:

```json
"data": {
    "annotation_file":   "/path/to/labels.csv",
    "image_folder":      "/path/to/four_images",
    "forestloss_folder": "/path/to/forestloss_tiles",
    "deduce_data":       "/path/to/deduce_yearly.csv",
    "output_folder":     "models_trained/"
}
```

- `annotation_file` — CSV with one row per sample. Columns: `sampleid`, `driver` (label),
  `geometry` (WKT point), `country`, the fold column (`split`), and `dst_size`.
- `image_folder` — one subfolder per `sampleid`, each holding the GeoTIFFs of the time
  series.
- `forestloss_folder` — one `{sampleid}.tif` per sample with the TMF deforestation mask.
- `deduce_data` — the country-level DeDuCE table.

`extract_patches.py` and `download_images.py` rebuild the dataset from scratch and are
not needed when using the released PLUTo dataset.

## Training

```bash
python train.py config/example.json
```

Checkpoints and the resolved config are written to `output_folder/<run_id>/`. `last.pt`
holds the most recent epoch; `best.pt` holds the highest validation-fold accuracy.

**Hyperparameter grids.** Every value under `hyperparams` is a list. `train.py` runs the
Cartesian product of these lists, one experiment per combination; this produced the
ablations and dataset-size sweeps in the paper. For a single run, give each field a
one-element list, as in `config/example.json` (which only varies the random seed).
Errors in one experiment are written to `logs_<date>.txt` and the remaining experiments
continue.

**Logging.** Weights & Biases logging is set in the `wandb` section. Set
`"log_to_wandb": "False"` to disable it, otherwise set `project` and `entity`.

## Evaluation

`evaluate.py` contains the evaluation loop (`evaluate`) and metric computation
(`compute_and_log_metrics`), reporting accuracy and per-class F1. During training the
model is evaluated on the validation fold after every epoch. To evaluate a trained
checkpoint on another split, load it with `utils.load_checkpoint`, build the dataloader
with `model.dataloader.fetch_dataloader`, and call `evaluate`. The split is selected by
`folds_to_split.mapping` in the config, which assigns each fold to `train`, `val`, or
`test`.

## Model and configuration

The model (`model/model.py`) encodes each modality with its own encoder and fuses them
with a Transformer. Each modality can be switched on or off, and each encoder is
configurable:

| Modality | Config key | Options |
|---|---|---|
| Image time series | `image_encoder` | `backbone` (`resnet18`), `pretrained_backbone`, `freeze_after` (epoch to freeze the backbone) |
| Deforestation mask | `use_forestloss`, `forestloss_params` | `kernel_size`, `sigma`, `apply_after_layer` |
| Location | `loc_encoder` | `positional_embedding` (`direct`, `wrap`, `cartesian3d`, `theory`), `nn` (`linear`, `fcnet`, `siren`) |
| Tabular | `tabular_encoder` | `dataset` (`deduce`), `type` (`token`), `temporal_processing` (`static`, `dynamic`) |
| Fusion | `classifier` | `type` (`transformer`, `fc`), `transformer_layers`, `attention_heads` |

The configuration of the final model from the paper is in `config/example.json`.

## Citation

```bibtex
@article{pisl2026mapping,
  title   = {Mapping land uses following tropical deforestation with location-aware deep learning},
  author  = {Pi{\v{s}}l, Jan and Sumbul, Gencer and Lenczner, Gaston and Zamora, Camilo and Herold, Martin and Wegner, Jan Dirk and Tuia, Devis},
  journal = {ISPRS Journal of Photogrammetry and Remote Sensing},
  volume  = {232},
  pages   = {578--593},
  year    = {2026},
  doi     = {10.1016/j.isprsjprs.2025.12.007}
}
```

## Contact

Jan Pišl — jan.pisl@epfl.ch
