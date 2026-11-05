# FAS-DiT

Training and inference code for **FAS-DiT: Frequency-Adaptive Structure-Aware
Diffusion Transformer for Medical Image Segmentation**.

## Installation

```bash
conda env create -f environment.yaml
conda activate fas-dit
```

Alternatively, with PyTorch already installed:

```bash
pip install -r requirements.txt
```

Run the following commands from the repository root.

## Data preparation

Datasets and trained checkpoints are not included. Obtain the data separately
and prepare the directories below.

### Binary segmentation

| Dataset | Directory | Source |
| --- | --- | --- |
| GlaS | `processed_glas/` | [GlaS Challenge](https://warwick.ac.uk/fac/cross_fac/tia/data/glascontest/) |
| MoNuSeg | `processed_monuseg/` | [MoNuSeg Challenge](https://monuseg.grand-challenge.org/Data/) |
| PH2 | `processed_ph2/` | [PH2 Database](https://www.fc.up.pt/addi/ph2%20database.html) |
| IMID | `processed_imid/` | Private dataset |
| TNBC | `processed_tnbc/` | [TNBC Dataset](https://zenodo.org/records/1174343) |

Each directory uses this structure:

```text
processed_glas/
├── train/
│   ├── images/
│   └── masks/
├── val/
│   ├── images/
│   └── masks/
└── test/
    ├── images/
    └── masks/
```

Match each image and mask by filename stem. Use single-channel binary masks
with background 0 and foreground 255; merge instance annotations into binary
foreground masks. Supported formats are PNG, JPEG, BMP and TIFF. The loader
resizes images and masks to the dataset's configured input size.

Prepare fixed train/validation/test splits before training. The paper uses a
6:2:2 split, with patient-level separation for GlaS. TNBC is used only for
cross-dataset evaluation and requires only `test/images/` and `test/masks/`.

### Synapse

Use the preprocessed TransUNet Synapse data (18 training cases and 12 test
volumes; background and 8 organ classes):

```text
Synapse/
├── train_npz/caseXXXX_sliceYYY.npz
└── test_vol_h5/caseXXXX.npy.h5
```

Generate the case-level split file:

```bash
python prepare_synapse_split.py
```

This writes `Synapse/split.json` without a validation set. To reserve validation
cases, use `python prepare_synapse_split.py --val_cases 4` instead.

## Training

```bash
python train.py --dataset glas
```

Training supports `glas`, `ph2`, `imid`, `monuseg` and `synapse`. Dataset defaults
are defined in `_DATASET_DEFAULTS` in `train.py`.

Checkpoints are saved to `checkpoints/<dataset>_<model>_<timestamp>/` and logs to
`logs/`. For binary datasets, evaluate `best_model.pth`. Under the default
Synapse split, training has no early stopping or validation-based model
selection; evaluate `final_model.pth`.

To resume training:

```bash
python train.py --dataset glas --resume checkpoints/<run>/checkpoint_epoch100.pth
```

## Inference

```bash
python inference.py \
    --checkpoint checkpoints/<run>/best_model.pth \
    --dataset glas --use_ema --K 25 --save_images
```

Metrics are saved to `results/<dataset>/metrics.txt`. Use `--save_images` to save
predictions and comparison images. `--K` controls the number of averaged sampling
trajectories (25 for the paper setting; 1 for faster inference), and
`--ddim_steps` controls the sampling steps (default: 20).

For Synapse volume-level evaluation:

```bash
python inference.py \
    --checkpoint checkpoints/<synapse-run>/final_model.pth \
    --dataset synapse --use_ema --K 25
```

For MoNuSeg-to-TNBC evaluation without fine-tuning:

```bash
python inference.py \
    --checkpoint checkpoints/<monuseg-run>/best_model.pth \
    --dataset tnbc --use_ema --K 25
```

See `python train.py --help` and `python inference.py --help` for all options.

## License

[MIT License](LICENSE).
