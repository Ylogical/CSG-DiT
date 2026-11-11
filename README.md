# CSG-DiT

Training and inference code for **CSG-DiT: Frequency-Decoupled Condition and
Uncertainty-Aware State Guidance in Diffusion Transformers for Medical Image
Segmentation**.

## Installation

```bash
conda env create -f environment.yaml
conda activate csg-dit
```

Alternatively, with PyTorch already installed:

```bash
pip install -r requirements.txt
```

Run the following commands from the repository root.

The image condition stream uses a frozen, pretrained DINOv2 ViT-B/14. The first
training run loads the official implementation and weights through PyTorch Hub.
For offline use, pass a checkout of the
[official DINOv2 repository](https://github.com/facebookresearch/dinov2) and its
weights:

```bash
python train.py --dataset glas     --dino_repo /path/to/dinov2     --dino_weights /path/to/dinov2_vitb14_pretrain.pth
```

## Repository structure

```text
CSG-DiT/
├── train.py            # training entry point (per-dataset defaults in _DATASET_DEFAULTS)
├── inference.py        # sampling and evaluation entry point
└── csg_dit/
    ├── model.py        # CSG-DiT network (HCSE, DiT blocks, DFCA/PBDF, UBAM, decoder)
    ├── diffusion.py    # noise schedule, DDIM sampling and training loss
    ├── data.py         # dataset loaders and the Synapse split generator
    ├── metrics.py      # IoU/DSC/HD95/GED/ECE, slice-level metrics, Params/GFLOPs
    └── utils.py        # EMA, early stopping, checkpoints, LR schedule, logger
```

## Data preparation

Datasets and trained checkpoints are not included. Obtain the data separately
and prepare the directories below.

| Dataset | Directory            | Source                                                                                                                                    |
| ------- | -------------------- | ----------------------------------------------------------------------------------------------------------------------------------------- |
| Synapse | `Synapse/`           | [Synapse BTCV](https://www.synapse.org/#!Synapse:syn3193805/wiki/217789), preprocessed by [TransUNet](https://github.com/Beckschen/TransUNet) |
| PH2     | `processed_ph2/`     | [PH2 Database](<https://www.fc.up.pt/addi/ph2%20database.html>)                                                                            |
| GlaS    | `processed_glas/`    | [GlaS Challenge](https://warwick.ac.uk/fac/cross_fac/tia/data/glascontest/)                                                                |
| MoNuSeg | `processed_monuseg/` | [MoNuSeg Challenge](https://monuseg.grand-challenge.org/Data/)                                                                             |
| IMID    | `processed_imid/`    | Private dataset                                                                                                                           |
| TNBC    | `processed_tnbc/`    | [TNBC Dataset](https://zenodo.org/records/1174343)                                                                                         |

PH2, GlaS, MoNuSeg, IMID and TNBC use this structure:

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

Match each image and mask by filename stem. Use single-channel masks with
background 0 and foreground 255; merge instance annotations into foreground
masks. Supported formats are PNG, JPEG, BMP and TIFF. The loader resizes images
and masks to the dataset's configured input size.

Prepare fixed train/validation/test splits before training. The paper uses a
6:2:2 split, with patient-level separation for GlaS. TNBC is used only for
cross-dataset evaluation and requires only `test/images/` and `test/masks/`.

Synapse uses the TransUNet files (18 training cases and 12 test volumes;
background and 8 organ classes):

```text
Synapse/
├── train_npz/caseXXXX_sliceYYY.npz
└── test_vol_h5/caseXXXX.npy.h5
```

Generate its case-level split file:

```bash
python -m csg_dit.data
```

This writes `Synapse/split.json` without a validation set. To reserve validation
cases, use `python -m csg_dit.data --val_cases 4` instead.

## Training

```bash
python train.py --dataset glas
```

This trains the complete CSG-DiT model with the settings of the paper.
Training supports `glas`, `ph2`, `imid`, `monuseg` and `synapse`; per-dataset
settings are in `_DATASET_DEFAULTS` and shared settings in `_COMMON` in
`train.py`. The command line only exposes common options: `--batch_size`,
`--epochs`, `--lr`, `--gpu`, `--num_workers`, `--seed`, `--resume`,
`--save_dir`, `--log_dir`, `--dfca_kv_mode`, `--dino_repo` and `--dino_weights`.

`--dfca_kv_mode` selects how DFCA builds its keys. The default, `split`,
follows the paper: each frequency band provides its own keys and values.
With `v_only`, both streams share one key projection of the unfiltered image
tokens, and the frequency bands provide only the values:

```bash
python train.py --dataset glas --dfca_kv_mode v_only
```

Inference detects the mode from the checkpoint.

Checkpoints are saved to `checkpoints/<dataset>_<model>_<timestamp>/` and logs to
`logs/`. A checkpoint stores the full model state, including the frozen DINOv2
weights. For binary datasets, evaluate `best_model.pth`. Under the default
Synapse split, training has no early stopping or validation-based model
selection; evaluate `final_model.pth`.

To train on several GPUs, launch with `torchrun`; the batch size stays global
and is split evenly over the GPUs:

```bash
torchrun --nproc_per_node=2 train.py --dataset glas
```

To resume training from the latest checkpoint:

```bash
python train.py --dataset glas --resume checkpoints/<run>/last_model.pth
```

## Inference

```bash
python inference.py \
    --checkpoint checkpoints/<run>/best_model.pth \
    --dataset glas --save_images
```

Inference follows the paper protocol: EMA weights, deterministic DDIM with
`--ddim_steps 20` and `--K 25` averaged trajectories (use `--K 1` for faster
inference). The model and diffusion settings are read from the checkpoint.
Metrics are saved to `results/<dataset>/metrics.txt`; `--save_images` also
saves predictions and comparison images. For offline inference add
`--dino_repo /path/to/dinov2`.

For Synapse volume-level evaluation:

```bash
python inference.py \
    --checkpoint checkpoints/<synapse-run>/final_model.pth \
    --dataset synapse
```

For MoNuSeg-to-TNBC evaluation without fine-tuning:

```bash
python inference.py \
    --checkpoint checkpoints/<monuseg-run>/best_model.pth \
    --dataset tnbc
```
