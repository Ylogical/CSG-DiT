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

Prepare fixed train/validation/test splits before training. GlaS and MoNuSeg
use their official splits (85/80 and 30/14 training/test images) with an empty
`val/`; PH2 and IMID are split per image with seed 42 into 120/40/40 and
240/80/80 training/validation/test images. TNBC is used only for cross-dataset
evaluation and requires only `test/images/` and `test/masks/`.

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
`--save_dir`, `--log_dir`, `--dfca_kv_mode`, `--dino_repo`, `--dino_weights`,
`--recycle_prob`, `--recycle_dt_min`, `--recycle_dt_max` and `--ubam_kappa`.

### State guidance (UBAM) and recycling

The implementation follows Sec. III-F, III-H and Algorithm 1 of the paper:

- **Uncertainty (Eq. 18)** is computed from the pre-activation decoder output
  `o` of the prediction that built the current state `x_t`, not from `x_t`
  itself: `u = 1 - (p_(1) - p_(2))` for multi-class masks and
  `u = 1 - mean_c |tanh(o_c / kappa_u)|` (`kappa_u = 4`) for binary masks
  (`CSGDiT_Segmentation.compute_uncertainty`). Without a previous estimate,
  `u == 1`.
- **Inference** (`DiffusionSchedule.ddim_sample`) starts with `u == 1` and,
  after every DDIM step, recomputes `u` from that step's logits and passes it
  to the next step.
- **Training** (`compute_loss`) recycles each sample with probability
  `rho = 0.5`: `Delta ~ U{10, ..., 100}`, `s = min(t + Delta, T)`, a
  gradient-free prediction from the forward-noised `x_s` (with `u == 1`) gives
  `x_t` by one deterministic DDIM step from `s` to `t` and `u_t` by Eq. 18
  (`DiffusionSchedule.recycle`). Other samples keep the forward-noised `x_t`
  with `u == 1`. The target is always the ground-truth mask.
- **Modulation (Eqs. 17, 20)**: a timestep FiLM adapts the mask-state skip
  channel-wise; the gate `g = 1 + delta * tanh(h - mean(h))` reweights it
  spatially, and the image skip is reweighted complementarily by
  `max(2 - g, 0)` before both enter the decoder.

`--recycle_prob 0` trains on forward-noised states only (`u == 1` throughout).

The multi-class (Synapse) objective adds a class-weighted cross-entropy with
`lambda_CE = 1`: foreground classes are weighted by the inverse square root of
their training pixel frequency (normalised to mean 1), the background by 0.1.

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
weights. Which checkpoint to evaluate depends on whether the split has a
validation set:

| Dataset              | Validation set | Training schedule                    | Evaluate           |
| -------------------- | -------------- | ------------------------------------ | ------------------ |
| PH2, IMID            | yes            | up to 10,000 epochs, early stopping  | `best_model.pth` (best validation IoU) |
| GlaS, MoNuSeg        | no (official)  | full 10,000 epochs                   | `final_model.pth`  |
| Synapse              | no (TransUNet) | full 5,000 epochs                    | `final_model.pth`  |

Without a validation set, early stopping and best-model selection are off and
no `best_model.pth` is written.

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
    --checkpoint checkpoints/<glas-run>/final_model.pth \
    --dataset glas --save_images
```

For PH2 and IMID, pass `checkpoints/<run>/best_model.pth` instead.

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
    --dataset synapse --spacing_json synapse_spacing.json
```

`--spacing_json` maps each test case to its voxel spacing `[z, y, x]` in mm,
read from the original BTCV NIfTI headers (the TransUNet `.h5` files store no
spacing), e.g. `{"case0001": [3.0, 0.76, 0.76], ...}`. With it, HD95 is
reported in mm; without it, in voxels.

### Evaluation rules

- **HD95** pools the symmetric surface distances of both masks and takes the
  95th percentile (the `medpy.metric.binary.hd95` definition used by
  TransUNet). It is computed per image (2D datasets, in pixels at the
  evaluation resolution) or per volume and organ (Synapse, at the native
  label resolution).
- **Empty ground truth** (e.g. the gallbladder of Synapse case0035): DSC and
  HD95 are undefined, so the pair is excluded from the means and listed in the
  output, together with any false positives predicted there.
- **Missed target** (empty prediction, non-empty ground truth): DSC is 0 and
  HD95 is the diagonal of the image or volume, never 0.

These rules differ from TransUNet's `calculate_metric_percase`, which scores
any degenerate case with HD95 = 0 and, for an empty ground truth, gives DSC = 1
to a false positive and DSC = 0 to a correct empty prediction.

For MoNuSeg-to-TNBC evaluation without fine-tuning:

```bash
python inference.py \
    --checkpoint checkpoints/<monuseg-run>/final_model.pth \
    --dataset tnbc
```
