import os
import time
import argparse

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

from csg_dit.model import CSGDiT_models
from csg_dit.data import get_dataloader, get_num_classes, SynapseDataset
from csg_dit.diffusion import DiffusionSchedule
from csg_dit.utils import set_seed
from csg_dit.metrics import (SCIPY_AVAILABLE, compute_metrics, compute_hd95,
                             compute_ged, compute_ece, count_params, estimate_gflops)


def make_comparison(img_np, gt_np, pred_np):
    def overlay(img, mask, color, alpha=0.4):
        out = img.copy().astype(np.float32)
        for c, v in enumerate(color):
            out[:, :, c] = np.where(mask > 0, out[:, :, c] * (1 - alpha) + v * alpha,
                                    out[:, :, c])
        return out.clip(0, 255).astype(np.uint8)

    return np.concatenate([img_np,
                           overlay(img_np, gt_np,   (0, 200, 0)),
                           overlay(img_np, pred_np, (255, 80, 0))], axis=1)


@torch.no_grad()
def run_inference(model, diffusion, test_loader, device, args):
    model.eval()

    ious, dices, precs, recs, accs = [], [], [], [], []
    all_hd95, all_ged = [], []
    all_prob, all_gt_flat = [], []

    if args.save_images:
        img_dir = os.path.join(args.output_dir, args.dataset, 'images')
        os.makedirs(img_dir, exist_ok=True)

    sample_idx = 0
    t_start = time.perf_counter()
    hd95_overhead = 0.0

    for batch in tqdm(test_loader, desc='Running inference'):
        images = batch['image'].to(device)
        masks  = batch['mask'].to(device)
        B, _, H, W = images.shape

        k_soft = [diffusion.ddim_sample(model, (B, 1, H, W), images, device,
                                        ddim_steps=args.ddim_steps, eta=args.eta,
                                        progress=False).float()
                  for _ in range(args.K)]

        pred_mask = torch.stack(k_soft, dim=0).mean(dim=0)
        prob_map  = (pred_mask.cpu().numpy()[:, 0] + 1.0) / 2.0
        pred_np   = (pred_mask.cpu().numpy()[:, 0] > args.threshold).astype(np.uint8)
        gt_np     = (masks.cpu().numpy()[:, 0] > args.threshold).astype(np.uint8)
        k_binary  = [(ks.cpu().numpy()[:, 0] > args.threshold).astype(np.uint8)
                     for ks in k_soft]

        for i in range(B):
            pred_b, gt_b = pred_np[i], gt_np[i]

            iou, dice, prec, rec, acc = compute_metrics(pred_b, gt_b)
            ious.append(iou); dices.append(dice)
            precs.append(prec); recs.append(rec); accs.append(acc)

            if not args.skip_hd95 and SCIPY_AVAILABLE:
                _t = time.perf_counter()
                h = compute_hd95(pred_b, gt_b)
                if h is not None:
                    all_hd95.append(h)
                hd95_overhead += time.perf_counter() - _t

            if args.K > 1:
                all_ged.append(compute_ged([kb[i] for kb in k_binary], gt_b))

            all_prob.append(prob_map[i].flatten())
            all_gt_flat.append(gt_b.flatten().astype(np.float32))

            if args.save_images:
                img_vis = ((images[i].cpu().numpy().transpose(1, 2, 0) + 1.0) / 2.0
                           * 255).clip(0, 255).astype(np.uint8)
                Image.fromarray(img_vis).save(
                    os.path.join(img_dir, f'{sample_idx:04d}_image.png'))
                Image.fromarray((gt_b * 255).astype(np.uint8)).save(
                    os.path.join(img_dir, f'{sample_idx:04d}_gt.png'))
                Image.fromarray((pred_b * 255).astype(np.uint8)).save(
                    os.path.join(img_dir, f'{sample_idx:04d}_pred.png'))
                Image.fromarray(make_comparison(img_vis, gt_b, pred_b)).save(
                    os.path.join(img_dir, f'{sample_idx:04d}_compare.png'))

            sample_idx += 1

    total_time = time.perf_counter() - t_start - hd95_overhead

    return {
        'iou':  float(np.mean(ious)),
        'dice': float(np.mean(dices)),
        'prec': float(np.mean(precs)),
        'rec':  float(np.mean(recs)),
        'acc':  float(np.mean(accs)),
        'hd95': float(np.mean(all_hd95)) if all_hd95 else None,
        'ged':  float(np.mean(all_ged)) if all_ged else None,
        'ece':  compute_ece(np.concatenate(all_prob), np.concatenate(all_gt_flat)),
        'n_images':   sample_idx,
        'total_sec':  total_time,
        'per_image_ms': total_time / max(sample_idx, 1) * 1000,
    }


_PALETTE = np.array([
    [0, 0, 0], [220, 20, 60], [0, 200, 0], [30, 144, 255], [255, 215, 0],
    [148, 0, 211], [255, 140, 0], [0, 255, 255], [255, 105, 180],
], dtype=np.uint8)


def _slice_index(filename, fallback):
    if '_slice' in filename:
        try:
            return int(filename.split('_slice')[-1])
        except ValueError:
            pass
    return fallback


@torch.no_grad()
def run_inference_multiclass(model, diffusion, test_loader, device, args):
    from collections import defaultdict

    model.eval()
    num_classes = args.mask_channels
    organ_names = SynapseDataset.ORGAN_NAMES

    slice_iou, slice_dice, slice_sen, slice_acc = [], [], [], []
    case_slices = defaultdict(list)

    if args.save_images:
        img_dir = os.path.join(args.output_dir, args.dataset, 'images')
        os.makedirs(img_dir, exist_ok=True)

    sample_idx = 0
    t_start = time.perf_counter()

    for batch in tqdm(test_loader, desc='Running inference (multi-class)'):
        images    = batch['image'].to(device)
        gt_masks  = batch['mask'].to(device)
        case_ids  = batch.get('case_id',  [''] * images.shape[0])
        filenames = batch.get('filename', [''] * images.shape[0])
        B, _, H, W = images.shape

        pred_avg = None
        for _ in range(args.K):
            pred_k = diffusion.ddim_sample(model, (B, num_classes, H, W), images,
                                           device, ddim_steps=args.ddim_steps,
                                           eta=args.eta, progress=False).float()
            pred_avg = pred_k if pred_avg is None else pred_avg + pred_k
        pred_avg = pred_avg / args.K

        label_full = batch.get('label_full')
        if label_full is not None:
            H0, W0 = label_full.shape[-2], label_full.shape[-1]
            pred_idx = torch.nn.functional.interpolate(
                pred_avg.argmax(dim=1, keepdim=True).float(), size=(H0, W0),
                mode='nearest').squeeze(1).long().cpu().numpy()
            gt_idx = label_full.numpy().astype(np.int64)
        else:
            pred_idx = pred_avg.argmax(dim=1).cpu().numpy()
            gt_idx   = gt_masks.argmax(dim=1).cpu().numpy()

        for i in range(B):
            p, g = pred_idx[i], gt_idx[i]

            cls_iou, cls_dice, cls_sen = [], [], []
            for c in range(1, num_classes):
                pc, gc = (p == c), (g == c)
                inter, p_sum, g_sum = float((pc & gc).sum()), float(pc.sum()), float(gc.sum())
                if p_sum == 0 and g_sum == 0:
                    continue
                cls_iou.append(inter / (p_sum + g_sum - inter + 1e-6))
                cls_dice.append(2 * inter / (p_sum + g_sum + 1e-6))
                cls_sen.append(inter / (g_sum + 1e-6))
            if cls_dice:
                slice_iou.append(float(np.mean(cls_iou)))
                slice_dice.append(float(np.mean(cls_dice)))
                slice_sen.append(float(np.mean(cls_sen)))
                slice_acc.append(float((p == g).sum()) / p.size)

            case_slices[case_ids[i]].append(
                (_slice_index(filenames[i], sample_idx),
                 p.astype(np.uint8), g.astype(np.uint8)))

            if args.save_images:
                img_vis = ((images[i].cpu().numpy().transpose(1, 2, 0) + 1.0) / 2.0
                           * 255).clip(0, 255).astype(np.uint8)
                Image.fromarray(img_vis).save(
                    os.path.join(img_dir, f'{sample_idx:04d}_image.png'))
                Image.fromarray(_PALETTE[g.clip(0, num_classes - 1)]).save(
                    os.path.join(img_dir, f'{sample_idx:04d}_gt.png'))
                Image.fromarray(_PALETTE[p.clip(0, num_classes - 1)]).save(
                    os.path.join(img_dir, f'{sample_idx:04d}_pred.png'))

            sample_idx += 1

    total_time = time.perf_counter() - t_start

    do_hd95 = (not args.skip_hd95) and SCIPY_AVAILABLE
    dsc_per_organ  = defaultdict(list)
    hd95_per_organ = defaultdict(list)
    n_degenerate = 0

    for cid in tqdm(sorted(case_slices), desc='Case-wise 3D metrics'):
        items    = sorted(case_slices[cid], key=lambda it: it[0])
        pred_vol = np.stack([it[1] for it in items], axis=0)
        gt_vol   = np.stack([it[2] for it in items], axis=0)
        for c in range(1, num_classes):
            pc, gc = (pred_vol == c), (gt_vol == c)
            inter, p_sum, g_sum = float((pc & gc).sum()), float(pc.sum()), float(gc.sum())
            dsc_per_organ[c].append(1.0 if p_sum + g_sum == 0
                                    else 2 * inter / (p_sum + g_sum + 1e-6))
            if do_hd95:
                if not pc.any() or not gc.any():
                    n_degenerate += 1
                hd95_per_organ[c].append(compute_hd95(pc, gc))
    case_slices.clear()

    organ_dsc  = {c: float(np.mean(v)) for c, v in dsc_per_organ.items()}
    organ_hd95 = {c: float(np.mean(v)) for c, v in hd95_per_organ.items()}

    return {
        'dice': float(np.mean(list(organ_dsc.values()))) if organ_dsc else 0.0,
        'hd95': float(np.mean(list(organ_hd95.values()))) if organ_hd95 else None,
        'organ_dsc':  organ_dsc,
        'organ_hd95': organ_hd95,
        'organ_names': organ_names,
        'n_cases': len(dsc_per_organ[1]) if dsc_per_organ else 0,
        'n_hd95_degenerate': n_degenerate,
        'slice_iou':  float(np.mean(slice_iou))  if slice_iou  else 0.0,
        'slice_dice': float(np.mean(slice_dice)) if slice_dice else 0.0,
        'slice_sen':  float(np.mean(slice_sen))  if slice_sen  else 0.0,
        'slice_acc':  float(np.mean(slice_acc))  if slice_acc  else 0.0,
        'n_images':   sample_idx,
        'total_sec':  total_time,
        'per_image_ms': total_time / max(sample_idx, 1) * 1000,
    }


def parse_args():
    p = argparse.ArgumentParser(description='CSG-DiT inference')
    p.add_argument('--checkpoint',  type=str, required=True)
    p.add_argument('--dataset',     type=str, default='glas',
                   choices=['glas', 'monuseg', 'ph2', 'imid', 'tnbc', 'synapse'])
    p.add_argument('--K',           type=int, default=25,
                   help='Number of DDIM samples averaged per image (paper: 25; 1 = fast)')
    p.add_argument('--ddim_steps',  type=int, default=20,
                   help="Number of DDIM reverse steps T' (paper: 20)")
    p.add_argument('--batch_size',  type=int, default=4)
    p.add_argument('--save_images', action='store_true', default=False,
                   help='Save predictions and comparison images')
    p.add_argument('--skip_hd95',   action='store_true', default=False,
                   help='Skip HD95 (faster)')
    p.add_argument('--output_dir',  type=str, default='results')
    p.add_argument('--gpu',         type=int, default=0)
    p.add_argument('--num_workers', type=int, default=4)
    p.add_argument('--dino_repo',   type=str, default=None,
                   help='Local checkout of the official DINOv2 repository (offline use); '
                        'the DINOv2 weights themselves are restored from the checkpoint')
    args = p.parse_args()
    args.eta = 0.0
    args.threshold = 0.0
    args.use_ema = True
    return args


def main():
    args = parse_args()
    set_seed(1234)

    if torch.cuda.is_available():
        torch.cuda.set_device(args.gpu)
        device = torch.device(f'cuda:{args.gpu}')
    else:
        device = torch.device('cpu')

    print(f"Loading checkpoint from {args.checkpoint}...")
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    cfg = ckpt.get('train_config', {})
    if cfg:
        print("\n[Checkpoint train config]")
        for k, v in cfg.items():
            print(f"  {k}: {v}")
        print()

    if cfg.get('model', '').startswith(('FAS', 'JiT')):
        raise RuntimeError('This checkpoint predates CSG-DiT (no HCSE / UBAM); '
                           'use a CSG-DiT checkpoint.')
    args.model          = cfg.get('model', 'CSGDiT-B/16')
    args.num_timesteps  = cfg.get('num_timesteps', 200)
    args.beta_schedule  = cfg.get('beta_schedule', 'cosine')
    args.image_channels = cfg.get('image_channels', 3)
    args.image_size     = cfg.get('image_size') or (
        224 if args.dataset == 'synapse'
        else 512 if args.dataset in ('monuseg', 'tnbc') else 256)
    args.mask_channels  = cfg.get('mask_channels') or get_num_classes(args.dataset)
    multiclass = args.mask_channels > 1

    print("Loading dataset...")
    _, test_loader = get_dataloader(args.dataset, batch_size=args.batch_size,
                                    image_size=args.image_size,
                                    num_workers=args.num_workers)
    print(f"Test samples: {len(test_loader.dataset)}")

    kv_mode = ('v_only' if 'dfca_layers.0.to_k.weight' in ckpt['model_state_dict']
               else 'split')
    print(f"Creating model {args.model} @ {args.image_size}x{args.image_size} "
          f"(DFCA kv_mode={kv_mode})...")
    model = CSGDiT_models[args.model](
        input_size=args.image_size,
        mask_channels=args.mask_channels,
        image_channels=args.image_channels,
        dfca_kv_mode=kv_mode,
        dino_repo=args.dino_repo,
        dino_pretrained=False,
    ).to(device)

    model.load_state_dict(ckpt['model_state_dict'], strict=True)
    if 'ema_shadow' in ckpt:
        print("Loading EMA weights...")
        ema_shadow = ckpt['ema_shadow']
        model_keys = {n for n, p in model.named_parameters() if p.requires_grad}
        if model_keys != set(ema_shadow.keys()):
            raise RuntimeError("The EMA weights do not match the model structure.")
        for name, param in model.named_parameters():
            if name in ema_shadow:
                param.data.copy_(ema_shadow[name])
    else:
        print("[WARNING] no ema_shadow in the checkpoint, using the raw weights")

    diffusion = DiffusionSchedule(num_timesteps=args.num_timesteps,
                                  schedule_type=args.beta_schedule)

    params = count_params(model)
    gflops = estimate_gflops(model, args.image_size, args.mask_channels,
                             args.image_channels, device)

    print("Running inference...")
    if multiclass:
        r = run_inference_multiclass(model, diffusion, test_loader, device, args)
    else:
        r = run_inference(model, diffusion, test_loader, device, args)

    header = [
        f"Checkpoint: {args.checkpoint}",
        f"Dataset:    {args.dataset} ({r['n_images']} images @ {args.image_size})",
        f"Sampling:   DDIM T'={args.ddim_steps}, eta={args.eta}, K={args.K}"
        f"{' (EMA)' if 'ema_shadow' in ckpt else ''}",
        "",
        f"Params:    {params:.2f} M",
        f"GFLOPs:    {gflops:.2f}" if gflops is not None else "GFLOPs:    N/A",
    ]

    if multiclass:
        names = r['organ_names']
        lines = header + [
            f"Cases:     {r['n_cases']} volumes, scored at the native label resolution",
            "",
            "Case-wise 3D (the numbers reported in the paper):",
            f"  DSC:     {r['dice'] * 100:.2f} %",
            (f"  HD95:    {r['hd95']:.2f} voxels "
             f"({r['n_hd95_degenerate']} of {r['n_cases'] * (args.mask_channels - 1)} "
             f"(case, organ) pairs degenerate, counted as 0)"
             if r['hd95'] is not None
             else "  HD95:    N/A (needs scipy, or disabled with --skip_hd95)"),
            "",
            "  Per organ (DSC % / HD95 voxels):",
        ] + [
            f"    {names[c]:<6} {r['organ_dsc'][c] * 100:6.2f}" +
            (f" / {r['organ_hd95'][c]:7.2f}" if c in r['organ_hd95'] else "")
            for c in sorted(r['organ_dsc'])
        ] + [
            "",
            "Slice level, macro over the foreground classes:",
            f"  mIoU:    {r['slice_iou']:.4f}",
            f"  DSC:     {r['slice_dice']:.4f}",
            f"  Sen:     {r['slice_sen']:.4f}",
            f"  Acc:     {r['slice_acc']:.4f}",
            "",
            f"Total time: {r['total_sec']:.2f} s  ({r['per_image_ms']:.1f} ms/slice)",
        ]
    else:
        lines = header + [
            f"mIoU:      {r['iou']:.4f}",
            f"DSC:       {r['dice']:.4f}",
            f"Sen:       {r['rec']:.4f}",
            f"Acc:       {r['acc']:.4f}",
            f"Precision: {r['prec']:.4f}",
            (f"HD95:      {r['hd95']:.2f} px" if r['hd95'] is not None
             else "HD95:      N/A (needs scipy, or disabled with --skip_hd95)"),
            (f"GED:       {r['ged']:.4f}" if r['ged'] is not None
             else "GED:       N/A (needs --K > 1)"),
            f"ECE:       {r['ece']:.4f}",
            "",
            f"Total time: {r['total_sec']:.2f} s  ({r['per_image_ms']:.1f} ms/image)",
        ]
    print("\n" + "=" * 50)
    print("\n".join(lines))
    print("=" * 50)

    dataset_dir = os.path.join(args.output_dir, args.dataset)
    os.makedirs(dataset_dir, exist_ok=True)
    metrics_path = os.path.join(dataset_dir, 'metrics.txt')
    with open(metrics_path, 'w', encoding='utf-8') as f:
        f.write("\n".join(lines) + "\n")
    print(f"Results saved to {metrics_path}")


if __name__ == '__main__':
    main()
