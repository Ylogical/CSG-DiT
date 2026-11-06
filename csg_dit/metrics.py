import numpy as np
import torch
import torch.nn as nn

try:
    from scipy.ndimage import distance_transform_edt, binary_erosion, generate_binary_structure
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False


def compute_metrics(pred_binary, gt_binary):
    pred = pred_binary.astype(bool)
    gt   = gt_binary.astype(bool)

    tp = (pred & gt).sum()
    fp = (pred & ~gt).sum()
    fn = (~pred & gt).sum()
    tn = (~pred & ~gt).sum()

    iou       = tp / (tp + fp + fn + 1e-8)
    dice      = 2 * tp / (2 * tp + fp + fn + 1e-8)
    precision = tp / (tp + fp + 1e-8)
    recall    = tp / (tp + fn + 1e-8)
    acc       = (tp + tn) / (tp + fp + fn + tn + 1e-8)
    return iou, dice, precision, recall, acc


def _surface_distances(pred, gt):
    conn = generate_binary_structure(pred.ndim, 1)
    pred_border = pred ^ binary_erosion(pred, structure=conn, iterations=1)
    gt_border   = gt   ^ binary_erosion(gt,   structure=conn, iterations=1)
    return distance_transform_edt(~gt_border)[pred_border]


def compute_hd95(pred, gt):
    if not SCIPY_AVAILABLE:
        return None
    pred = np.asarray(pred).astype(bool)
    gt   = np.asarray(gt).astype(bool)
    if not pred.any() or not gt.any():
        return 0.0

    idx = np.nonzero(pred | gt)
    slicer = tuple(slice(max(int(i.min()) - 1, 0), min(int(i.max()) + 2, s))
                   for i, s in zip(idx, pred.shape))
    pred, gt = pred[slicer], gt[slicer]

    sds = np.hstack((_surface_distances(pred, gt), _surface_distances(gt, pred)))
    return float(np.percentile(sds, 95))


def _dice_dist(a, b):
    a, b = a.astype(bool), b.astype(bool)
    return 1.0 - 2.0 * (a & b).sum() / (a.sum() + b.sum() + 1e-8)


def compute_ged(k_preds, gt):
    K = len(k_preds)
    sg = 2.0 * sum(_dice_dist(s, gt) for s in k_preds) / K
    ss = sum(_dice_dist(k_preds[i], k_preds[j])
             for i in range(K) for j in range(K)) / (K * K)
    return float(sg - ss)


def compute_ece(prob_flat, gt_flat, n_bins=15):
    prob_flat = np.clip(prob_flat, 0.0, 1.0)
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece, N = 0.0, len(prob_flat)
    for lo, hi in zip(bins[:-1], bins[1:]):
        mask = (prob_flat >= lo) & (prob_flat < hi)
        if mask.sum() == 0:
            continue
        ece += mask.sum() / N * abs(prob_flat[mask].mean() - gt_flat[mask].mean())
    return float(ece)


def binary_slice_metrics(pred_bin, gt_bin, out):
    for i in range(pred_bin.shape[0]):
        p = pred_bin[i].reshape(-1)
        g = gt_bin[i].reshape(-1)
        tp = (p * g).sum().item()
        fp = (p * (1 - g)).sum().item()
        fn = ((1 - p) * g).sum().item()
        tn = ((1 - p) * (1 - g)).sum().item()
        out[0].append(tp / (tp + fp + fn + 1e-6))
        out[1].append(2 * tp / (2 * tp + fp + fn + 1e-6))
        out[2].append(tp / (tp + fn + 1e-6))
        out[3].append((tp + tn) / (tp + fp + fn + tn + 1e-6))


def multiclass_slice_metrics(pred_idx, gt_idx, num_classes, out):
    for i in range(pred_idx.shape[0]):
        p, g = pred_idx[i], gt_idx[i]
        cls_iou, cls_dice, cls_sen = [], [], []
        for c in range(1, num_classes):
            pc, gc = (p == c), (g == c)
            inter, p_sum, g_sum = (pc & gc).sum().item(), pc.sum().item(), gc.sum().item()
            if p_sum == 0 and g_sum == 0:
                continue
            cls_iou.append(inter / (p_sum + g_sum - inter + 1e-6))
            cls_dice.append(2 * inter / (p_sum + g_sum + 1e-6))
            cls_sen.append(inter / (g_sum + 1e-6))
        if cls_dice:
            out[0].append(float(np.mean(cls_iou)))
            out[1].append(float(np.mean(cls_dice)))
            out[2].append(float(np.mean(cls_sen)))
            out[3].append((p == g).float().mean().item())


def count_params(model):
    return sum(p.numel() for p in model.parameters()) / 1e6


def estimate_gflops(model, image_size, mask_channels, image_channels, device):
    macs = [0]
    hooks = []

    def _hook_conv2d(m, inp, out):
        B, Co, Ho, Wo = out.shape
        kH, kW = m.kernel_size if isinstance(m.kernel_size, tuple) else (m.kernel_size,) * 2
        macs[0] += B * Co * Ho * Wo * (m.in_channels // m.groups) * kH * kW

    def _hook_deconv2d(m, inp, out):
        B, Ci, Hi, Wi = inp[0].shape
        kH, kW = m.kernel_size if isinstance(m.kernel_size, tuple) else (m.kernel_size,) * 2
        macs[0] += B * Ci * Hi * Wi * (m.out_channels // m.groups) * kH * kW

    def _hook_linear(m, inp, out):
        macs[0] += (inp[0].numel() // m.in_features) * m.in_features * m.out_features

    for mod in model.modules():
        if isinstance(mod, nn.Conv2d):
            hooks.append(mod.register_forward_hook(_hook_conv2d))
        elif isinstance(mod, nn.ConvTranspose2d):
            hooks.append(mod.register_forward_hook(_hook_deconv2d))
        elif isinstance(mod, nn.Linear):
            hooks.append(mod.register_forward_hook(_hook_linear))

    model.eval()
    try:
        with torch.no_grad():
            model(torch.zeros(1, mask_channels, image_size, image_size, device=device),
                  torch.zeros(1, dtype=torch.long, device=device),
                  torch.zeros(1, image_channels, image_size, image_size, device=device))
        gflops = 2.0 * macs[0] / 1e9
    except Exception:
        gflops = None
    finally:
        for h in hooks:
            h.remove()
    return gflops
