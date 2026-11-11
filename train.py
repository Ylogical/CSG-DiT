import os
import json
import argparse
from datetime import datetime
from collections import deque

import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from csg_dit.model import CSGDiT_models
from csg_dit.data import get_dataloader_with_val, get_num_classes
from csg_dit.diffusion import DiffusionSchedule, compute_loss
from csg_dit.metrics import binary_slice_metrics, multiclass_slice_metrics
from csg_dit.utils import (EMA, EarlyStopping, TrainingLogger, set_seed,
                           get_lr_scheduler, save_checkpoint, load_checkpoint)

_DATASET_DEFAULTS = {
    'glas':    {'image_size': 256, 'model': 'CSGDiT-B/16', 'batch_size': 8,
                'lr': 2e-4, 'epochs': 10000, 'warmup_epochs': 50,
                'patience': 20, 'metrics_eval_freq': 30, 'es_min_epochs': 0,
                'weight_decay': 0.05, 'attn_drop': 0.1, 'proj_drop': 0.1},
    'ph2':     {'image_size': 256, 'model': 'CSGDiT-B/16', 'batch_size': 8,
                'lr': 2e-4, 'epochs': 10000, 'warmup_epochs': 50,
                'patience': 20, 'metrics_eval_freq': 30, 'es_min_epochs': 0,
                'weight_decay': 0.05, 'attn_drop': 0.1, 'proj_drop': 0.1},
    'imid':    {'image_size': 256, 'model': 'CSGDiT-B/16', 'batch_size': 4,
                'lr': 2e-4, 'epochs': 10000, 'warmup_epochs': 50,
                'patience': 20, 'metrics_eval_freq': 30, 'es_min_epochs': 0,
                'weight_decay': 0.05, 'attn_drop': 0.1, 'proj_drop': 0.1},
    'monuseg': {'image_size': 512, 'model': 'CSGDiT-B/32', 'batch_size': 4,
                'lr': 1e-4, 'epochs': 10000, 'warmup_epochs': 60,
                'patience': 20, 'metrics_eval_freq': 25, 'es_min_epochs': 0,
                'weight_decay': 0.05, 'attn_drop': 0.1, 'proj_drop': 0.1},
    'synapse': {'image_size': 224, 'model': 'CSGDiT-B/16', 'batch_size': 8,
                'lr': 2e-4, 'epochs': 5000, 'warmup_epochs': 30,
                'patience': 15, 'metrics_eval_freq': 20, 'es_min_epochs': 0,
                'weight_decay': 0.02, 'attn_drop': 0.05, 'proj_drop': 0.05,
                'ce_weight': 0.5},
}

for _cfg in _DATASET_DEFAULTS.values():
    _cfg.setdefault('ce_weight', 0.0)

_COMMON = {
    'image_channels': 3,
    'num_timesteps': 200,
    'beta_schedule': 'cosine',
    'dice_weight': 1.0,
    'grad_clip': 1.0,
    'ema_decay': 0.9999,
    'es_delta': 1e-4,
    'val_seed': 1234,
}


def parse_args():
    parser = argparse.ArgumentParser(
        description='Train CSG-DiT',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python train.py --dataset glas
  python train.py --dataset synapse --gpu 1
  python train.py --dataset glas --resume checkpoints/<run>/last_model.pth
  torchrun --nproc_per_node=2 train.py --dataset glas
""")
    parser.add_argument('--dataset', type=str, default='glas',
                        choices=['glas', 'monuseg', 'ph2', 'imid', 'synapse'])
    parser.add_argument('--batch_size', type=int, default=None,
                        help='Global batch size, split evenly over the GPUs. '
                             'Default: glas/ph2/synapse=8, monuseg/imid=4')
    parser.add_argument('--epochs', type=int, default=None,
                        help='Default: synapse=5000, others=10000 (with early stopping)')
    parser.add_argument('--lr', type=float, default=None,
                        help='Default: 2e-4, monuseg=1e-4')
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--resume', type=str, default=None,
                        help='Checkpoint to resume from, e.g. last_model.pth')
    parser.add_argument('--save_dir', type=str, default='checkpoints')
    parser.add_argument('--log_dir', type=str, default='logs')
    parser.add_argument('--dfca_kv_mode', type=str, default='split',
                        choices=['split', 'v_only'],
                        help="DFCA keys: 'split' takes K_b from each frequency band (paper, "
                             "default); 'v_only' takes one shared K from the unfiltered "
                             "image tokens and applies the bands to V only")
    parser.add_argument('--dino_repo', type=str, default=None,
                        help='Local checkout of the official DINOv2 repository (offline use)')
    parser.add_argument('--dino_weights', type=str, default=None,
                        help='Local dinov2_vitb14_pretrain.pth (offline use)')
    args = parser.parse_args()

    cfg = dict(_COMMON, **_DATASET_DEFAULTS[args.dataset])
    for key, value in cfg.items():
        if getattr(args, key, None) is None:
            setattr(args, key, value)
    args.eval_freq = args.metrics_eval_freq
    args.mask_channels = get_num_classes(args.dataset)
    return args


class _NullWriter:
    def add_scalar(self, *args, **kwargs):
        pass

    def close(self):
        pass


def setup_distributed(args):
    args.world_size = int(os.environ.get('WORLD_SIZE', 1))
    args.rank = int(os.environ.get('RANK', 0))
    args.distributed = args.world_size > 1
    if args.distributed:
        local_rank = int(os.environ['LOCAL_RANK'])
        torch.cuda.set_device(local_rank)
        dist.init_process_group('nccl')
        device = torch.device(f'cuda:{local_rank}')
        if args.batch_size % args.world_size:
            raise ValueError('batch_size must be divisible by the number of GPUs')
    elif torch.cuda.is_available():
        torch.cuda.set_device(args.gpu)
        device = torch.device(f'cuda:{args.gpu}')
    else:
        device = torch.device('cpu')
    args.is_main = args.rank == 0
    return device


def train_one_epoch(model, diffusion, train_loader, optimizer, device, epoch,
                    writer, args, ema):
    model.train()
    if isinstance(train_loader.sampler, DistributedSampler):
        train_loader.sampler.set_epoch(epoch)
    total_loss = 0.0
    loss_window = deque(maxlen=min(20, len(train_loader)))

    pbar = tqdm(train_loader, desc=f'Epoch {epoch + 1}/{args.epochs}',
                disable=not args.is_main)
    for step, batch in enumerate(pbar):
        loss, _, _ = compute_loss(
            model, diffusion, batch, device,
            dice_weight=args.dice_weight,
            ce_weight=args.ce_weight,
        )
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        optimizer.step()
        ema.update()

        loss_val = loss.item()
        total_loss += loss_val
        loss_window.append(loss_val)
        smooth_loss = sum(loss_window) / len(loss_window)
        pbar.set_postfix({'loss(smooth)': f'{smooth_loss:.4f}',
                          'loss(step)': f'{loss_val:.4f}'})

        global_step = epoch * len(train_loader) + step
        writer.add_scalar('train/loss_step', loss_val, global_step)
        writer.add_scalar('train/loss_smooth', smooth_loss, global_step)

    return total_loss / len(train_loader)


@torch.no_grad()
def evaluate(model, diffusion, val_loader, device, epoch, writer, args):
    model.eval()
    total_loss = 0.0
    for batch in tqdm(val_loader, desc='Evaluating', leave=False):
        loss, _, _ = compute_loss(
            model, diffusion, batch, device,
            dice_weight=args.dice_weight,
            ce_weight=args.ce_weight,
        )
        total_loss += loss.item()
    avg_loss = total_loss / len(val_loader)
    writer.add_scalar('eval/loss', avg_loss, epoch)
    return avg_loss


@torch.no_grad()
def evaluate_metrics(model, diffusion, val_loader, device, epoch, writer, args):
    model.eval()
    iou_list, dice_list, sen_list, acc_list = [], [], [], []
    acc = (iou_list, dice_list, sen_list, acc_list)
    generator = torch.Generator(device=device)
    generator.manual_seed(args.val_seed)

    for batch in tqdm(val_loader, desc=f'Metrics Eval (Epoch {epoch + 1})', leave=False):
        images   = batch['image'].to(device)
        gt_masks = batch['mask'].to(device)
        B, C, H, W = gt_masks.shape

        pred_masks = diffusion.ddim_sample(model, (B, C, H, W), images, device,
                                           ddim_steps=20, eta=0.0, progress=False,
                                           generator=generator)

        if C > 1:
            multiclass_slice_metrics(pred_masks.argmax(dim=1),
                                      gt_masks.argmax(dim=1), C, acc)
        else:
            binary_slice_metrics((pred_masks > 0.0).float(),
                                  (gt_masks > 0.0).float(), acc)

    mean_iou  = float(np.mean(iou_list))
    mean_dice = float(np.mean(dice_list))
    mean_sen  = float(np.mean(sen_list))
    mean_acc  = float(np.mean(acc_list))

    writer.add_scalar('eval/iou',  mean_iou,  epoch)
    writer.add_scalar('eval/dice', mean_dice, epoch)
    writer.add_scalar('eval/sen',  mean_sen,  epoch)
    writer.add_scalar('eval/acc',  mean_acc,  epoch)

    return mean_iou, mean_dice, mean_sen, mean_acc


def main():
    args = parse_args()
    device = setup_distributed(args)

    set_seed(args.seed + args.rank)

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    if args.distributed:
        stamp = [timestamp]
        dist.broadcast_object_list(stamp, src=0)
        timestamp = stamp[0]
    exp_name = f"{args.dataset}_{args.model.replace('/', '_')}_{timestamp}"
    save_dir = os.path.join(args.save_dir, exp_name)
    log_dir  = os.path.join(args.log_dir,  exp_name)
    if args.is_main:
        os.makedirs(save_dir, exist_ok=True)
        os.makedirs(log_dir,  exist_ok=True)

    writer = SummaryWriter(log_dir) if args.is_main else _NullWriter()
    if not args.is_main:
        import builtins
        builtins.print = lambda *a, **k: None

    print("=" * 60)
    print(f"Experiment: {exp_name}")
    print(f"Dataset: {args.dataset} | Image: {args.image_size} | Batch: {args.batch_size}")
    if args.mask_channels > 1:
        print(f"Multi-class: {args.mask_channels} mask channels, softmax x0 head, "
              f"lambda_CE={args.ce_weight}")
    print(f"Model: {args.model} | Epochs: {args.epochs} | LR: {args.lr}")
    print(f"WeightDecay: {args.weight_decay} | Dropout(attn/proj): "
          f"{args.attn_drop}/{args.proj_drop}")
    print(f"Warmup: {args.warmup_epochs} | Patience: {args.patience} "
          f"(early stop after {args.patience} x {args.metrics_eval_freq} = "
          f"{args.patience * args.metrics_eval_freq} epochs w/o improvement)")
    print(f"EMA: decay={args.ema_decay}"
          f" | Frozen pretrained DINOv2-B/14; remaining modules trained from scratch")
    print("=" * 60)

    if device.type == 'cuda':
        print(f"Using {args.world_size} GPU(s): {torch.cuda.get_device_name(device)}"
              + (f", {args.batch_size // args.world_size} images per GPU"
                 if args.distributed else ""))
    else:
        print("Using CPU")

    print("Loading dataset...")
    train_loader, val_loader, test_loader = get_dataloader_with_val(
        args.dataset,
        batch_size=args.batch_size,
        image_size=args.image_size,
        num_workers=args.num_workers,
    )
    if args.distributed:
        train_loader = DataLoader(
            train_loader.dataset, batch_size=args.batch_size // args.world_size,
            sampler=DistributedSampler(train_loader.dataset, shuffle=True, seed=args.seed,
                                       drop_last=True),
            num_workers=args.num_workers, pin_memory=True, drop_last=True)
    has_val = val_loader is not None
    print(f"Train: {len(train_loader.dataset)}  "
          f"Val: {len(val_loader.dataset) if has_val else 0}  "
          f"Test: {len(test_loader.dataset)}")
    if not has_val:
        print("No validation split: early stopping and best-model selection are "
              "off, training runs the full schedule and final_model.pth is the "
              "model to evaluate. Run `python -m csg_dit.data --val_cases N` to "
              "hold cases out instead.")

    print("Creating model...")
    model = CSGDiT_models[args.model](
        input_size=args.image_size,
        mask_channels=args.mask_channels,
        image_channels=args.image_channels,
        attn_drop=args.attn_drop,
        proj_drop=args.proj_drop,
        dfca_kv_mode=args.dfca_kv_mode,
        dino_repo=args.dino_repo,
        dino_weights=args.dino_weights,
        dino_pretrained=args.resume is None,
    ).to(device)
    if args.distributed:
        model = DDP(model, device_ids=[device.index])
    net = model.module if args.distributed else model
    total_p = sum(p.numel() for p in model.parameters())
    train_p = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Params: {total_p:,} ({total_p / 1e6:.2f}M), "
          f"trainable {train_p:,} ({train_p / 1e6:.2f}M)")

    diffusion = DiffusionSchedule(num_timesteps=args.num_timesteps,
                                  schedule_type=args.beta_schedule)

    optimizer = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad), lr=args.lr,
                                  weight_decay=args.weight_decay, betas=(0.9, 0.999))
    scheduler = get_lr_scheduler(optimizer, args.warmup_epochs, args.epochs)

    ema = EMA(net, decay=args.ema_decay)

    start_epoch = 0
    if args.resume:
        start_epoch = load_checkpoint(net, optimizer, scheduler, args.resume, ema) + 1

    early_stopper = (EarlyStopping(patience=args.patience, min_delta=args.es_delta,
                                   min_epochs=args.es_min_epochs)
                     if has_val else None)

    logger = TrainingLogger(log_dir=log_dir, exp_name=exp_name) if args.is_main else None
    log = logger.log if logger is not None else (lambda *a, **k: None)
    log(f"Training config:\n{json.dumps(vars(args), indent=2, default=str)}")
    log(f"Dataset: {args.dataset}  Train={len(train_loader.dataset)}  "
               f"Val={len(val_loader.dataset) if has_val else 0}  "
               f"Test={len(test_loader.dataset)}")
    log(f"Model params: {total_p:,} ({total_p / 1e6:.2f}M)")

    print("Start training...")
    best_iou = -1.0
    best_loss = float('inf')
    train_loss = float('inf')
    last_epoch = start_epoch

    for epoch in range(start_epoch, args.epochs):
        last_epoch = epoch
        train_loss = train_one_epoch(model, diffusion, train_loader, optimizer,
                                     device, epoch, writer, args, ema)
        scheduler.step()
        current_lr = optimizer.param_groups[0]['lr']
        writer.add_scalar('train/lr', current_lr, epoch)
        writer.add_scalar('train/loss_epoch', train_loss, epoch)

        eval_loss = None
        iou = dice = sen = acc = None
        stop = False
        if args.is_main:
            if has_val and (epoch + 1) % args.eval_freq == 0:
                ema.apply_shadow()
                try:
                    eval_loss = evaluate(net, diffusion, val_loader, device, epoch, writer, args)
                finally:
                    ema.restore()

                if best_iou < 0 and eval_loss is not None and eval_loss < best_loss:
                    best_loss = eval_loss
                    ema.apply_shadow()
                    save_checkpoint(net, optimizer, scheduler, epoch, eval_loss,
                                    os.path.join(save_dir, 'best_model.pth'), ema, args)
                    ema.restore()

            if has_val and (epoch + 1) % args.metrics_eval_freq == 0:
                ema.apply_shadow()
                try:
                    iou, dice, sen, acc = evaluate_metrics(net, diffusion, val_loader,
                                                           device, epoch, writer, args)
                    if iou > best_iou:
                        best_iou = iou
                        save_checkpoint(
                            net, optimizer, scheduler, epoch,
                            eval_loss if eval_loss is not None else train_loss,
                            os.path.join(save_dir, 'best_model.pth'), ema, args,
                        )
                        log(f"  >> Best model (mIoU={best_iou:.4f} DSC={dice:.4f} "
                            f"Sen={sen:.4f} Acc={acc:.4f})")
                finally:
                    ema.restore()
                log(f"  >> Metrics @ {epoch + 1}: mIoU={iou:.4f} DSC={dice:.4f} "
                    f"Sen={sen:.4f} Acc={acc:.4f}  best_mIoU={best_iou:.4f}")

                if early_stopper is not None and early_stopper(iou, epoch):
                    print(f'[Early Stop] epoch={epoch + 1}  '
                          f'best_mIoU={early_stopper.best_score:.4f} '
                          f'@ epoch {early_stopper.best_epoch + 1}')
                    log(f'[Early Stop] epoch={epoch + 1}  '
                        f'best_mIoU={early_stopper.best_score:.4f} '
                        f'@ epoch {early_stopper.best_epoch + 1}')
                    stop = True
                elif early_stopper is not None:
                    print(f'  {early_stopper.status_str()}')

            if (epoch + 1) % args.metrics_eval_freq == 0:
                save_checkpoint(net, optimizer, scheduler, epoch, train_loss,
                                os.path.join(save_dir, 'last_model.pth'), ema, args)

            logger.log_epoch(epoch, train_loss, eval_loss, current_lr, iou=iou, dice=dice)

        if args.distributed:
            flag = torch.tensor([float(stop)], device=device)
            dist.broadcast(flag, src=0)
            stop = bool(flag.item())
        if stop:
            break

    if args.is_main:
        ema.apply_shadow()
        save_checkpoint(net, optimizer, scheduler, last_epoch, train_loss,
                        os.path.join(save_dir, 'final_model.pth'), ema, args)
        logger.close()
    writer.close()
    if args.distributed:
        dist.barrier()
        dist.destroy_process_group()
    print("Training completed! Evaluate "
          + ("best_model.pth (best validation mIoU)." if has_val
             else "final_model.pth (this split has no validation set)."))


if __name__ == '__main__':
    main()
