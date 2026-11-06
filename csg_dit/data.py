import os
import json
import random
import argparse

import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF

try:
    import h5py
except ImportError:
    h5py = None


_DATASET_ROOT = {
    'glas':    'processed_glas',
    'monuseg': 'processed_monuseg',
    'ph2':     'processed_ph2',
    'imid':    'processed_imid',
    'tnbc':    'processed_tnbc',
    'synapse': 'Synapse',
}

_DATASET_NUM_CLASSES = {
    'glas': 1, 'monuseg': 1, 'ph2': 1, 'imid': 1, 'tnbc': 1,
    'synapse': 9,
}


def get_num_classes(dataset_name: str) -> int:
    return _DATASET_NUM_CLASSES.get(dataset_name.lower(), 1)


def is_multiclass(dataset_name: str) -> bool:
    return get_num_classes(dataset_name) > 1


def get_dataset_root(dataset_name: str) -> str:
    root = _DATASET_ROOT.get(dataset_name.lower())
    if root is None:
        raise ValueError(f"Unknown dataset: {dataset_name}, "
                         f"supported: {list(_DATASET_ROOT)}")
    return root


class MedicalSegmentationDataset(Dataset):
    def __init__(self, root_dir: str, split: str = 'train', image_size: int = 256):
        self.root_dir   = root_dir
        self.split      = split
        self.image_size = image_size
        self.is_train   = (split == 'train')

        self.image_dir = os.path.join(root_dir, split, 'images')
        self.mask_dir  = os.path.join(root_dir, split, 'masks')

        if not os.path.isdir(self.image_dir):
            raise FileNotFoundError(
                f"Dataset directory not found: {self.image_dir}\n"
                f"Expected {root_dir}/{{train,val,test}}/{{images,masks}}/ "
                f"- see 'Data preparation' in README.md"
            )

        self.filenames = sorted([
            f for f in os.listdir(self.image_dir)
            if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff'))
        ])
        if len(self.filenames) == 0:
            raise RuntimeError(f"Empty dataset: {self.image_dir}")

        _exts = ('.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff')
        mask_by_stem = {
            os.path.splitext(m)[0]: m
            for m in os.listdir(self.mask_dir) if m.lower().endswith(_exts)
        }
        self.mask_filenames = []
        for f in self.filenames:
            stem = os.path.splitext(f)[0]
            if stem not in mask_by_stem:
                raise FileNotFoundError(
                    f"No mask found for image {f} in {self.mask_dir} (stem '{stem}')"
                )
            self.mask_filenames.append(mask_by_stem[stem])

        self.resize      = transforms.Resize((image_size, image_size))
        self.resize_mask = transforms.Resize(
            (image_size, image_size),
            interpolation=transforms.InterpolationMode.NEAREST
        )
        self.to_tensor = transforms.ToTensor()

        if self.is_train:
            self.color_jitter = transforms.ColorJitter(
                brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1
            )

    def __len__(self):
        return len(self.filenames)

    def apply_augmentation(self, image: Image.Image, mask: Image.Image):
        if random.random() > 0.5:
            angle = random.choice([90, 180, 270])
            image = TF.rotate(image, angle)
            mask  = TF.rotate(mask,  angle)

        if random.random() > 0.5:
            image = TF.hflip(image)
            mask  = TF.hflip(mask)

        if random.random() > 0.5:
            image = TF.vflip(image)
            mask  = TF.vflip(mask)

        if random.random() > 0.3:
            scale    = random.uniform(0.8, 1.2)
            new_size = int(self.image_size * scale)
            image = TF.resize(image, new_size)
            mask  = TF.resize(mask,  new_size,
                              interpolation=transforms.InterpolationMode.NEAREST)
            if scale > 1.0:
                i, j, h, w = transforms.RandomCrop.get_params(
                    image, (self.image_size, self.image_size))
                image = TF.crop(image, i, j, h, w)
                mask  = TF.crop(mask,  i, j, h, w)
            else:
                image = TF.resize(image, self.image_size)
                mask  = TF.resize(mask,  self.image_size,
                                  interpolation=transforms.InterpolationMode.NEAREST)

        if random.random() > 0.3:
            image = self.color_jitter(image)

        if random.random() > 0.7:
            image = TF.gaussian_blur(image, random.choice([3, 5]))

        return image, mask

    def __getitem__(self, idx):
        fname     = self.filenames[idx]
        img_path  = os.path.join(self.image_dir, fname)
        mask_path = os.path.join(self.mask_dir,  self.mask_filenames[idx])

        image = Image.open(img_path).convert('RGB')
        mask  = Image.open(mask_path).convert('L')

        image = self.resize(image)
        mask  = self.resize_mask(mask)

        if self.is_train:
            image, mask = self.apply_augmentation(image, mask)

        image = self.to_tensor(image)
        mask  = self.to_tensor(mask)

        return {
            'image':    image * 2.0 - 1.0,
            'mask':     (mask > 0.5).float() * 2.0 - 1.0,
            'filename': fname,
        }


class SynapseDataset(Dataset):
    NUM_CLASSES = 9
    CLASS_NAMES = ['background', 'aorta', 'gallbladder', 'kidney_left',
                   'kidney_right', 'liver', 'pancreas', 'spleen', 'stomach']
    ORGAN_NAMES = {1: 'Aorta', 2: 'GB', 3: 'KL', 4: 'KR',
                   5: 'Liver', 6: 'PC', 7: 'SP', 8: 'SM'}

    CT_CLIP_MIN = -125.0
    CT_CLIP_MAX = 275.0

    def __init__(self, root_dir: str, split: str = 'train', image_size: int = 224,
                 split_json: str = None, native_label: bool = None,
                 allow_empty: bool = False):
        if split == 'test' and h5py is None:
            raise ImportError('The Synapse test split needs h5py (pip install h5py)')

        self.root_dir     = root_dir
        self.split        = split
        self.image_size   = image_size
        self.is_train     = (split == 'train')
        self.native_label = (split == 'test') if native_label is None else native_label

        if split_json is None:
            split_json = os.path.join(root_dir, 'split.json')
        if not os.path.isfile(split_json):
            raise FileNotFoundError(
                f"Synapse split.json not found: {split_json}\n"
                f"Run `python -m csg_dit.data` first - see 'Data preparation' "
                f"in README.md"
            )
        with open(split_json, 'r', encoding='utf-8') as f:
            split_info = json.load(f)
        if split not in split_info:
            raise ValueError(f"Unknown split: {split} (train | val | test)")

        self.cases = set(split_info[split].get('cases', []))

        if split in ('train', 'val'):
            npz_dir = os.path.join(root_dir, 'train_npz')
            if not os.path.isdir(npz_dir):
                raise FileNotFoundError(f"Dataset directory not found: {npz_dir}")
            self.samples = [
                (os.path.join(npz_dir, f), f.split('_')[0], None, f[:-4])
                for f in sorted(os.listdir(npz_dir))
                if f.endswith('.npz') and f.split('_')[0] in self.cases
            ]
        else:
            h5_dir = os.path.join(root_dir, 'test_vol_h5')
            self.samples = []
            for fname in sorted(split_info[split].get('files', [])):
                fpath = os.path.join(h5_dir, fname)
                with h5py.File(fpath, 'r') as f:
                    n_slices = f['image'].shape[0]
                case_id = fname.split('.')[0]
                self.samples += [(fpath, case_id, s, f'{case_id}_slice{s:03d}')
                                 for s in range(n_slices)]

        if len(self.samples) == 0 and not allow_empty:
            raise RuntimeError(
                f"Empty Synapse {split} split - check split.json and the data "
                f"directory (val is empty by design after --val_cases 0)"
            )

        if self.is_train:
            self.color_jitter = transforms.ColorJitter(
                brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1
            )

    def __len__(self):
        return len(self.samples)

    @property
    def n_cases(self):
        return len({s[1] for s in self.samples})

    @classmethod
    def _normalize_ct(cls, image):
        img = np.clip(image.astype(np.float32), cls.CT_CLIP_MIN, cls.CT_CLIP_MAX)
        return (img - cls.CT_CLIP_MIN) / (cls.CT_CLIP_MAX - cls.CT_CLIP_MIN)

    def _load_sample(self, fpath, slice_idx):
        if slice_idx is None:
            data  = np.load(fpath)
            image = data['image'].astype(np.float32)
            label = data['label'].astype(np.int64)
        else:
            with h5py.File(fpath, 'r') as f:
                image = f['image'][slice_idx].astype(np.float32)
                label = f['label'][slice_idx].astype(np.int64)
        if image.max() > 1.5 or image.min() < -0.5:
            image = self._normalize_ct(image)
        return image, label

    @staticmethod
    def _resize_pair(image_t, label_t, size):
        if image_t.shape[-2:] == (size, size):
            return image_t, label_t
        image_t = TF.resize(image_t, [size, size],
                            interpolation=transforms.InterpolationMode.BILINEAR,
                            antialias=True)
        label_t = TF.resize(label_t.unsqueeze(0).float(), [size, size],
                            interpolation=transforms.InterpolationMode.NEAREST
                            ).squeeze(0).long()
        return image_t, label_t

    def _geometric_augmentation(self, image_t, label_t):
        H = self.image_size

        if random.random() > 0.5:
            k = {90: 1, 180: 2, 270: 3}[random.choice([90, 180, 270])]
            image_t = torch.rot90(image_t, k, dims=[1, 2])
            label_t = torch.rot90(label_t, k, dims=[0, 1])

        if random.random() > 0.5:
            image_t = torch.flip(image_t, dims=[2])
            label_t = torch.flip(label_t, dims=[1])

        if random.random() > 0.5:
            image_t = torch.flip(image_t, dims=[1])
            label_t = torch.flip(label_t, dims=[0])

        if random.random() > 0.3:
            scale    = random.uniform(0.8, 1.2)
            new_size = int(H * scale)
            image_t, label_t = self._resize_pair(image_t, label_t, new_size)
            if scale > 1.0:
                i, j, h, w = transforms.RandomCrop.get_params(image_t, (H, H))
                image_t = TF.crop(image_t, i, j, h, w)
                label_t = TF.crop(label_t.unsqueeze(0), i, j, h, w).squeeze(0)
            else:
                image_t, label_t = self._resize_pair(image_t, label_t, H)

        return image_t.contiguous(), label_t.contiguous()

    def _photometric_augmentation(self, image_3ch):
        if random.random() > 0.3:
            image_3ch = self.color_jitter(image_3ch)
        if random.random() > 0.7:
            image_3ch = TF.gaussian_blur(image_3ch, random.choice([3, 5]))
        return image_3ch.clamp(0.0, 1.0)

    def __getitem__(self, idx):
        fpath, case_id, slice_idx, fname = self.samples[idx]
        image, label = self._load_sample(fpath, slice_idx)

        label_native = (torch.from_numpy(label.copy()).long()
                        if self.native_label else None)

        image_t = torch.from_numpy(np.ascontiguousarray(image)).float().unsqueeze(0)
        label_t = torch.from_numpy(np.ascontiguousarray(label)).long()
        image_t, label_t = self._resize_pair(image_t, label_t, self.image_size)
        if self.is_train:
            image_t, label_t = self._geometric_augmentation(image_t, label_t)

        image_3ch = image_t.repeat(3, 1, 1).clamp(0.0, 1.0)
        if self.is_train:
            image_3ch = self._photometric_augmentation(image_3ch)

        label_t = label_t.clamp(0, self.NUM_CLASSES - 1)
        onehot  = torch.nn.functional.one_hot(label_t, self.NUM_CLASSES)
        onehot  = onehot.permute(2, 0, 1).float()

        out = {
            'image':     image_3ch * 2.0 - 1.0,
            'mask':      onehot * 2.0 - 1.0,
            'label_idx': label_t,
            'filename':  fname,
            'case_id':   case_id,
        }
        if label_native is not None:
            out['label_full'] = label_native.clamp(0, self.NUM_CLASSES - 1)
        return out


def synapse_has_val(root_dir: str = None, split_json: str = None) -> bool:
    if split_json is None:
        split_json = os.path.join(root_dir or _DATASET_ROOT['synapse'], 'split.json')
    try:
        with open(split_json, 'r', encoding='utf-8') as f:
            return len(json.load(f).get('val', {}).get('cases', [])) > 0
    except (OSError, ValueError):
        return False


def _make_dataset(dataset_name: str, split: str, image_size: int):
    root = get_dataset_root(dataset_name)
    if is_multiclass(dataset_name):
        return SynapseDataset(root, split, image_size,
                              allow_empty=(split == 'val'))
    return MedicalSegmentationDataset(root, split, image_size)


def _make_loader(dataset, batch_size, shuffle, num_workers, drop_last=False):
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=drop_last,
    )


def get_dataloader(dataset_name: str, batch_size: int = 8, image_size: int = 256,
                   num_workers: int = 4):
    return (
        _make_loader(_make_dataset(dataset_name, 'train', image_size),
                     batch_size, True, num_workers, drop_last=True),
        _make_loader(_make_dataset(dataset_name, 'test', image_size),
                     batch_size, False, num_workers),
    )


def get_dataloader_with_val(dataset_name: str, batch_size: int = 8,
                            image_size: int = 256, num_workers: int = 4):
    val_ds = _make_dataset(dataset_name, 'val', image_size)
    return (
        _make_loader(_make_dataset(dataset_name, 'train', image_size),
                     batch_size, True, num_workers, drop_last=True),
        (_make_loader(val_ds, batch_size, False, num_workers)
         if len(val_ds) > 0 else None),
        _make_loader(_make_dataset(dataset_name, 'test', image_size),
                     batch_size, False, num_workers),
    )


SYNAPSE_CLASS_NAMES = ['background', 'aorta', 'gallbladder', 'kidney_left',
                       'kidney_right', 'liver', 'pancreas', 'spleen', 'stomach']


def _parse_split_args():
    p = argparse.ArgumentParser(description='Generate the Synapse train/val/test split.')
    p.add_argument('--root', type=str, default=_DATASET_ROOT['synapse'],
                   help='Synapse root directory (holds train_npz / test_vol_h5)')
    p.add_argument('--val_cases', type=int, default=0,
                   help='How many of the 18 training cases to hold out for '
                        'validation (0 = the protocol used in the paper)')
    p.add_argument('--seed', type=int, default=42,
                   help='Seed of the case shuffle, so the split is reproducible')
    p.add_argument('--output', type=str, default=None,
                   help='Output path (default: <root>/split.json)')
    return p.parse_args()


def prepare_synapse_split():
    args = _parse_split_args()

    train_npz_dir = os.path.join(args.root, 'train_npz')
    test_h5_dir   = os.path.join(args.root, 'test_vol_h5')
    for d in (train_npz_dir, test_h5_dir):
        if not os.path.isdir(d):
            raise FileNotFoundError(f"Directory not found: {d}")

    npz_files = [f for f in os.listdir(train_npz_dir) if f.endswith('.npz')]
    all_train_cases = sorted({f.split('_')[0] for f in npz_files})

    test_files = sorted(f for f in os.listdir(test_h5_dir) if f.endswith('.h5'))
    test_cases = [f.split('.')[0] for f in test_files]

    if args.val_cases >= len(all_train_cases):
        raise ValueError(f"--val_cases {args.val_cases} must be smaller than the "
                         f"{len(all_train_cases)} available training cases")

    shuffled = list(all_train_cases)
    random.Random(args.seed).shuffle(shuffled)
    val_cases   = sorted(shuffled[:args.val_cases])
    train_cases = sorted(shuffled[args.val_cases:])

    n_train_slices = sum(1 for f in npz_files if f.split('_')[0] in train_cases)
    n_val_slices   = sum(1 for f in npz_files if f.split('_')[0] in val_cases)

    split = {
        'description': 'Synapse multi-organ segmentation '
                       '(9 classes: background + aorta, gallbladder, left kidney, '
                       'right kidney, liver, pancreas, spleen, stomach)',
        'num_classes': len(SYNAPSE_CLASS_NAMES),
        'class_names': SYNAPSE_CLASS_NAMES,
        'seed': args.seed,
        'val_cases_n': args.val_cases,
        'train': {'cases': train_cases, 'n_cases': len(train_cases),
                  'n_slices': n_train_slices, 'source': 'train_npz'},
        'val':   {'cases': val_cases, 'n_cases': len(val_cases),
                  'n_slices': n_val_slices, 'source': 'train_npz'},
        'test':  {'cases': test_cases, 'files': test_files,
                  'n_cases': len(test_cases), 'source': 'test_vol_h5 (3D volumes)'},
    }

    out_path = args.output or os.path.join(args.root, 'split.json')
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(split, f, indent=2)

    print(f"Train cases ({len(train_cases)}): {train_cases}")
    print(f"Val   cases ({len(val_cases)}): {val_cases}")
    print(f"Test  cases ({len(test_cases)}): {test_cases}")
    print(f"Train slices (2D npz): {n_train_slices}")
    print(f"Val   slices (2D npz): {n_val_slices}")
    print(f"Test  volumes (3D h5): {len(test_cases)}")
    if not val_cases:
        print("No validation split: training runs for the full --epochs and "
              "final_model.pth is the model to evaluate.")
    print(f"\nSplit written to {out_path}")


if __name__ == '__main__':
    prepare_synapse_split()
