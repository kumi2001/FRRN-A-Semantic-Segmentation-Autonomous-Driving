"""Dataset helpers for controlled synthetic smoke tests and authorized labeled images."""
from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
import torch


class ImageMaskDataset(Dataset):
    """Loads 8-bit RGB PNG/JPG inputs and single-channel integer-class PNG masks.

    Input folder: root/{split}/images, root/{split}/masks
    Mask values must be 0..num_classes-1 or 255 (ignore).
    """
    def __init__(self, root: str | Path, split: str, size: tuple[int, int], num_classes: int):
        self.path = Path(root) / split
        self.size = size  # height, width
        self.num_classes = num_classes
        image_dir = self.path / 'images'
        mask_dir = self.path / 'masks'
        if not image_dir.is_dir() or not mask_dir.is_dir():
            raise FileNotFoundError(f'Missing {image_dir} or {mask_dir}')
        extensions = {'.png', '.jpg', '.jpeg'}
        images = sorted(p for p in image_dir.iterdir() if p.suffix.lower() in extensions)
        self.pairs = []
        for im in images:
            mask = mask_dir / (im.stem + '.png')
            if mask.is_file():
                self.pairs.append((im, mask))
        if not self.pairs:
            raise ValueError(f'No image/mask pairs in {self.path}')
        if len(self.pairs) != len(images):
            raise ValueError(f'{len(images)-len(self.pairs)} image(s) missing corresponding masks in {self.path}')

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, index: int):
        im, mask = self.pairs[index]
        h, w = self.size
        rgb = np.asarray(Image.open(im).convert('RGB').resize((w, h), Image.Resampling.BILINEAR), dtype=np.uint8).copy()
        labels = np.asarray(Image.open(mask).resize((w, h), Image.Resampling.NEAREST), dtype=np.int64).copy()
        if labels.ndim != 2:
            raise ValueError(f'{mask} must be a single-channel ID mask, not an RGB visualization')
        invalid = (labels >= self.num_classes) & (labels != 255)
        if invalid.any():
            bad = np.unique(labels[invalid]).tolist()[:10]
            raise ValueError(f'{mask}: unsupported label values {bad}; remap class IDs first')
        return torch.from_numpy(rgb).permute(2,0,1).float() / 255.0, torch.from_numpy(labels)


def make_synthetic(root: Path, train_count=12, val_count=4, height=32, width=64, seed=42):
    """Generate deterministic, unmistakably synthetic road-like toy images/masks."""
    rng = np.random.default_rng(seed)
    root = Path(root)
    for split, count in [('train', train_count), ('val', val_count)]:
        for folder in ['images', 'masks']:
            (root / split / folder).mkdir(parents=True, exist_ok=True)
        for idx in range(count):
            labels = np.zeros((height,width),dtype=np.uint8)
            road_start = int(height * (0.49 + rng.uniform(-0.07,0.07)))
            labels[road_start:] = 1 # road
            vx = int(rng.integers(3,max(4,width-11)))
            vy = int(rng.integers(max(0,road_start-4),max(1,height-8)))
            labels[vy:min(vy+7,height), vx:min(vx+10,width)] = 2 # vehicle-like rectangle
            px = int(rng.integers(1,width-3))
            py = int(rng.integers(2, max(3,road_start-9)))
            labels[py:py+8,px:px+3] = 3 # pedestrian-like rectangle
            colors = np.array([[125,180,218],[75,79,82],[216,62,45],[214,189,118]],dtype=np.uint8)
            img = np.clip(colors[labels].astype(np.float32)+rng.normal(0,8,(height,width,3)),0,255).astype(np.uint8)
            name=f'{idx:04}.png'
            Image.fromarray(img,'RGB').save(root / split / 'images' / name)
            Image.fromarray(labels,'L').save(root / split / 'masks' / name)


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--output',type=Path,default=Path('artifacts/toy_data'))
    p.add_argument('--train-count',type=int,default=12)
    p.add_argument('--val-count',type=int,default=4)
    p.add_argument('--height',type=int,default=32)
    p.add_argument('--width',type=int,default=64)
    a=p.parse_args()
    make_synthetic(a.output,a.train_count,a.val_count,a.height,a.width)
    print('Synthetic sample dataset written to',a.output)

if __name__=='__main__':main()
