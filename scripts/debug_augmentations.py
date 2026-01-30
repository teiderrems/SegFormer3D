#!/usr/bin/env python3
"""Debug tool to verify that augmentations are applied to the training dataset.

Usage:
    python scripts/debug_augmentations.py --config configs/config_segformer3d.yaml --n 10

The script will:
- Load dataset configuration
- Build train dataset (not dataloader)
- For the first N samples it will fetch the raw sample (without transform) and the transformed
  sample (with transform) and report whether they differ (by comparing tensors).
"""
import argparse
import sys
import os
from pathlib import Path
import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dataloaders.build_dataset import build_dataset


def tensors_differ(a, b):
    try:
        import torch
    except Exception:
        # If torch not available, do string comparison
        return str(a) != str(b)

    if a is None or b is None:
        return a != b
    try:
        if isinstance(a, torch.Tensor) and isinstance(b, torch.Tensor):
            return not torch.equal(a, b)
        # Handle lists/tuples
        if isinstance(a, (list, tuple)) and isinstance(b, (list, tuple)):
            if len(a) != len(b):
                return True
            for ai, bi in zip(a, b):
                if tensors_differ(ai, bi):
                    return True
            return False
    except Exception:
        return str(a) != str(b)
    return str(a) != str(b)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--n", type=int, default=10, help="Number of samples to check")
    args = parser.parse_args()

    with open(args.config, 'r', encoding='utf-8') as f:
        cfg = yaml.safe_load(f) or {}

    # Build train dataset args according to config format
    if 'dataset_parameters' in cfg:
        ds_cfg = cfg['dataset_parameters']
        dataset_type = ds_cfg.get('dataset_type', 'prostate_seg')
        train_args = ds_cfg.get('train_dataset_args', {})
    else:
        # Old format not supported here
        print('Config missing dataset_parameters')
        return

    print(f"Dataset type: {dataset_type}")
    print(f"Train dataset args (summary): root={train_args.get('root')}, augmentations={train_args.get('augmentations')}")

    # Build dataset (this will use the transform as configured)
    ds = build_dataset(dataset_type, train_args)

    # Temporarily disable transforms to get raw sample
    original_transform = getattr(ds, 'transform', None)

    count = min(args.n, len(ds))

    for i in range(count):
        # raw sample
        ds.transform = None
        raw = ds[i]
        # transformed sample (may be dict or list of dicts)
        ds.transform = original_transform
        aug = ds[i]

        # Normalize types: raw is dict, aug may be dict or list
        def _get_samples(x):
            if isinstance(x, list):
                return x
            return [x]

        aug_samples = _get_samples(aug)

        # If any augmented sample differs from raw, we report changed=True
        image_changed = False
        label_changed = False
        for s in aug_samples:
            # safe access
            img_s = s['image'] if isinstance(s, dict) and 'image' in s else s
            lbl_s = s['label'] if isinstance(s, dict) and 'label' in s else None
            if tensors_differ(raw.get('image'), img_s):
                image_changed = True
            if lbl_s is not None and tensors_differ(raw.get('label'), lbl_s):
                label_changed = True

        print(f"Sample {i}: image_changed={image_changed}, label_changed={label_changed}")

    # restore
    ds.transform = original_transform

if __name__ == '__main__':
    main()
