"""
Stage 1: Data Augmentation

Walks data/ directory, finds person folders with train/test splits,
and applies 8 augmentations per image. Idempotent — skips existing files.
"""

import os
from typing import Tuple
import cv2
import numpy as np
import albumentations as A

DATA_DIR = os.path.join(os.path.dirname(__file__), '..', 'data')

# Define augmentations with their suffixes
AUGMENTATIONS = {
    '_hflip': A.HorizontalFlip(p=1.0),
    '_rot': A.Rotate(limit=15, p=1.0, border_mode=cv2.BORDER_REFLECT_101),
    '_shiftscale': A.Affine(
        translate_percent=(-0.1, 0.1), scale=(0.9, 1.1), rotate=0, p=1.0,
        border_mode=cv2.BORDER_REFLECT_101
    ),
    '_bright': A.RandomBrightnessContrast(
        brightness_limit=0.2, contrast_limit=0.15, p=1.0
    ),
    '_blur': A.GaussianBlur(blur_limit=(3, 7), sigma_limit=(1, 3), p=1.0),
    '_compress': A.ImageCompression(quality_range=(70, 90), p=1.0),
    '_occlude': A.CoarseDropout(
        num_holes_range=(2, 5),
        hole_height_range=(8, 16), hole_width_range=(8, 16),
        fill=0, p=1.0
    ),
    '_gray': A.ToGray(p=1.0),
}

# Combined augmentation pipelines — apply multiple transforms together
# for more realistic variability (crucial for small datasets)
COMBINED_AUGMENTATIONS = {
    '_combo1': A.Compose([
        A.HorizontalFlip(p=0.5),
        A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.15, p=0.7),
        A.GaussianBlur(blur_limit=(3, 5), p=0.3),
    ]),
    '_combo2': A.Compose([
        A.Affine(
            translate_percent=(-0.05, 0.05), scale=(0.9, 1.1), rotate=(-10, 10), p=0.8,
            border_mode=cv2.BORDER_REFLECT_101
        ),
        A.RandomBrightnessContrast(brightness_limit=0.15, contrast_limit=0.1, p=0.5),
        A.CoarseDropout(
            num_holes_range=(1, 3),
            hole_height_range=(6, 12), hole_width_range=(6, 12),
            fill=0, p=0.4
        ),
    ]),
    '_combo3': A.Compose([
        A.HorizontalFlip(p=0.5),
        A.Rotate(limit=10, p=0.6, border_mode=cv2.BORDER_REFLECT_101),
        A.ImageCompression(quality_range=(75, 95), p=0.4),
    ]),
    '_combo4': A.Compose([
        A.ToGray(p=0.3),
        A.RandomBrightnessContrast(brightness_limit=0.25, contrast_limit=0.2, p=0.7),
        A.Affine(
            translate_percent=(-0.05, 0.05), scale=(0.95, 1.05), rotate=(-5, 5), p=0.5,
            border_mode=cv2.BORDER_REFLECT_101
        ),
    ]),
}

ALL_AUGMENTATIONS = {**AUGMENTATIONS, **COMBINED_AUGMENTATIONS}

IMAGE_EXTENSIONS = {'.png', '.jpg', '.jpeg', '.bmp'}


def is_original_image(filename: str) -> bool:
    """Check if a file is an original image (not an augmented one)."""
    name = os.path.splitext(filename)[0]
    for suffix in ALL_AUGMENTATIONS:
        if name.endswith(suffix):
            return False
    return True


def augment_folder(folder_path: str) -> Tuple[int, int]:
    """Apply all augmentations to original images in a folder.

    Returns (num_originals, num_created) count.
    """
    num_originals = 0
    num_created = 0

    files = sorted(os.listdir(folder_path))
    for filename in files:
        ext = os.path.splitext(filename)[1].lower()
        if ext not in IMAGE_EXTENSIONS:
            continue
        if not is_original_image(filename):
            continue

        num_originals += 1
        name_no_ext = os.path.splitext(filename)[0]
        img_path = os.path.join(folder_path, filename)
        img = cv2.imread(img_path)
        if img is None:
            print(f'  Warning: Could not read {img_path}, skipping')
            continue

        # Convert BGR to RGB for albumentations
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        for suffix, transform in ALL_AUGMENTATIONS.items():
            out_filename = f'{name_no_ext}{suffix}.png'
            out_path = os.path.join(folder_path, out_filename)

            # Skip if already exists (idempotent)
            if os.path.exists(out_path):
                continue

            augmented = transform(image=img_rgb)['image']
            # Convert back to BGR for cv2.imwrite
            augmented_bgr = cv2.cvtColor(augmented, cv2.COLOR_RGB2BGR)
            cv2.imwrite(out_path, augmented_bgr)
            num_created += 1

    return num_originals, num_created


def main():
    data_dir = os.path.abspath(DATA_DIR)
    if not os.path.isdir(data_dir):
        print(f'Data directory not found: {data_dir}')
        print('Please create data/<Person>/train/ and data/<Person>/test/ folders with images.')
        return

    print(f'Augmenting images in: {data_dir}')
    print(f'Augmentations per image: {len(ALL_AUGMENTATIONS)}')
    print()

    total_originals = 0
    total_created = 0

    # Walk person folders
    for person in sorted(os.listdir(data_dir)):
        person_dir = os.path.join(data_dir, person)
        if not os.path.isdir(person_dir):
            continue

        for split in ['train', 'test']:
            split_dir = os.path.join(person_dir, split)
            if not os.path.isdir(split_dir):
                continue

            num_originals, num_created = augment_folder(split_dir)
            total_originals += num_originals
            total_created += num_created
            total = num_originals * (1 + len(ALL_AUGMENTATIONS))
            print(f'  {person}/{split}: {num_originals} originals -> {total} total ({num_created} new)')

    print()
    print(f'Summary: {total_originals} originals, {total_created} new augmented images created')
    print(f'Each original has {len(ALL_AUGMENTATIONS)} augmented variants')


if __name__ == '__main__':
    main()
