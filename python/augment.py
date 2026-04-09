"""
Stage 1: Data Augmentation

Walks data/ directory, finds person folders with train/test splits,
and applies augmentations per image. Idempotent — skips existing files.

By default only train/ is augmented so test metrics stay on clean held-out
images. Use --include-test only if you intentionally want augmented test
data (e.g. TTA-style experiments).
"""

import argparse
import os
from typing import Tuple

import albumentations as A
import cv2

DATA_DIR = os.path.join(os.path.dirname(__file__), '..', 'data')

# Define augmentations with their suffixes (each applied independently from the original)
AUGMENTATIONS = {
    '_hflip': A.HorizontalFlip(p=1.0),
    '_rot': A.Rotate(limit=15, p=1.0, border_mode=cv2.BORDER_REFLECT_101),
    # Albumentations 2.x: Affine replaces ShiftScaleRotate for shift+scale (no rotation here)
    '_shiftscale': A.Affine(
        scale=(0.9, 1.1),
        translate_percent={'x': (-0.1, 0.1), 'y': (-0.1, 0.1)},
        rotate=(0.0, 0.0),
        shear=(0.0, 0.0),
        border_mode=cv2.BORDER_REFLECT_101,
        p=1.0,
    ),
    '_bright': A.RandomBrightnessContrast(
        brightness_limit=0.2, contrast_limit=0.15, p=1.0
    ),
    '_hue': A.HueSaturationValue(
        hue_shift_limit=10, sat_shift_limit=20, val_shift_limit=0, p=1.0
    ),
    '_blur': A.GaussianBlur(blur_limit=(3, 7), sigma_limit=(1, 3), p=1.0),
    '_compress': A.ImageCompression(
        compression_type='jpeg', quality_range=(70, 90), p=1.0
    ),
    # Mild occlusion: small rare patches so identity features are not wiped out
    '_occlude': A.CoarseDropout(
        num_holes_range=(1, 2),
        hole_height_range=(4, 8),
        hole_width_range=(4, 8),
        fill=0,
        p=1.0,
    ),
    '_gray': A.ToGray(p=1.0),
}

IMAGE_EXTENSIONS = {'.png', '.jpg', '.jpeg', '.bmp'}


def is_original_image(filename: str) -> bool:
    """Check if a file is an original image (not an augmented one)."""
    name = os.path.splitext(filename)[0]
    for suffix in AUGMENTATIONS:
        if name.endswith(suffix):
            return False
    return True


def _rgb_to_bgr_for_write(image_rgb):
    """Convert model output to BGR for cv2.imwrite (handles 1- or 3-channel)."""
    if image_rgb.ndim == 2:
        return cv2.cvtColor(image_rgb, cv2.COLOR_GRAY2BGR)
    if image_rgb.shape[2] == 1:
        return cv2.cvtColor(image_rgb, cv2.COLOR_GRAY2BGR)
    return cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)


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

        for suffix, transform in AUGMENTATIONS.items():
            out_filename = f'{name_no_ext}{suffix}.png'
            out_path = os.path.join(folder_path, out_filename)

            # Skip if already exists (idempotent)
            if os.path.exists(out_path):
                continue

            augmented = transform(image=img_rgb)['image']
            augmented_bgr = _rgb_to_bgr_for_write(augmented)
            cv2.imwrite(out_path, augmented_bgr)
            num_created += 1

    return num_originals, num_created


def main():
    parser = argparse.ArgumentParser(description='Offline face dataset augmentation')
    parser.add_argument(
        '--include-test',
        action='store_true',
        help='Also augment test/ splits (default: train only)',
    )
    args = parser.parse_args()

    data_dir = os.path.abspath(DATA_DIR)
    if not os.path.isdir(data_dir):
        print(f'Data directory not found: {data_dir}')
        print('Please create data/<Person>/train/ and data/<Person>/test/ folders with images.')
        return

    splits = ['train', 'test'] if args.include_test else ['train']

    print(f'Augmenting images in: {data_dir}')
    print(f'Splits: {", ".join(splits)}')
    print(f'Augmentations per image: {len(AUGMENTATIONS)}')
    print()

    total_originals = 0
    total_created = 0

    for person in sorted(os.listdir(data_dir)):
        person_dir = os.path.join(data_dir, person)
        if not os.path.isdir(person_dir):
            continue

        for split in splits:
            split_dir = os.path.join(person_dir, split)
            if not os.path.isdir(split_dir):
                continue

            num_originals, num_created = augment_folder(split_dir)
            total_originals += num_originals
            total_created += num_created
            total = num_originals * (1 + len(AUGMENTATIONS))
            print(f'  {person}/{split}: {num_originals} originals -> {total} total ({num_created} new)')

    print()
    print(f'Summary: {total_originals} originals, {total_created} new augmented images created')
    print(f'Each original has {len(AUGMENTATIONS)} augmented variants')


if __name__ == '__main__':
    main()
