"""
Stage 2: Preprocessing

Loads all images from data/, detects and crops faces using MediaPipe,
resizes to 96x96, normalizes to [-1,1], and saves as NumPy arrays.

Test-split policy: augmented variants are skipped when loading the test
split. Augmentation must apply to train only — see the F1 finding in
docs/decision.md and the references in
docs/report/methods_test_hygiene.md (Goodfellow §5.3/§7.4; Kapoor &
Narayanan 2023, arXiv:2207.07048; Sculley et al. NeurIPS 2015). The
filter is defence-in-depth: even if augmented files reappear under
data/<person>/test/, they will not enter ``x_test.npy``.
"""

import os
import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks.python import vision
from mediapipe.tasks.python.core.base_options import BaseOptions

from utils.train_val_split import AUG_SUFFIXES

DATA_DIR = os.path.join(os.path.dirname(__file__), '..', 'data')
GEN_DIR = os.path.join(os.path.dirname(__file__), 'gen')
FACE_MODEL_PATH = os.path.join(os.path.dirname(__file__), 'blaze_face_short_range.tflite')

IMG_SIZE = 96
IMAGE_EXTENSIONS = {'.png', '.jpg', '.jpeg', '.bmp'}

# Label mapping (alphabetical order for reproducibility)
LABELS = {'Amine': 0, 'Rifki': 1, 'Jakub': 2}
NUM_CLASSES = len(LABELS)


def _is_augmented_filename(filename: str) -> bool:
    """Return True if the filename's stem ends with a known augmentation suffix."""
    stem = os.path.splitext(filename)[0].lower()
    return any(stem.endswith(suffix) for suffix in AUG_SUFFIXES)


def detect_and_crop_face(img_rgb: np.ndarray, detector: vision.FaceDetector) -> np.ndarray:
    """Detect face using MediaPipe Tasks API and crop. Falls back to center crop."""
    h, w, _ = img_rgb.shape

    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=img_rgb)
    result = detector.detect(mp_image)

    if result.detections:
        bbox = result.detections[0].bounding_box

        # Apply 15% padding around the detected face
        pad_x = int(bbox.width * 0.15)
        pad_y = int(bbox.height * 0.15)
        x_min = max(0, bbox.origin_x - pad_x)
        y_min = max(0, bbox.origin_y - pad_y)
        x_max = min(w, bbox.origin_x + bbox.width + pad_x)
        y_max = min(h, bbox.origin_y + bbox.height + pad_y)

        if x_max > x_min and y_max > y_min:
            return img_rgb[y_min:y_max, x_min:x_max]

    # Fallback: center crop (largest centered square)
    if h > w:
        offset = (h - w) // 2
        return img_rgb[offset:offset + w, :]
    else:
        offset = (w - h) // 2
        return img_rgb[:, offset:offset + h]


def load_and_preprocess_split(person_dir: str, split: str,
                              detector: vision.FaceDetector) -> tuple[list[np.ndarray], list[int]]:
    """Load all images from a person/split folder, preprocess them."""
    split_dir = os.path.join(person_dir, split)
    if not os.path.isdir(split_dir):
        return [], []

    person_name = os.path.basename(person_dir)
    if person_name not in LABELS:
        print(f'  Warning: Unknown person "{person_name}", skipping')
        return [], []

    label = LABELS[person_name]
    images = []
    labels = []

    for filename in sorted(os.listdir(split_dir)):
        ext = os.path.splitext(filename)[1].lower()
        if ext not in IMAGE_EXTENSIONS:
            continue

        # Test-split hygiene (F1 fix): never load augmented variants into
        # the test arrays. See module docstring.
        if split == 'test' and _is_augmented_filename(filename):
            continue

        img_path = os.path.join(split_dir, filename)
        img_bgr = cv2.imread(img_path)
        if img_bgr is None:
            continue

        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

        # Detect and crop face
        face_crop = detect_and_crop_face(img_rgb, detector)

        # Resize to target size
        face_resized = cv2.resize(face_crop, (IMG_SIZE, IMG_SIZE),
                                  interpolation=cv2.INTER_AREA)

        # Normalize to [-1, 1] (expected by MobileNetV2 pretrained on ImageNet)
        face_normalized = face_resized.astype(np.float32) / 127.5 - 1.0

        images.append(face_normalized)
        labels.append(label)

    return images, labels


def preprocess_all(data_dir: str = DATA_DIR, gen_dir: str = GEN_DIR):
    """Process all person folders and save train/test arrays."""
    data_dir = os.path.abspath(data_dir)
    gen_dir = os.path.abspath(gen_dir)
    os.makedirs(gen_dir, exist_ok=True)

    print(f'Preprocessing images from: {data_dir}')
    print(f'Output directory: {gen_dir}')
    print(f'Target image size: {IMG_SIZE}x{IMG_SIZE}')
    print(f'Labels: {LABELS}')
    print()

    # Initialize MediaPipe Face Detection (Tasks API)
    options = vision.FaceDetectorOptions(
        base_options=BaseOptions(model_asset_path=FACE_MODEL_PATH),
        min_detection_confidence=0.5
    )
    detector = vision.FaceDetector.create_from_options(options)

    train_images, train_labels = [], []
    test_images, test_labels = [], []

    for person in sorted(os.listdir(data_dir)):
        person_dir = os.path.join(data_dir, person)
        if not os.path.isdir(person_dir):
            continue
        # Skip quarantine and other meta folders (e.g. ``_quarantine``).
        if person.startswith('_'):
            continue

        # Process train split
        imgs, lbls = load_and_preprocess_split(person_dir, 'train', detector)
        train_images.extend(imgs)
        train_labels.extend(lbls)
        print(f'  {person}/train: {len(imgs)} images')

        # Process test split
        imgs, lbls = load_and_preprocess_split(person_dir, 'test', detector)
        test_images.extend(imgs)
        test_labels.extend(lbls)
        print(f'  {person}/test: {len(imgs)} images')

    detector.close()

    # Convert to NumPy arrays
    x_train = np.array(train_images, dtype=np.float32)
    y_train = np.array(train_labels, dtype=np.int32)
    x_test = np.array(test_images, dtype=np.float32)
    y_test = np.array(test_labels, dtype=np.int32)

    # Save
    np.save(os.path.join(gen_dir, 'x_train.npy'), x_train)
    np.save(os.path.join(gen_dir, 'y_train.npy'), y_train)
    np.save(os.path.join(gen_dir, 'x_test.npy'), x_test)
    np.save(os.path.join(gen_dir, 'y_test.npy'), y_test)

    print()
    print(f'Saved: x_train {x_train.shape}, y_train {y_train.shape}')
    print(f'Saved: x_test  {x_test.shape}, y_test  {y_test.shape}')

    # Print class distribution
    test_counts: list[int] = []
    for name, idx in LABELS.items():
        n_train = np.sum(y_train == idx)
        n_test = np.sum(y_test == idx)
        test_counts.append(int(n_test))
        print(f'  {name} (label {idx}): {n_train} train, {n_test} test')

    # Hygiene assertion (F1 fix): after the augmentation-skip filter the
    # test set must hold only original captures, balanced per class.
    if test_counts and len(set(test_counts)) != 1:
        raise RuntimeError(
            'preprocess.py: test-split per-class counts are not balanced: '
            f'{dict(zip(LABELS.keys(), test_counts, strict=True))}. '
            'Either originals are missing or augmented files leaked in. '
            'Run `python tools/clean_test_augmentations.py --quarantine`.'
        )


if __name__ == '__main__':
    preprocess_all()
