"""
Stage 2: Preprocessing

Loads all images from data/, detects and crops faces using MediaPipe,
resizes to 96x96, normalizes to [0,1], and saves as NumPy arrays.
"""

import os
import cv2
import numpy as np
import mediapipe as mp

DATA_DIR = os.path.join(os.path.dirname(__file__), '..', 'data')
GEN_DIR = os.path.join(os.path.dirname(__file__), 'gen')

IMG_SIZE = 96
IMAGE_EXTENSIONS = {'.png', '.jpg', '.jpeg', '.bmp'}

# Label mapping (alphabetical order for reproducibility)
LABELS = {'Amine': 0, 'Rifki': 1, 'Jakub': 2}
NUM_CLASSES = len(LABELS)


def detect_and_crop_face(img_rgb: np.ndarray, face_detection) -> np.ndarray:
    """Detect face using MediaPipe and crop. Falls back to center crop."""
    h, w, _ = img_rgb.shape
    results = face_detection.process(img_rgb)

    if results.detections:
        detection = results.detections[0]
        bbox = detection.location_data.relative_bounding_box

        # Convert relative coordinates to absolute with padding
        pad = 0.15  # 15% padding around the face
        x_min = max(0, int((bbox.xmin - pad) * w))
        y_min = max(0, int((bbox.ymin - pad) * h))
        x_max = min(w, int((bbox.xmin + bbox.width + pad) * w))
        y_max = min(h, int((bbox.ymin + bbox.height + pad) * h))

        # Ensure we have a valid crop region
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
                              face_detection) -> tuple[list[np.ndarray], list[int]]:
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

        img_path = os.path.join(split_dir, filename)
        img_bgr = cv2.imread(img_path)
        if img_bgr is None:
            continue

        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

        # Detect and crop face
        face_crop = detect_and_crop_face(img_rgb, face_detection)

        # Resize to target size
        face_resized = cv2.resize(face_crop, (IMG_SIZE, IMG_SIZE),
                                  interpolation=cv2.INTER_AREA)

        # Normalize to [0, 1]
        face_normalized = face_resized.astype(np.float32) / 255.0

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

    # Initialize MediaPipe Face Detection
    mp_face_detection = mp.solutions.face_detection
    face_detection = mp_face_detection.FaceDetection(
        model_selection=1,  # Full-range model (better for varied distances)
        min_detection_confidence=0.5
    )

    train_images, train_labels = [], []
    test_images, test_labels = [], []

    for person in sorted(os.listdir(data_dir)):
        person_dir = os.path.join(data_dir, person)
        if not os.path.isdir(person_dir):
            continue

        # Process train split
        imgs, lbls = load_and_preprocess_split(person_dir, 'train', face_detection)
        train_images.extend(imgs)
        train_labels.extend(lbls)
        print(f'  {person}/train: {len(imgs)} images')

        # Process test split
        imgs, lbls = load_and_preprocess_split(person_dir, 'test', face_detection)
        test_images.extend(imgs)
        test_labels.extend(lbls)
        print(f'  {person}/test: {len(imgs)} images')

    face_detection.close()

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
    for name, idx in LABELS.items():
        n_train = np.sum(y_train == idx)
        n_test = np.sum(y_test == idx)
        print(f'  {name} (label {idx}): {n_train} train, {n_test} test')


if __name__ == '__main__':
    preprocess_all()
