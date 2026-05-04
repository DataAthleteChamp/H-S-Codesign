"""Build original-capture-only test arrays for calibration."""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import cv2
import numpy as np

try:
    import mediapipe as mp
    from mediapipe.tasks.python import vision
    from mediapipe.tasks.python.core.base_options import BaseOptions
except Exception as exc:  # pragma: no cover - exercised only when MediaPipe is unavailable.
    mp = None
    vision = None
    BaseOptions = None
    MEDIAPIPE_IMPORT_ERROR = exc
else:
    MEDIAPIPE_IMPORT_ERROR = None

CLASSES = ("Amine", "Rifki", "Jakub")
LABELS = {"Amine": 0, "Rifki": 1, "Jakub": 2}
AUG_SUFFIXES = (
    "_hflip",
    "_rot",
    "_shiftscale",
    "_bright",
    "_blur",
    "_compress",
    "_occlude",
    "_gray",
    "_combo1",
    "_combo2",
    "_combo3",
    "_combo4",
)
IMAGE_RE = re.compile(r"^(.+?)(\.(png|jpg|jpeg|bmp))$", re.IGNORECASE)


def _warn(message: str) -> None:
    print(f"warning: {message}", file=sys.stderr)


def is_original_filename(filename: str) -> bool:
    match = IMAGE_RE.match(filename)
    if not match:
        return False
    stem = match.group(1).lower()
    return not any(stem.endswith(suffix) for suffix in AUG_SUFFIXES)


def center_crop(img_rgb: np.ndarray) -> np.ndarray:
    h, w, _ = img_rgb.shape
    if h > w:
        offset = (h - w) // 2
        return img_rgb[offset : offset + w, :]
    offset = (w - h) // 2
    return img_rgb[:, offset : offset + h]


def create_detector(face_model: Path):
    if MEDIAPIPE_IMPORT_ERROR is not None:
        _warn(f"MediaPipe import failed; using center crop: {MEDIAPIPE_IMPORT_ERROR}")
        return None
    try:
        options = vision.FaceDetectorOptions(
            base_options=BaseOptions(model_asset_path=str(face_model)),
            min_detection_confidence=0.5,
        )
        return vision.FaceDetector.create_from_options(options)
    except Exception as exc:
        _warn(f"Face detector creation failed; using center crop: {exc}")
        return None


def detect_and_crop_face(img_rgb: np.ndarray, detector) -> np.ndarray:
    if detector is None or mp is None:
        return center_crop(img_rgb)
    try:
        h, w, _ = img_rgb.shape
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=img_rgb)
        result = detector.detect(mp_image)
        if result.detections:
            bbox = result.detections[0].bounding_box
            pad_x = int(bbox.width * 0.15)
            pad_y = int(bbox.height * 0.15)
            x_min = max(0, bbox.origin_x - pad_x)
            y_min = max(0, bbox.origin_y - pad_y)
            x_max = min(w, bbox.origin_x + bbox.width + pad_x)
            y_max = min(h, bbox.origin_y + bbox.height + pad_y)
            if x_max > x_min and y_max > y_min:
                return img_rgb[y_min:y_max, x_min:x_max]
    except Exception as exc:
        _warn(f"face detection failed on one image; using center crop: {exc}")
    return center_crop(img_rgb)


def iter_original_images(data_dir: Path, class_name: str):
    split_dir = data_dir / class_name / "test"
    if not split_dir.is_dir():
        _warn(f"missing test directory: {split_dir}")
        return
    for path in sorted(p for p in split_dir.rglob("*") if p.is_file()):
        if is_original_filename(path.name):
            yield path


def capture_id_for(class_name: str, split_dir: Path, image_path: Path) -> str:
    rel = image_path.relative_to(split_dir).with_suffix("")
    stem = rel.as_posix().replace("/", "_")
    return f"{class_name}_{stem}"


def build_arrays(data_dir: Path, face_model: Path):
    detector = create_detector(face_model)
    x96: list[np.ndarray] = []
    x160: list[np.ndarray] = []
    labels: list[int] = []
    capture_ids: list[str] = []
    class_counts: dict[str, int] = {}

    try:
        for class_name in CLASSES:
            split_dir = data_dir / class_name / "test"
            count = 0
            for image_path in iter_original_images(data_dir, class_name):
                img_bgr = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
                if img_bgr is None:
                    _warn(f"could not read image: {image_path}")
                    continue
                img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
                face_crop = detect_and_crop_face(img_rgb, detector)

                resized96 = cv2.resize(face_crop, (96, 96), interpolation=cv2.INTER_AREA)
                resized160 = cv2.resize(face_crop, (160, 160), interpolation=cv2.INTER_AREA)

                x96.append(resized96.astype(np.float32) / 127.5 - 1.0)
                x160.append(resized160.astype(np.float32))
                labels.append(LABELS[class_name])
                capture_ids.append(capture_id_for(class_name, split_dir, image_path))
                count += 1
            class_counts[class_name] = count
    finally:
        if detector is not None:
            detector.close()

    return (
        np.asarray(x96, dtype=np.float32),
        np.asarray(x160, dtype=np.float32),
        np.asarray(labels, dtype=np.int64),
        np.asarray(capture_ids, dtype=object),
        class_counts,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--face-model", default="python/blaze_face_short_range.tflite")
    parser.add_argument("--out-dir", default="bench/results")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    data_dir = Path(args.data_dir)
    out_dir = Path(args.out_dir)
    face_model = Path(args.face_model)
    out_dir.mkdir(parents=True, exist_ok=True)

    x96, x160, y, capture_ids, class_counts = build_arrays(data_dir, face_model)
    if len(y) == 0:
        raise RuntimeError("no original test images were found")

    np.save(out_dir / "x_test_originals_96_pm1.npy", x96)
    np.save(out_dir / "x_test_originals_160_raw.npy", x160)
    np.save(out_dir / "y_test_originals.npy", y)
    np.save(out_dir / "capture_ids_originals.npy", capture_ids, allow_pickle=True)

    counts = ", ".join(f"{name}={class_counts.get(name, 0)}" for name in CLASSES)
    print(
        "saved originals-only arrays: "
        f"x96={x96.shape} x160={x160.shape} y={y.shape} ({counts})"
    )


if __name__ == "__main__":
    main()
