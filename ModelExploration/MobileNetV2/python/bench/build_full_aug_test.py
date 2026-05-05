"""Regenerate the augmentation-robustness panel npz from the cleaned dataset.

After the F1 cleanup, ``data/<person>/test/`` contains only the 60 original
captures; the 720 augmented variants live under
``data/_quarantine/test_augmented/<person>/`` with a SHA-256 manifest
(``data/_quarantine/test_augmented/manifest.json``). This script reads the
60 originals plus the 720 quarantined augmented files, runs the same
face-detect / crop / resize / normalise pipeline that ``preprocess.py`` and
``build_originals_test.py`` use, evaluates ``python/gen/model.tflite``, and
writes ``bench/results/jakubs_qat_full_aug_test.npz``.

The augmentation-robustness panel produced from this artefact is reported
only as a *biased* diagnostic (``stats_summary.md`` Section
"Augmentation-robustness panel"); the headline numbers come from the
originals-only test set built by ``build_originals_test.py``.

Reproducibility: every input image is on disk (60 in ``data/<person>/test/``
plus 720 in ``data/_quarantine/test_augmented/<person>/``) and every input
filename is recorded in ``capture_ids`` so the panel can be audited file by
file. The script is deterministic — no random transforms are applied at
runtime; we re-evaluate the *same* augmented PNGs that the original pipeline
produced.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
import tensorflow as tf

# Reuse the face-detect / crop helpers from build_originals_test so this
# script and the originals-only build use byte-identical preprocessing.
sys.path.insert(0, str(Path(__file__).resolve().parent))
from build_originals_test import (  # noqa: E402
    AUG_SUFFIXES,
    CLASSES,
    IMAGE_RE,
    LABELS,
    capture_id_for,
    create_detector,
    detect_and_crop_face,
)


def _strip_aug_suffix(stem: str) -> tuple[str, str | None]:
    """Return (base_stem, suffix_or_None) for an augmented filename stem."""
    lowered = stem.lower()
    for suffix in AUG_SUFFIXES:
        if lowered.endswith(suffix):
            return stem[: -len(suffix)], suffix
    return stem, None


def _iter_test_images(data_dir: Path, quarantine_dir: Path, class_name: str):
    """Yield (image_path, capture_id, is_augmented) for class_name.

    capture_id groups originals with their augmented variants by the
    *original* base capture, e.g. ``Amine_image_20260319_132049`` covers
    both the original PNG in ``data/Amine/test/`` and the 12 augmented
    siblings in ``data/_quarantine/test_augmented/Amine/``.
    """
    test_dir = data_dir / class_name / "test"
    if test_dir.is_dir():
        for path in sorted(p for p in test_dir.rglob("*") if p.is_file()):
            if not IMAGE_RE.match(path.name):
                continue
            yield path, capture_id_for(class_name, test_dir, path), False

    quar_dir = quarantine_dir / class_name
    if quar_dir.is_dir():
        for path in sorted(p for p in quar_dir.rglob("*") if p.is_file()):
            if not IMAGE_RE.match(path.name):
                continue
            base_stem, _ = _strip_aug_suffix(path.stem)
            yield path, f"{class_name}_{base_stem}", True


def _build_inputs(data_dir: Path, quarantine_dir: Path, face_model: Path):
    detector = create_detector(face_model)
    x96: list[np.ndarray] = []
    labels: list[int] = []
    capture_ids: list[str] = []
    file_paths: list[str] = []
    is_aug_flags: list[bool] = []
    counts: dict[str, dict[str, int]] = {c: {"orig": 0, "aug": 0} for c in CLASSES}

    try:
        for class_name in CLASSES:
            for image_path, capture_id, is_aug in _iter_test_images(
                data_dir, quarantine_dir, class_name
            ):
                img_bgr = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
                if img_bgr is None:
                    print(f"warning: could not read {image_path}", file=sys.stderr)
                    continue
                img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
                face_crop = detect_and_crop_face(img_rgb, detector)
                resized96 = cv2.resize(face_crop, (96, 96), interpolation=cv2.INTER_AREA)
                x96.append(resized96.astype(np.float32) / 127.5 - 1.0)
                labels.append(LABELS[class_name])
                capture_ids.append(capture_id)
                file_paths.append(str(image_path.relative_to(data_dir.parent)))
                is_aug_flags.append(is_aug)
                counts[class_name]["aug" if is_aug else "orig"] += 1
    finally:
        if detector is not None:
            detector.close()

    return (
        np.asarray(x96, dtype=np.float32),
        np.asarray(labels, dtype=np.int64),
        np.asarray(capture_ids, dtype=object),
        np.asarray(file_paths, dtype=object),
        np.asarray(is_aug_flags, dtype=bool),
        counts,
    )


def _run_tflite(model_path: Path, x: np.ndarray):
    interpreter = tf.lite.Interpreter(model_path=str(model_path))
    interpreter.allocate_tensors()
    in_det = interpreter.get_input_details()[0]
    out_det = interpreter.get_output_details()[0]
    in_scale, in_zp = in_det["quantization"]
    out_scale, out_zp = out_det["quantization"]

    n = x.shape[0]
    n_classes = out_det["shape"][-1]
    probs = np.zeros((n, n_classes), dtype=np.float32)
    preds = np.zeros((n,), dtype=np.int64)

    for i in range(n):
        sample = x[i : i + 1]
        if in_det["dtype"] == np.int8:
            q = np.round(sample / in_scale + in_zp).astype(np.int8)
            interpreter.set_tensor(in_det["index"], q)
        else:
            interpreter.set_tensor(in_det["index"], sample.astype(in_det["dtype"]))
        interpreter.invoke()
        out = interpreter.get_tensor(out_det["index"])[0]
        if out_det["dtype"] == np.int8:
            out = (out.astype(np.float32) - out_zp) * out_scale
        probs[i] = out
        preds[i] = int(np.argmax(out))

    return preds, probs, {
        "input_scale": float(in_scale),
        "input_zero_point": int(in_zp),
        "output_scale": float(out_scale),
        "output_zero_point": int(out_zp),
        "input_shape": np.asarray(in_det["shape"], dtype=np.int64),
        "input_dtype": np.dtype(in_det["dtype"]).name,
        "output_dtype": np.dtype(out_det["dtype"]).name,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", default="data")
    parser.add_argument(
        "--quarantine-dir",
        default="data/_quarantine/test_augmented",
        help="Folder containing the cleaned augmented test variants.",
    )
    parser.add_argument("--face-model", default="python/blaze_face_short_range.tflite")
    parser.add_argument("--model", default="python/gen/model.tflite")
    parser.add_argument(
        "--out",
        default="bench/results/jakubs_qat_full_aug_test.npz",
        help=(
            "Output npz path. Overwrites the legacy artefact consumed by "
            "run_stats.py to keep the augmentation-robustness panel reproducible."
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    data_dir = Path(args.data_dir).resolve()
    quarantine_dir = Path(args.quarantine_dir).resolve()
    face_model = Path(args.face_model).resolve()
    model_path = Path(args.model).resolve()
    out_path = Path(args.out).resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if not quarantine_dir.is_dir():
        raise SystemExit(
            f"quarantine directory {quarantine_dir} not found — run "
            "`python tools/clean_test_augmentations.py --quarantine` first."
        )

    x96, labels, capture_ids, file_paths, is_aug, counts = _build_inputs(
        data_dir, quarantine_dir, face_model
    )
    n = len(labels)
    if n == 0:
        raise SystemExit("no test images found")

    preds, probs, qmeta = _run_tflite(model_path, x96)
    acc = float((preds == labels).mean())

    np.savez(
        out_path,
        predictions=preds,
        probs=probs,
        labels=labels,
        capture_ids=capture_ids,
        file_paths=file_paths,
        is_augmented=is_aug,
        model_path=str(model_path.relative_to(data_dir.parent))
        if model_path.is_relative_to(data_dir.parent)
        else str(model_path),
        x_path=str(out_path.relative_to(data_dir.parent))
        if out_path.is_relative_to(data_dir.parent)
        else str(out_path),
        y_path=str(out_path.relative_to(data_dir.parent))
        if out_path.is_relative_to(data_dir.parent)
        else str(out_path),
        norm="pm1",
        **qmeta,
    )

    summary = ", ".join(
        f"{c}=orig{counts[c]['orig']}+aug{counts[c]['aug']}" for c in CLASSES
    )
    print(f"saved {out_path} (n={n}; {summary}); accuracy={acc:.4f}")


if __name__ == "__main__":
    main()
