"""Train/validation splitting helpers that keep augmented captures grouped."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Mapping

import numpy as np

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp"}
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


def capture_prefix(filename: str) -> str:
    """Return the original-capture prefix for an image filename."""
    stem = Path(filename).stem
    stem_lower = stem.lower()
    for suffix in AUG_SUFFIXES:
        if stem_lower.endswith(suffix):
            return stem[: -len(suffix)]
    return stem


def train_image_records(
    data_dir: str | Path,
    labels: Mapping[str, int],
    split: str = "train",
) -> list[dict[str, object]]:
    """Return image records in the same deterministic order as preprocess.py."""
    data_path = Path(data_dir).resolve()
    records: list[dict[str, object]] = []

    for person_dir in sorted(p for p in data_path.iterdir() if p.is_dir()):
        person = person_dir.name
        if person not in labels:
            continue
        split_dir = person_dir / split
        if not split_dir.is_dir():
            continue

        for image_path in sorted(p for p in split_dir.iterdir() if p.is_file()):
            if image_path.suffix.lower() not in IMAGE_EXTENSIONS:
                continue
            capture_id = f"{person}/{capture_prefix(image_path.name)}"
            try:
                relative_path = image_path.relative_to(data_path.parent).as_posix()
            except ValueError:
                relative_path = image_path.as_posix()
            records.append(
                {
                    "path": relative_path,
                    "class_name": person,
                    "label": int(labels[person]),
                    "capture_id": capture_id,
                }
            )
    return records


def select_grouped_validation_indices(
    records: list[dict[str, object]],
    val_fraction: float,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, dict[str, object]]:
    """Split records by class-stratified capture groups."""
    if not 0.0 < val_fraction < 1.0:
        raise ValueError(f"val_fraction must be in (0, 1), got {val_fraction}")
    if not records:
        raise ValueError("no training records found for validation split")

    rng = np.random.default_rng(seed)
    labels = sorted({int(record["label"]) for record in records})
    val_capture_ids: set[str] = set()
    class_metadata: dict[str, object] = {}

    for label in labels:
        class_records = [record for record in records if int(record["label"]) == label]
        class_name = str(class_records[0]["class_name"])
        captures = sorted({str(record["capture_id"]) for record in class_records})
        if len(captures) < 2:
            raise ValueError(
                f"class {class_name} needs at least two captures for a train/val split"
            )

        n_val_captures = int(round(len(captures) * val_fraction))
        n_val_captures = max(1, min(n_val_captures, len(captures) - 1))
        shuffled = np.array(captures, dtype=object)
        rng.shuffle(shuffled)
        selected = sorted(str(capture_id) for capture_id in shuffled[:n_val_captures])
        val_capture_ids.update(selected)

        val_files = [
            str(record["path"])
            for record in class_records
            if str(record["capture_id"]) in selected
        ]
        class_metadata[class_name] = {
            "label": label,
            "total_captures": len(captures),
            "val_captures": selected,
            "train_capture_count": len(captures) - len(selected),
            "val_file_count": len(val_files),
            "train_file_count": len(class_records) - len(val_files),
        }

    val_indices = np.array(
        [idx for idx, record in enumerate(records) if str(record["capture_id"]) in val_capture_ids],
        dtype=np.int64,
    )
    train_indices = np.array(
        [idx for idx, record in enumerate(records) if str(record["capture_id"]) not in val_capture_ids],
        dtype=np.int64,
    )

    metadata: dict[str, object] = {
        "seed": seed,
        "val_fraction": val_fraction,
        "source_split": "train",
        "total_file_count": len(records),
        "train_file_count": int(len(train_indices)),
        "val_file_count": int(len(val_indices)),
        "classes": class_metadata,
        "val_files": [str(records[idx]["path"]) for idx in val_indices],
        "train_files": [str(records[idx]["path"]) for idx in train_indices],
    }
    return train_indices, val_indices, metadata


def write_split_metadata(path: str | Path, metadata: Mapping[str, object]) -> None:
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n")


def split_train_for_validation(
    x_train: np.ndarray,
    y_train: np.ndarray,
    data_dir: str | Path,
    gen_dir: str | Path,
    labels: Mapping[str, int],
    val_fraction: float = 0.15,
    seed: int = 42,
    split_filename: str = "val_split_seed42.json",
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict[str, object]]:
    """Create a reproducible train/val split from training data only."""
    records = train_image_records(data_dir, labels, split="train")
    if len(records) != len(x_train):
        raise ValueError(
            f"training image count mismatch: {len(records)} files in data/train vs "
            f"{len(x_train)} cached samples"
        )

    y_train = np.asarray(y_train).reshape(-1)
    expected_labels = np.array([int(record["label"]) for record in records], dtype=y_train.dtype)
    mismatches = np.nonzero(expected_labels != y_train)[0]
    if len(mismatches):
        first = int(mismatches[0])
        raise ValueError(
            "cached y_train.npy order does not match data/train file order; "
            f"first mismatch at index {first}: expected {expected_labels[first]}, got {y_train[first]}"
        )

    fit_indices, val_indices, metadata = select_grouped_validation_indices(
        records=records,
        val_fraction=val_fraction,
        seed=seed,
    )
    split_path = Path(gen_dir) / split_filename
    write_split_metadata(split_path, metadata)
    metadata["split_path"] = str(split_path)

    return (
        x_train[fit_indices],
        y_train[fit_indices],
        x_train[val_indices],
        y_train[val_indices],
        metadata,
    )
