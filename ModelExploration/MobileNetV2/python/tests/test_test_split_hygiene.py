"""Regression tests for test-split hygiene (finding F1).

These tests fail loudly if augmentation contamination ever returns to
``data/<person>/test/`` or to the cached ``x_test.npy``. They are
designed to be runnable two ways:

    # From the repo root:
    python -m unittest ModelExploration.MobileNetV2.python.tests.test_test_split_hygiene

    # From ModelExploration/MobileNetV2/python/:
    python -m unittest tests.test_test_split_hygiene

The data path is auto-detected: we look for ``data/`` in successive
parent directories starting from this file. Tests that depend on the
on-disk dataset are skipped if the data is not present (so CI without
the dataset still passes).
"""

from __future__ import annotations

import os
import sys
import unittest
from pathlib import Path

# Make sibling packages importable when running this file directly.
_PY_DIR = Path(__file__).resolve().parent.parent
if str(_PY_DIR) not in sys.path:
    sys.path.insert(0, str(_PY_DIR))

from utils.train_val_split import AUG_SUFFIXES, IMAGE_EXTENSIONS  # noqa: E402


def _find_data_dir() -> Path | None:
    """Search upward from this file for a directory that contains data/<class>/test/."""
    here = Path(__file__).resolve()
    for parent in [here.parent, *here.parents]:
        candidate = parent / "data"
        if candidate.is_dir() and any(
            (candidate / sub / "test").is_dir() for sub in os.listdir(candidate)
            if (candidate / sub).is_dir()
        ):
            return candidate
    return None


def _is_augmented(filename: str) -> bool:
    stem = os.path.splitext(filename)[0].lower()
    return any(stem.endswith(suffix) for suffix in AUG_SUFFIXES)


class TestSplitHygiene(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.data_dir = _find_data_dir()

    def setUp(self) -> None:
        if self.data_dir is None:
            self.skipTest("data/ directory not found in any parent of this file")

    def test_no_augmented_files_in_test_split(self) -> None:
        """Finding F1 fix: no file under data/<person>/test/ may be augmented."""
        offenders: list[str] = []
        for person_dir in sorted(p for p in self.data_dir.iterdir() if p.is_dir()):
            if person_dir.name.startswith("_"):
                continue  # skip _quarantine and any meta folders
            test_dir = person_dir / "test"
            if not test_dir.is_dir():
                continue
            for path in test_dir.iterdir():
                if not path.is_file():
                    continue
                if path.suffix.lower() not in IMAGE_EXTENSIONS:
                    continue
                if _is_augmented(path.name):
                    offenders.append(str(path))
        self.assertFalse(
            offenders,
            msg=(
                f"{len(offenders)} augmented file(s) found in data/<person>/test/ — "
                "this violates the train-only augmentation policy. "
                "Run `python tools/clean_test_augmentations.py --quarantine` to fix.\n"
                + "\n".join(f"  {o}" for o in offenders[:10])
            ),
        )

    def test_test_split_balanced_per_class(self) -> None:
        """Each class's test split must contain the same number of original images."""
        per_class: dict[str, int] = {}
        for person_dir in sorted(p for p in self.data_dir.iterdir() if p.is_dir()):
            if person_dir.name.startswith("_"):
                continue
            test_dir = person_dir / "test"
            if not test_dir.is_dir():
                continue
            count = sum(
                1
                for path in test_dir.iterdir()
                if path.is_file()
                and path.suffix.lower() in IMAGE_EXTENSIONS
                and not _is_augmented(path.name)
            )
            per_class[person_dir.name] = count
        if not per_class:
            self.skipTest("no test splits found under data/")
        unique_counts = set(per_class.values())
        self.assertEqual(
            len(unique_counts),
            1,
            msg=(
                f"per-class test counts are not balanced: {per_class}. "
                "Either originals are missing or augmented files leaked back in."
            ),
        )

    def test_cached_x_test_shape_matches_originals(self) -> None:
        """If python/gen/x_test.npy exists it must be the originals-only set."""
        try:
            import numpy as np  # noqa: WPS433
        except ImportError:
            self.skipTest("numpy not installed; skipping cache shape test")

        gen_dir = _PY_DIR / "gen"
        x_path = gen_dir / "x_test.npy"
        y_path = gen_dir / "y_test.npy"
        if not (x_path.is_file() and y_path.is_file()):
            self.skipTest(f"{x_path} / {y_path} not present; run `make preprocess`")

        x = np.load(x_path)
        y = np.load(y_path)

        per_class_originals = []
        for person_dir in sorted(p for p in self.data_dir.iterdir() if p.is_dir()):
            if person_dir.name.startswith("_"):
                continue
            test_dir = person_dir / "test"
            if not test_dir.is_dir():
                continue
            count = sum(
                1
                for path in test_dir.iterdir()
                if path.is_file()
                and path.suffix.lower() in IMAGE_EXTENSIONS
                and not _is_augmented(path.name)
            )
            per_class_originals.append(count)
        if not per_class_originals:
            self.skipTest("no originals found under data/<person>/test/")
        expected = sum(per_class_originals)

        self.assertEqual(
            x.shape[0],
            expected,
            msg=(
                f"x_test.npy has {x.shape[0]} rows but data/<person>/test/ "
                f"contains {expected} originals. Re-run `make preprocess` "
                f"after a fresh cleanup."
            ),
        )
        self.assertEqual(y.shape[0], expected)
        self.assertEqual(x.ndim, 4)
        self.assertEqual(x.shape[1:], (96, 96, 3))


    def test_quarantine_inventory_matches_manifest(self) -> None:
        """Quarantine must be self-consistent: file count == manifest count == 720."""
        import json

        quar = self.data_dir / "_quarantine" / "test_augmented"
        manifest = quar / "manifest.json"
        if not manifest.is_file():
            self.skipTest(f"{manifest} not present; run `make clean-test-aug`")

        with manifest.open() as fh:
            data = json.load(fh)

        # manifest['count'] is the number of files acted on in the most recent
        # run. The on-disk PNGs under _quarantine/test_augmented/<person>/
        # should equal that count.
        on_disk = sum(
            1
            for path in quar.rglob("*")
            if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
        )
        self.assertEqual(
            on_disk,
            int(data.get("count", 0)),
            msg=(
                f"_quarantine has {on_disk} image files but manifest says "
                f"{data.get('count')}; the cleanup audit trail is out of sync."
            ),
        )

    def test_full_aug_npz_reproduces_when_present(self) -> None:
        """If bench/results/jakubs_qat_full_aug_test.npz exists it must be the n=780 panel."""
        try:
            import numpy as np  # noqa: WPS433
        except ImportError:
            self.skipTest("numpy not installed")

        npz_path = (
            _PY_DIR.parent / "bench" / "results" / "jakubs_qat_full_aug_test.npz"
        )
        if not npz_path.is_file():
            self.skipTest(
                f"{npz_path} not present; run `make full-aug` to regenerate."
            )

        with np.load(npz_path, allow_pickle=True) as data:
            n = int(data["labels"].shape[0])
            preds = np.asarray(data["predictions"])
            labels = np.asarray(data["labels"])

        # 60 originals + 720 augmented = 780, balanced 260 / class.
        self.assertEqual(
            n,
            780,
            msg=(
                f"jakubs_qat_full_aug_test.npz has n={n} but the augmentation "
                "panel must be 780 (60 originals + 720 quarantined augmented). "
                "Re-run `make full-aug`."
            ),
        )
        unique, counts = np.unique(labels, return_counts=True)
        self.assertEqual(
            sorted(counts.tolist()),
            [260, 260, 260],
            msg=f"per-class file counts not balanced: {dict(zip(unique.tolist(), counts.tolist(), strict=True))}",
        )
        # The historical reference accuracy is 762/780 = 97.6923...%. Any
        # change in this number indicates either the model or the cropping
        # pipeline has drifted; both are headline-shifting events that need a
        # human review.
        n_correct = int((preds == labels).sum())
        self.assertEqual(
            n_correct,
            762,
            msg=(
                f"augmentation panel accuracy drift: {n_correct}/780 correct, "
                "expected 762/780 (97.69 %). Either model.tflite changed, the "
                "preprocess pipeline drifted, or the quarantine was modified."
            ),
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)
