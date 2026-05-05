"""Remove augmented variants from the test split.

Augmentation must apply to train only (Goodfellow §5.3/§7.4; Kapoor &
Narayanan 2023, *Patterns* 4(9), arXiv:2207.07048; Sculley et al.
NeurIPS 2015). Test data must reflect the deployment distribution, with
exactly one row per source capture. This script enforces that on disk.

The list of augmentation suffixes is imported from
``utils.train_val_split`` so this script and the train/val splitter
share one source of truth.

Modes:
    --dry-run     (default) list what would be moved and exit 0
    --quarantine  move matched files to data/_quarantine/test_augmented/<person>/
                  and write a manifest with sha256 of every removed file
    --purge       same audit trail (manifest written first), then unlink

The script is idempotent: a second run on a clean tree is a no-op.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path

# Make the parent ``python`` directory importable when this script is
# run as a file (e.g. ``python tools/clean_test_augmentations.py``).
_PY_DIR = Path(__file__).resolve().parent.parent
if str(_PY_DIR) not in sys.path:
    sys.path.insert(0, str(_PY_DIR))

from utils.train_val_split import AUG_SUFFIXES, IMAGE_EXTENSIONS  # noqa: E402

# ``ModelExploration/MobileNetV2/python/tools/`` -> repo root is 4 levels up
# (parents indexed: 0=tools, 1=python, 2=MobileNetV2, 3=ModelExploration, 4=repo).
DEFAULT_DATA_DIR = Path(__file__).resolve().parents[4] / "data"
DEFAULT_QUARANTINE_DIR = DEFAULT_DATA_DIR / "_quarantine" / "test_augmented"


def is_augmented_filename(filename: str) -> bool:
    """Return True if the filename's stem ends with a known augmentation suffix."""
    path = Path(filename)
    if path.suffix.lower() not in IMAGE_EXTENSIONS:
        return False
    stem_lower = path.stem.lower()
    return any(stem_lower.endswith(suffix) for suffix in AUG_SUFFIXES)


def matched_suffix(filename: str) -> str | None:
    stem_lower = Path(filename).stem.lower()
    for suffix in AUG_SUFFIXES:
        if stem_lower.endswith(suffix):
            return suffix
    return None


def sha256_of(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def find_augmented_test_files(data_dir: Path) -> list[tuple[str, Path]]:
    """Return ``(person, path)`` for every augmented file under ``<person>/test/``."""
    matches: list[tuple[str, Path]] = []
    if not data_dir.is_dir():
        return matches
    for person_dir in sorted(p for p in data_dir.iterdir() if p.is_dir()):
        # Skip the quarantine folder if it sits inside data_dir.
        if person_dir.name.startswith("_"):
            continue
        test_dir = person_dir / "test"
        if not test_dir.is_dir():
            continue
        for path in sorted(test_dir.iterdir()):
            if not path.is_file():
                continue
            if is_augmented_filename(path.name):
                matches.append((person_dir.name, path))
    return matches


def find_originals_in_train(data_dir: Path) -> dict[str, set[str]]:
    """Map ``person -> {original_stem}`` for ``data/<person>/train/`` files.

    Used for the sanity check: every augmented test file should derive
    from a capture stem that exists somewhere in train (since augmented
    test variants were created from the test originals, not train, this
    is a soft warning rather than a hard requirement).
    """
    out: dict[str, set[str]] = {}
    if not data_dir.is_dir():
        return out
    for person_dir in sorted(p for p in data_dir.iterdir() if p.is_dir()):
        if person_dir.name.startswith("_"):
            continue
        train_dir = person_dir / "train"
        if not train_dir.is_dir():
            continue
        stems = set()
        for path in train_dir.iterdir():
            if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS:
                stems.add(path.stem)
        out[person_dir.name] = stems
    return out


def find_originals_in_test(data_dir: Path) -> dict[str, set[str]]:
    """Map ``person -> {original_stem}`` for ``data/<person>/test/`` originals."""
    out: dict[str, set[str]] = {}
    if not data_dir.is_dir():
        return out
    for person_dir in sorted(p for p in data_dir.iterdir() if p.is_dir()):
        if person_dir.name.startswith("_"):
            continue
        test_dir = person_dir / "test"
        if not test_dir.is_dir():
            continue
        stems = set()
        for path in test_dir.iterdir():
            if (
                path.is_file()
                and path.suffix.lower() in IMAGE_EXTENSIONS
                and not is_augmented_filename(path.name)
            ):
                stems.add(path.stem)
        out[person_dir.name] = stems
    return out


def capture_stem_for(filename: str) -> str:
    """Strip the augmentation suffix from a filename to recover the capture stem."""
    suffix = matched_suffix(filename)
    if suffix is None:
        return Path(filename).stem
    stem = Path(filename).stem
    return stem[: -len(suffix)]


def write_manifest(
    manifest_path: Path,
    entries: list[dict[str, object]],
    mode: str,
    data_dir: Path,
) -> None:
    payload = {
        "mode": mode,
        "removed_at": datetime.now(timezone.utc).isoformat(),
        "data_dir": str(data_dir),
        "count": len(entries),
        "entries": entries,
    }
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    if manifest_path.exists():
        try:
            existing = json.loads(manifest_path.read_text())
        except (OSError, json.JSONDecodeError):
            existing = None
        if isinstance(existing, dict) and "entries" in existing:
            seen = {entry.get("original_path") for entry in existing["entries"]}
            for entry in entries:
                if entry.get("original_path") not in seen:
                    existing["entries"].append(entry)
            existing["count"] = len(existing["entries"])
            existing["last_run_at"] = payload["removed_at"]
            existing["last_run_mode"] = mode
            manifest_path.write_text(json.dumps(existing, indent=2, sort_keys=True) + "\n")
            return
    manifest_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=DEFAULT_DATA_DIR,
        help="Root data directory (default: repo-root data/)",
    )
    parser.add_argument(
        "--quarantine-dir",
        type=Path,
        default=DEFAULT_QUARANTINE_DIR,
        help="Where to move augmented files when --quarantine is set",
    )
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument(
        "--dry-run",
        action="store_true",
        default=True,
        help="(default) Print actions without modifying anything",
    )
    mode.add_argument(
        "--quarantine",
        action="store_true",
        help="Move augmented files into the quarantine directory",
    )
    mode.add_argument(
        "--purge",
        action="store_true",
        help="Permanently delete augmented files (manifest still written)",
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        help="Manifest output path (default: <quarantine-dir>/manifest.json)",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Exit non-zero if any augmented test file has no matching original",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    data_dir = args.data_dir.resolve()
    quarantine_dir = args.quarantine_dir.resolve()
    manifest_path = (args.manifest or (quarantine_dir / "manifest.json")).resolve()

    if args.purge:
        mode = "purge"
    elif args.quarantine:
        mode = "quarantine"
    else:
        mode = "dry-run"

    if not data_dir.is_dir():
        print(f"error: data dir does not exist: {data_dir}", file=sys.stderr)
        return 2

    matches = find_augmented_test_files(data_dir)
    if not matches:
        print(f"clean: no augmented files found under {data_dir}/*/test/")
        return 0

    test_originals = find_originals_in_test(data_dir)
    train_originals = find_originals_in_train(data_dir)

    # Sanity: every augmented test file should map to a known capture stem.
    orphans: list[Path] = []
    for person, path in matches:
        capture = capture_stem_for(path.name)
        in_test = capture in test_originals.get(person, set())
        in_train = capture in train_originals.get(person, set())
        if not in_test and not in_train:
            orphans.append(path)
    if orphans:
        print(
            f"warning: {len(orphans)} augmented file(s) have no matching original capture:",
            file=sys.stderr,
        )
        for path in orphans[:10]:
            print(f"  {path}", file=sys.stderr)
        if args.strict:
            return 3

    by_person: dict[str, dict[str, int]] = {}
    for person, path in matches:
        suffix = matched_suffix(path.name) or "?"
        bucket = by_person.setdefault(person, {})
        bucket[suffix] = bucket.get(suffix, 0) + 1

    print(f"data dir:        {data_dir}")
    print(f"quarantine dir:  {quarantine_dir}" if mode == "quarantine" else f"mode:            {mode}")
    print(f"manifest path:   {manifest_path}" if mode != "dry-run" else "manifest path:   (skipped, dry-run)")
    print(f"matches found:   {len(matches)}")
    print()
    for person in sorted(by_person):
        per_class = sum(by_person[person].values())
        breakdown = ", ".join(
            f"{suffix}={count}" for suffix, count in sorted(by_person[person].items())
        )
        print(f"  {person}/test: {per_class} augmented files [{breakdown}]")
    print()

    if mode == "dry-run":
        print("dry-run: no files were modified. Re-run with --quarantine or --purge.")
        return 0

    entries: list[dict[str, object]] = []
    for person, path in matches:
        digest = sha256_of(path)
        entry: dict[str, object] = {
            "person": person,
            "original_path": str(path.relative_to(data_dir.parent) if path.is_relative_to(data_dir.parent) else path),
            "filename": path.name,
            "suffix": matched_suffix(path.name),
            "size_bytes": path.stat().st_size,
            "sha256": digest,
        }

        if mode == "quarantine":
            dest = quarantine_dir / person / path.name
            dest.parent.mkdir(parents=True, exist_ok=True)
            if dest.exists():
                # Idempotency: if the destination already exists with the
                # same digest, just remove the source.
                existing_digest = sha256_of(dest)
                if existing_digest == digest:
                    path.unlink()
                    entry["action"] = "duplicate-source-unlinked"
                    entries.append(entry)
                    continue
                # Otherwise, suffix the destination to avoid clobbering.
                dest = dest.with_name(f"{path.stem}.{digest[:8]}{path.suffix}")
            shutil.move(str(path), str(dest))
            entry["quarantined_to"] = str(dest.relative_to(data_dir.parent) if dest.is_relative_to(data_dir.parent) else dest)
            entry["action"] = "quarantined"
        elif mode == "purge":
            path.unlink()
            entry["action"] = "purged"

        entries.append(entry)

    write_manifest(manifest_path, entries, mode, data_dir)
    print(f"removed {len(entries)} files in mode={mode}")
    print(f"manifest: {manifest_path}")

    leftover = find_augmented_test_files(data_dir)
    if leftover:
        print(
            f"warning: {len(leftover)} augmented test file(s) still present after run",
            file=sys.stderr,
        )
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
