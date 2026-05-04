"""
Deploy step: copy the freshly generated INT8 model into the firmware tree.

After ``main.py`` or ``qat_export.py`` writes ``python/gen/model.tflite``,
``python/gen/model.h`` and ``python/gen/model.c``, this script syncs the C
sources into ``esp32/main/`` so the next ``idf.py build`` picks them up.

Run from the repo root:

    python python/deploy.py [--model-name model] [--dst esp32/main]

Use ``--model-name baseline_model`` to deploy the F2-clean baseline retrain
instead of the production model.
"""
from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
GEN_DIR = REPO_ROOT / "python" / "gen"
DEFAULT_DST = REPO_ROOT / "esp32" / "main"


def deploy(model_name: str, dst: Path) -> int:
    src_h = GEN_DIR / f"{model_name}.h"
    src_c = GEN_DIR / f"{model_name}.c"
    src_tflite = GEN_DIR / f"{model_name}.tflite"

    missing = [p for p in (src_h, src_c) if not p.exists()]
    if missing:
        print("ERROR: missing generated artefact(s):", file=sys.stderr)
        for p in missing:
            print(f"  - {p}", file=sys.stderr)
        print("Run `make train` (or `make qat`) before `make deploy`.",
              file=sys.stderr)
        return 1

    dst.mkdir(parents=True, exist_ok=True)
    dst_h = dst / "model.h"
    dst_c = dst / "model.c"

    shutil.copy2(src_h, dst_h)
    shutil.copy2(src_c, dst_c)

    print(f"Deployed {model_name}.tflite "
          f"({src_tflite.stat().st_size / 1024:.1f} KB)")
    print(f"  {src_h.relative_to(REPO_ROOT)} -> {dst_h.relative_to(REPO_ROOT)}")
    print(f"  {src_c.relative_to(REPO_ROOT)} -> {dst_c.relative_to(REPO_ROOT)}")
    print("Next: cd esp32 && idf.py build flash")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model-name",
        default="model",
        help="Stem of the .h/.c/.tflite triple in python/gen/ (default: model)",
    )
    parser.add_argument(
        "--dst",
        type=Path,
        default=DEFAULT_DST,
        help=f"Destination dir (default: {DEFAULT_DST.relative_to(REPO_ROOT)})",
    )
    args = parser.parse_args()
    return deploy(args.model_name, args.dst)


if __name__ == "__main__":
    raise SystemExit(main())
