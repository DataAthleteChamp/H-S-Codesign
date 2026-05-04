"""Parse on-device latency lines from the ESP32 serial stream.

Connects to the firmware over USB-CDC, listens for log lines that match
`latency_ms=<float>` (optionally with `preprocess_ms=<float>` and
`inference_ms=<float>` companions), and writes per-sample CSV plus a
summary markdown with p50/p90/p95/p99 statistics and a histogram PNG.

Usage:
    python python/tools/serial_latency_logger.py \
        --port /dev/cu.usbmodem* --num-samples 200

If the firmware emits a different log line format, override the regex
with --regex. Pass --replay <path> to re-process an existing log file
without opening the serial port (useful for `idf.py monitor > log.txt`
captures).

Outputs:
    bench/results/onboard_latency.csv
    bench/results/onboard_latency.md
    bench/results/onboard_latency.png   (histogram)
"""

from __future__ import annotations

import argparse
import csv
import re
import sys
import time
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CSV = REPO_ROOT / "bench" / "results" / "onboard_latency.csv"
DEFAULT_MD = REPO_ROOT / "bench" / "results" / "onboard_latency.md"
DEFAULT_PNG = REPO_ROOT / "bench" / "results" / "onboard_latency.png"
DEFAULT_REGEX = (
    r"latency_ms=(?P<latency>\d+(?:\.\d+)?)"
    r"(?:.*?preprocess_ms=(?P<preprocess>\d+(?:\.\d+)?))?"
    r"(?:.*?inference_ms=(?P<inference>\d+(?:\.\d+)?))?"
)
CSV_HEADER = ["timestamp_iso", "latency_ms", "preprocess_ms", "inference_ms"]


def _rel(path: Path) -> str:
    try:
        return str(path.relative_to(REPO_ROOT))
    except ValueError:
        return str(path)


def parse_line(pattern: re.Pattern, line: str) -> dict | None:
    match = pattern.search(line)
    if not match:
        return None
    g = match.groupdict()
    out: dict[str, float | None] = {
        "latency_ms": float(g["latency"]) if g.get("latency") else None,
        "preprocess_ms": float(g["preprocess"]) if g.get("preprocess") else None,
        "inference_ms": float(g["inference"]) if g.get("inference") else None,
    }
    if out["latency_ms"] is None:
        return None
    return out


def collect_serial(args: argparse.Namespace, pattern: re.Pattern) -> list[dict]:
    try:
        import serial  # type: ignore
    except ImportError as exc:
        raise SystemExit(
            "pyserial is required for serial mode (pip install pyserial). "
            f"Underlying error: {exc}"
        ) from exc

    print(f"opening {args.port} @ {args.baud} ...", flush=True)
    try:
        ser = serial.Serial(args.port, args.baud, timeout=args.timeout)
    except serial.SerialException as exc:
        raise SystemExit(f"serial open failed: {exc}") from exc

    samples: list[dict] = []
    deadline = time.time() + args.duration if args.duration > 0 else None
    duration_str = f" or {args.duration}s" if args.duration > 0 else ""
    print(f"listening for up to {args.num_samples} samples{duration_str}; "
          "press Ctrl-C to stop early.", flush=True)

    try:
        while len(samples) < args.num_samples:
            if deadline is not None and time.time() > deadline:
                print("duration deadline reached", flush=True)
                break
            try:
                raw = ser.readline()
            except serial.SerialException as exc:
                print(f"serial read error: {exc}", file=sys.stderr)
                break
            if not raw:
                continue
            try:
                line = raw.decode("utf-8", errors="replace").rstrip()
            except Exception:  # noqa: BLE001
                continue
            entry = parse_line(pattern, line)
            if entry is None:
                if args.verbose:
                    print(f"[skip] {line}", flush=True)
                continue
            entry["timestamp_iso"] = time.strftime("%Y-%m-%dT%H:%M:%S")
            samples.append(entry)
            print(f"  [{len(samples):4d}/{args.num_samples}] "
                  f"latency={entry['latency_ms']:.2f}ms "
                  f"pre={entry.get('preprocess_ms')} "
                  f"inf={entry.get('inference_ms')}", flush=True)
    except KeyboardInterrupt:
        print("interrupted by user", flush=True)
    finally:
        ser.close()
    return samples


def collect_replay(args: argparse.Namespace, pattern: re.Pattern) -> list[dict]:
    samples: list[dict] = []
    with args.replay.open(encoding="utf-8", errors="replace") as fh:
        for line in fh:
            entry = parse_line(pattern, line.rstrip())
            if entry is None:
                continue
            entry["timestamp_iso"] = ""
            samples.append(entry)
            if len(samples) >= args.num_samples:
                break
    print(f"parsed {len(samples)} latency lines from {_rel(args.replay)}",
          flush=True)
    return samples


def write_csv(path: Path, samples: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(CSV_HEADER)
        for s in samples:
            writer.writerow([
                s.get("timestamp_iso", ""),
                f"{s['latency_ms']:.4f}",
                "" if s.get("preprocess_ms") is None else f"{s['preprocess_ms']:.4f}",
                "" if s.get("inference_ms") is None else f"{s['inference_ms']:.4f}",
            ])


def percentile_table(values: list[float], name: str) -> list[str]:
    if not values:
        return [f"_no `{name}` samples_", ""]
    arr = np.asarray(values, dtype=np.float64)
    rows = [f"### `{name}` (n = {len(values)})", ""]
    rows.append("| stat | value (ms) |")
    rows.append("|---|---:|")
    rows.append(f"| min | {arr.min():.2f} |")
    rows.append(f"| p50 (median) | {np.percentile(arr, 50):.2f} |")
    rows.append(f"| p90 | {np.percentile(arr, 90):.2f} |")
    rows.append(f"| p95 | {np.percentile(arr, 95):.2f} |")
    rows.append(f"| p99 | {np.percentile(arr, 99):.2f} |")
    rows.append(f"| max | {arr.max():.2f} |")
    rows.append(f"| mean | {arr.mean():.2f} |")
    rows.append(f"| std | {arr.std(ddof=1) if len(values) > 1 else 0.0:.2f} |")
    rows.append("")
    rows.append(f"FPS-equivalent at p50: {1000.0 / np.percentile(arr, 50):.2f}")
    rows.append("")
    return rows


def write_histogram(png_path: Path, values: list[float]) -> bool:
    if not values:
        return False
    try:
        import matplotlib  # type: ignore
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt  # type: ignore
    except ImportError:
        print("matplotlib not available; skipping histogram", flush=True)
        return False

    arr = np.asarray(values, dtype=np.float64)
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(arr, bins=min(40, max(10, len(arr) // 5)),
            color="#3e7cb1", alpha=0.85, edgecolor="white")
    ax.axvline(np.percentile(arr, 50), color="#c0392b", linestyle="--",
               label=f"p50 = {np.percentile(arr, 50):.1f} ms")
    ax.axvline(np.percentile(arr, 95), color="#27ae60", linestyle="--",
               label=f"p95 = {np.percentile(arr, 95):.1f} ms")
    ax.set_xlabel("latency (ms)")
    ax.set_ylabel("frame count")
    ax.set_title(f"On-device end-to-end latency distribution (n={len(arr)})")
    ax.legend()
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    png_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(png_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    return True


def write_report(md_path: Path, samples: list[dict], png_path: Path | None,
                 source_label: str) -> None:
    if not samples:
        body = ["# On-device latency report", "",
                f"_source_: {source_label}", "",
                "_no samples collected_"]
        md_path.parent.mkdir(parents=True, exist_ok=True)
        md_path.write_text("\n".join(body) + "\n")
        return

    latency = [s["latency_ms"] for s in samples]
    preprocess = [s["preprocess_ms"] for s in samples
                  if s.get("preprocess_ms") is not None]
    inference = [s["inference_ms"] for s in samples
                 if s.get("inference_ms") is not None]

    lines: list[str] = []
    lines.append("# On-device latency report")
    lines.append("")
    lines.append(f"- Source: {source_label}")
    lines.append(f"- Samples: {len(samples)}")
    lines.append("- Hardware: XIAO ESP32-S3 Sense (ESP32-S3, 240 MHz, "
                 "8 MiB PSRAM, 8 MiB flash) — TensorFlow Lite Micro")
    lines.append("- Tensor arena lives in PSRAM (declared 1 MiB; see "
                 "`esp32/main/inference.cpp`).")
    if png_path is not None and png_path.exists():
        lines.append(f"- Histogram: `{_rel(png_path)}`")
    lines.append("")

    lines.extend(percentile_table(latency, "latency_ms"))
    lines.extend(percentile_table(preprocess, "preprocess_ms"))
    lines.extend(percentile_table(inference, "inference_ms"))

    lines.append("## Notes")
    lines.append("")
    lines.append("- p50 / p95 are the headline numbers for the report's "
                 "real-world test section.")
    lines.append("- If `preprocess_ms` and `inference_ms` are missing, your "
                 "firmware log line only emitted the aggregate `latency_ms`. "
                 "Add per-stage timing in `esp32/main/main.cpp` if you want a "
                 "breakdown.")
    lines.append("- Re-run with the `--replay` flag against an "
                 "`idf.py monitor` log file to reprocess without re-flashing.")

    md_path.parent.mkdir(parents=True, exist_ok=True)
    md_path.write_text("\n".join(lines) + "\n")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", help="serial device, e.g. /dev/cu.usbmodem*")
    parser.add_argument("--baud", type=int, default=115200,
                        help="serial baud rate (default 115200)")
    parser.add_argument("--num-samples", type=int, default=200,
                        help="stop after this many parsed samples")
    parser.add_argument("--duration", type=float, default=0,
                        help="optional wall-clock cap in seconds (0 = no cap)")
    parser.add_argument("--timeout", type=float, default=2.0)
    parser.add_argument("--regex", default=DEFAULT_REGEX,
                        help="regex with named groups latency, preprocess, "
                             "inference (latency required)")
    parser.add_argument("--replay", type=Path,
                        help="parse a captured log file instead of opening "
                             "the serial port")
    parser.add_argument("--csv", type=Path, default=DEFAULT_CSV)
    parser.add_argument("--report", type=Path, default=DEFAULT_MD)
    parser.add_argument("--png", type=Path, default=DEFAULT_PNG)
    parser.add_argument("--no-histogram", action="store_true")
    parser.add_argument("--verbose", action="store_true",
                        help="print non-matching serial lines")
    args = parser.parse_args()

    pattern = re.compile(args.regex)

    if args.replay is not None:
        samples = collect_replay(args, pattern)
        source_label = f"`{_rel(args.replay)}`"
    else:
        if not args.port:
            parser.error("--port is required unless --replay is given")
        samples = collect_serial(args, pattern)
        source_label = f"serial {args.port} @ {args.baud}"

    write_csv(args.csv, samples)
    print(f"wrote {_rel(args.csv)} ({len(samples)} rows)", flush=True)

    png = None
    latency = [s["latency_ms"] for s in samples]
    if latency and not args.no_histogram:
        if write_histogram(args.png, latency):
            png = args.png
            print(f"wrote {_rel(args.png)}", flush=True)

    write_report(args.report, samples, png, source_label)
    print(f"wrote {_rel(args.report)}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
