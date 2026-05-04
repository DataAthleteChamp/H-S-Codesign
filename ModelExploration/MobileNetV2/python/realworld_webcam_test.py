"""Desktop-webcam real-world proxy test for the deployed TFLite model.

Captures frames from the host machine's webcam, runs the same MediaPipe
face-detect → 96×96 → MobileNetV2 normalization → INT8 TFLite inference
that the firmware performs on device, and records per-frame predictions
to `bench/results/realworld_webcam.csv` plus a summary to
`bench/results/realworld_webcam.md`.

This is **not** an on-device test. It is a desktop proxy that exercises
the deployed `model.tflite` against live faces, separate from the
n=60 originals test set. Intended for the `p7-real-world-proxy` task
documented in the project plan.

Usage:
    # capture 30 frames per team member
    python python/realworld_webcam_test.py --label Amine --num-frames 30
    python python/realworld_webcam_test.py --label Rifki --num-frames 30
    python python/realworld_webcam_test.py --label Jakub --num-frames 30
    # capture 30 frames of a non-team person to test rejection
    python python/realworld_webcam_test.py --label none --num-frames 30
    # render aggregate summary across everything captured so far
    python python/realworld_webcam_test.py --finalize

Captured CSV row: timestamp, session_id, frame_idx, label, predicted,
top1_prob, accepted, p_amine, p_rifki, p_jakub.
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
import time
import uuid
from datetime import datetime
from pathlib import Path

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_TFLITE = REPO_ROOT / "python" / "gen" / "model.tflite"
DEFAULT_CSV = REPO_ROOT / "bench" / "results" / "realworld_webcam.csv"
DEFAULT_REPORT = REPO_ROOT / "bench" / "results" / "realworld_webcam.md"

CLASS_NAMES = ("Amine", "Rifki", "Jakub")
NONE_LABEL = "none"
VALID_LABELS = (*CLASS_NAMES, NONE_LABEL)
DEFAULT_THRESHOLD = 0.77        # matches firmware operating point q=0.77
DEFAULT_NUM_FRAMES = 30
DEFAULT_INTERVAL = 0.5
DEFAULT_DEVICE = 0
IMG_SIZE = 96

CSV_HEADER = [
    "timestamp_iso",
    "session_id",
    "frame_idx",
    "label",
    "predicted",
    "top1_prob",
    "accepted",
    "p_amine",
    "p_rifki",
    "p_jakub",
]


def _import_optional() -> tuple:
    """Import cv2 / mediapipe / tflite lazily so --finalize works without them."""
    try:
        import cv2  # type: ignore
    except ImportError as exc:
        raise SystemExit(
            "opencv-python is required for capture (pip install opencv-python). "
            f"Underlying error: {exc}"
        ) from exc
    try:
        import mediapipe as mp  # type: ignore
        from mediapipe.tasks.python import vision  # type: ignore
        from mediapipe.tasks.python.core.base_options import BaseOptions  # type: ignore
    except ImportError as exc:
        raise SystemExit(
            "mediapipe is required for capture (pip install mediapipe). "
            f"Underlying error: {exc}"
        ) from exc
    try:
        import tensorflow as tf  # type: ignore
    except ImportError as exc:
        raise SystemExit(
            "tensorflow is required for capture (pip install tensorflow). "
            f"Underlying error: {exc}"
        ) from exc
    return cv2, mp, vision, BaseOptions, tf


def detect_and_crop(img_rgb: np.ndarray, detector) -> np.ndarray | None:
    """Mirror python/preprocess.py::detect_and_crop_face exactly."""
    import mediapipe as mp  # type: ignore

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
    return None


def center_crop_square(img: np.ndarray) -> np.ndarray:
    h, w, _ = img.shape
    if h > w:
        offset = (h - w) // 2
        return img[offset:offset + w, :]
    offset = (w - h) // 2
    return img[:, offset:offset + h]


def quantize_input(face_pm1: np.ndarray, scale: float, zero_point: int) -> np.ndarray:
    """Match firmware: quantize float [-1,1] tensor to INT8 with the model's quant params."""
    raw = np.round(face_pm1 / scale + zero_point)
    raw = np.clip(raw, -128, 127).astype(np.int8)
    return raw


def dequantize_output(out_int8: np.ndarray, scale: float, zero_point: int) -> np.ndarray:
    return (out_int8.astype(np.float32) - zero_point) * scale


def _rel(path: Path) -> str:
    try:
        return str(path.relative_to(REPO_ROOT))
    except ValueError:
        return str(path)


def append_csv(csv_path: Path, rows: list[list]) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    new_file = not csv_path.exists()
    with csv_path.open("a", newline="") as fh:
        writer = csv.writer(fh)
        if new_file:
            writer.writerow(CSV_HEADER)
        writer.writerows(rows)


def capture(args: argparse.Namespace) -> int:
    if args.label not in VALID_LABELS:
        print(f"--label must be one of {VALID_LABELS}", file=sys.stderr)
        return 2

    cv2, mp, vision, BaseOptions, tf = _import_optional()

    face_model = REPO_ROOT / "python" / "blaze_face_short_range.tflite"
    if not face_model.exists():
        print(f"missing MediaPipe BlazeFace model: {face_model}", file=sys.stderr)
        return 2

    options = vision.FaceDetectorOptions(
        base_options=BaseOptions(model_asset_path=str(face_model)),
        min_detection_confidence=0.5,
    )
    detector = vision.FaceDetector.create_from_options(options)

    interpreter = tf.lite.Interpreter(model_path=str(args.tflite))
    interpreter.allocate_tensors()
    in_det = interpreter.get_input_details()[0]
    out_det = interpreter.get_output_details()[0]
    in_scale, in_zp = in_det["quantization"]
    out_scale, out_zp = out_det["quantization"]

    cap = cv2.VideoCapture(args.device)
    if not cap.isOpened():
        print(f"could not open webcam device {args.device}", file=sys.stderr)
        return 2

    print(f"Recording label='{args.label}' for {args.num_frames} frames "
          f"@ {args.interval}s/frame. Press 'q' in the preview window to abort.")

    session_id = uuid.uuid4().hex[:8]
    rows: list[list] = []
    saved = 0
    last_save = 0.0
    skipped_no_face = 0

    try:
        while saved < args.num_frames:
            ok, frame_bgr = cap.read()
            if not ok:
                print("webcam read failed; aborting", file=sys.stderr)
                break
            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

            crop = detect_and_crop(frame_rgb, detector)
            if crop is None:
                crop_used = "fallback_center"
                crop = center_crop_square(frame_rgb)
            else:
                crop_used = "mediapipe"

            face_resized = cv2.resize(crop, (IMG_SIZE, IMG_SIZE),
                                      interpolation=cv2.INTER_AREA)
            face_pm1 = face_resized.astype(np.float32) / 127.5 - 1.0
            x = quantize_input(face_pm1, in_scale, in_zp)
            interpreter.set_tensor(in_det["index"], x[np.newaxis, ...])
            interpreter.invoke()
            out_q = interpreter.get_tensor(out_det["index"])[0]
            probs = dequantize_output(out_q, out_scale, out_zp)
            probs = np.clip(probs, 0.0, 1.0)
            if probs.sum() > 0:
                probs = probs / probs.sum()

            top1 = int(np.argmax(probs))
            top1_prob = float(probs[top1])
            predicted = CLASS_NAMES[top1]
            accepted = top1_prob >= args.threshold

            now = time.time()
            if now - last_save < args.interval:
                cv2.imshow("realworld_webcam_test", frame_bgr)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    print("aborted by user")
                    break
                continue

            last_save = now
            if crop_used == "fallback_center":
                skipped_no_face += 1

            rows.append([
                datetime.now().isoformat(timespec="seconds"),
                session_id,
                saved,
                args.label,
                predicted,
                f"{top1_prob:.4f}",
                int(bool(accepted)),
                f"{float(probs[0]):.4f}",
                f"{float(probs[1]):.4f}",
                f"{float(probs[2]):.4f}",
            ])
            saved += 1
            tag = "ACC" if accepted else "REJ"
            print(f"  [{saved:3d}/{args.num_frames}] {tag} pred={predicted:<6s} "
                  f"prob={top1_prob:.3f}  detector={crop_used}")

            cv2.imshow("realworld_webcam_test", frame_bgr)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                print("aborted by user")
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()

    if rows:
        append_csv(args.csv, rows)
        print(f"wrote {len(rows)} rows to {_rel(args.csv)}")
        if skipped_no_face:
            print(f"  warning: {skipped_no_face}/{len(rows)} frames had no face "
                  "detected; fallback center-crop was used.")
    else:
        print("no frames captured")
    return 0


def finalize(args: argparse.Namespace) -> int:
    if not args.csv.exists():
        print(f"missing CSV: {args.csv}", file=sys.stderr)
        return 2

    with args.csv.open() as fh:
        reader = csv.DictReader(fh)
        records = list(reader)
    if not records:
        print(f"empty CSV: {args.csv}", file=sys.stderr)
        return 2

    by_label: dict[str, list[dict]] = {}
    for r in records:
        by_label.setdefault(r["label"], []).append(r)

    lines: list[str] = []
    lines.append("# Real-world webcam proxy results")
    lines.append("")
    lines.append(f"- Source: `{_rel(args.csv)}` "
                 f"({len(records)} frames)")
    lines.append(f"- Threshold: q ≥ {args.threshold:.2f}")
    lines.append("- This is a desktop webcam proxy, **not** an on-device test.")
    lines.append("- Pipeline: MediaPipe BlazeFace → 96×96 INT8 → "
                 "`python/gen/model.tflite`")
    lines.append("")

    lines.append("## Per-label summary")
    lines.append("")
    lines.append("| label | frames | top-1 acc (correct/n) | accept-rate | "
                 "acc on accepted | mean top-1 prob |")
    lines.append("|---|---:|---:|---:|---:|---:|")

    overall_correct = 0
    overall_total = 0
    overall_accept = 0
    overall_accepted_correct = 0

    for label in sorted(by_label):
        rows = by_label[label]
        n = len(rows)
        accepted = [r for r in rows if int(r["accepted"]) == 1]
        if label == NONE_LABEL:
            # for non-team frames, "correct" means rejected
            correct = [r for r in rows if int(r["accepted"]) == 0]
            accepted_correct = []
        else:
            correct = [r for r in rows if r["predicted"] == label]
            accepted_correct = [r for r in accepted if r["predicted"] == label]
        mean_p = float(np.mean([float(r["top1_prob"]) for r in rows])) if n else 0.0
        if label != NONE_LABEL:
            overall_correct += len(correct)
            overall_total += n
            overall_accept += len(accepted)
            overall_accepted_correct += len(accepted_correct)

        acc_pct = (100.0 * len(correct) / n) if n else 0.0
        accept_pct = (100.0 * len(accepted) / n) if n else 0.0
        if accepted:
            acc_on_acc_pct = 100.0 * len(accepted_correct) / len(accepted)
            acc_on_acc_str = f"{acc_on_acc_pct:.2f}%"
        else:
            acc_on_acc_str = "n/a"
        lines.append(
            f"| {label} | {n} | {len(correct)}/{n} ({acc_pct:.2f}%) | "
            f"{len(accepted)}/{n} ({accept_pct:.2f}%) | "
            f"{len(accepted_correct)}/{len(accepted)} ({acc_on_acc_str}) | "
            f"{mean_p:.3f} |"
        )
    lines.append("")

    if overall_total:
        lines.append("## Team-class headline (excluding `none`)")
        lines.append("")
        lines.append("| metric | value |")
        lines.append("|---|---:|")
        lines.append(f"| frames | {overall_total} |")
        lines.append(f"| top-1 accuracy | {overall_correct}/{overall_total} "
                     f"({100.0*overall_correct/overall_total:.2f}%) |")
        lines.append(f"| accept rate (q ≥ {args.threshold:.2f}) | "
                     f"{overall_accept}/{overall_total} "
                     f"({100.0*overall_accept/overall_total:.2f}%) |")
        if overall_accept:
            lines.append(f"| accuracy on accepted | "
                         f"{overall_accepted_correct}/{overall_accept} "
                         f"({100.0*overall_accepted_correct/overall_accept:.2f}%) |")
        lines.append("")

    if NONE_LABEL in by_label:
        none_rows = by_label[NONE_LABEL]
        rejected = [r for r in none_rows if int(r["accepted"]) == 0]
        lines.append("## Non-team rejection panel (`label = none`)")
        lines.append("")
        lines.append("| metric | value |")
        lines.append("|---|---:|")
        lines.append(f"| frames | {len(none_rows)} |")
        lines.append(f"| rejected (good) | {len(rejected)}/{len(none_rows)} "
                     f"({100.0*len(rejected)/len(none_rows):.2f}%) |")
        lines.append(f"| accepted (false-accept) | "
                     f"{len(none_rows)-len(rejected)}/{len(none_rows)} |")
        lines.append("")

    lines.append("## Caveats")
    lines.append("")
    lines.append("- Desktop webcam optics differ from the OV2640 onboard camera "
                 "(field of view, white balance, JPEG pipeline). Numbers should "
                 "be read as a sanity check, not a substitute for live "
                 "on-device measurement.")
    lines.append("- MediaPipe BlazeFace is used as a face crop fallback; the "
                 "firmware does not run BlazeFace and instead expects the "
                 "OV2640's already-captured framing.")
    lines.append("- Sample sizes are small. Take Wilson 95 % CIs from the CSV "
                 "if you want bounded numbers.")

    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text("\n".join(lines) + "\n")
    print(f"wrote {_rel(args.report)}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tflite", type=Path, default=DEFAULT_TFLITE)
    parser.add_argument("--csv", type=Path, default=DEFAULT_CSV)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--threshold", type=float, default=DEFAULT_THRESHOLD,
                        help="firmware-equivalent rejection threshold q "
                             f"(default {DEFAULT_THRESHOLD})")
    parser.add_argument("--label", choices=VALID_LABELS,
                        help="ground-truth label for this capture session")
    parser.add_argument("--num-frames", type=int, default=DEFAULT_NUM_FRAMES)
    parser.add_argument("--interval", type=float, default=DEFAULT_INTERVAL,
                        help="seconds between saved frames")
    parser.add_argument("--device", type=int, default=DEFAULT_DEVICE,
                        help="webcam device index for cv2.VideoCapture")
    parser.add_argument("--finalize", action="store_true",
                        help="aggregate the CSV into the markdown report and exit")
    args = parser.parse_args()

    if args.finalize:
        return finalize(args)

    if args.label is None:
        parser.error("--label is required unless --finalize is given")
    if not args.tflite.exists():
        parser.error(f"missing TFLite model: {args.tflite}")

    rc = capture(args)
    if rc == 0:
        finalize(args)
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
