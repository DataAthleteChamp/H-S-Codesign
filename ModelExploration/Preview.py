import argparse
import csv
import os
import sys
import time
from datetime import datetime

import pygame
import serial


DEFAULT_PORT = "COM7"
DEFAULT_OUTPUT_PATH = "labeled_pred_frames"
DEFAULT_SAVE_INTERVAL_SECONDS = 2.0
BAUD_RATE = 921600
SERIAL_TIMEOUT = 10.0
WIDTH = 320
HEIGHT = 240
FRAME_PREAMBLE = b"===FRAME===\n"
PREDICTION_PREAMBLE = b"===PRED===\n"
FRAME_SIZE = WIDTH * HEIGHT * 3
CLASS_NAMES = ("Amine", "Rifki", "Jakub")
PREDICTION_THRESHOLD = 0.85


def preview_stream(port: str, output_path: str, save_interval_seconds: float):
    print(f"Opening serial port {port}... ", end="")
    preamble_timeouts = 0
    try:
        serial_port = serial.Serial(port, BAUD_RATE, timeout=SERIAL_TIMEOUT)
        serial_port.reset_input_buffer()
    except serial.SerialException as exc:
        print(f"failed: {exc}", file=sys.stderr)
        return

    pygame.init()
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption("ESP32 Preview")
    font = pygame.font.Font(None, 24)

    print("connected.")
    print("Press SPACE to save current annotated frame.")
    print(f"Saving one frame and prediction every {save_interval_seconds:g} seconds.")
    print("Press 'r' to toggle timed recording.")
    print("Press 's' to toggle streaming, 'q' or ESC to quit.")
    serial_port.write(b"1")

    os.makedirs(output_path, exist_ok=True)
    last_surface = None
    last_prediction = None
    recording = True
    last_save_time = 0.0
    running = True
    try:
        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.KEYDOWN:
                    if event.key in (pygame.K_q, pygame.K_ESCAPE):
                        running = False
                    elif event.key == pygame.K_s:
                        serial_port.write(b"S")
                    elif event.key == pygame.K_r:
                        recording = not recording
                        print(f"Recording {'enabled' if recording else 'disabled'}.")
                    elif event.key == pygame.K_SPACE and last_surface is not None:
                        save_frame(output_path, last_surface, last_prediction, font)

            result = capture_frame(serial_port)
            if result is None:
                preamble_timeouts += 1
                if preamble_timeouts % 3 == 0:
                    print("No frame preamble after repeated timeouts; re-sending stream enable.")
                    serial_port.write(b"1")
                continue

            preamble_timeouts = 0
            surface, prediction = result
            last_surface = surface.copy()
            last_prediction = prediction
            now = time.monotonic()
            if recording and now - last_save_time >= save_interval_seconds:
                save_frame(output_path, last_surface, last_prediction, font)
                last_save_time = now

            screen.blit(surface, (0, 0))
            draw_prediction_overlay(screen, font, prediction)
            pygame.display.flip()
            time.sleep(0.001)
    finally:
        serial_port.close()
        pygame.quit()


def capture_frame(serial_port: serial.Serial) -> tuple[pygame.Surface, dict[str, object] | None] | None:
    chunk = serial_port.read_until(FRAME_PREAMBLE)
    if not chunk.endswith(FRAME_PREAMBLE):
        print("Preamble timeout, retrying...")
        return None

    prediction = parse_prediction(chunk)

    frame = serial_port.read(FRAME_SIZE)
    if len(frame) != FRAME_SIZE:
        print(f"Incomplete frame received ({len(frame)} bytes), skipping...")
        return None

    return pygame.image.frombuffer(frame, (WIDTH, HEIGHT), "RGB"), prediction


def parse_prediction(chunk: bytes) -> dict[str, object] | None:
    marker_index = chunk.rfind(PREDICTION_PREAMBLE)
    if marker_index < 0:
        return None

    start = marker_index + len(PREDICTION_PREAMBLE)
    end = chunk.find(b"\n", start)
    if end < 0:
        return None

    line = chunk[start:end].decode("utf-8", errors="replace").strip()
    fields = line.split(",")
    if len(fields) != 16:
        return None

    try:
        scores = [float(value) for value in fields[3:6]]
        return {
            "label": fields[0],
            "index": int(fields[1]),
            "confidence": float(fields[2]),
            "scores": scores,
            "detection_state": fields[6],
            "detector_score": float(fields[7]),
            "bbox": tuple(int(value) for value in fields[8:12]),
            "crop": tuple(int(value) for value in fields[12:16]),
        }
    except ValueError:
        return None


def prediction_overlay_lines(prediction: dict[str, object] | None) -> tuple[list[str], str, int, float]:
    if prediction is None:
        return ["Prediction: NO_DATA"], "", -1, 0.0

    scores = [float(score) for score in prediction["scores"]]
    detection_state = str(prediction.get("detection_state", "NONE"))

    if detection_state == "FACE" and scores:
        best_index = max(range(len(scores)), key=scores.__getitem__)
        best_label = CLASS_NAMES[best_index]
        best_score = scores[best_index]
        result_label = best_label if best_score >= PREDICTION_THRESHOLD else "Unknown"
        lines = [
            f"Prediction: {result_label}",
            f"Best: {best_label} ({best_score * 100:.1f}%)",
            "  ".join(f"{name}: {score * 100:.1f}%" for name, score in zip(CLASS_NAMES, scores)),
        ]
        return lines, result_label, best_index, best_score

    return ["Prediction: NO_FACE", "Best: none"], "NO_FACE", -1, 0.0


def draw_prediction_overlay(screen: pygame.Surface, font: pygame.font.Font, prediction: dict[str, object] | None):
    lines, _, _, _ = prediction_overlay_lines(prediction)
    padding = 8
    rendered = [font.render(line, True, (255, 255, 255)) for line in lines]
    width = max(surface.get_width() for surface in rendered) + padding * 2
    height = sum(surface.get_height() for surface in rendered) + padding * 2 + 4

    overlay = pygame.Surface((width, height), pygame.SRCALPHA)
    overlay.fill((0, 0, 0, 180))

    y = padding
    for surface in rendered:
        overlay.blit(surface, (padding, y))
        y += surface.get_height() + 4

    screen.blit(overlay, (8, 8))


def save_frame(
    output_path: str,
    surface: pygame.Surface,
    prediction: dict[str, object] | None,
    font: pygame.font.Font,
):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    image_filename = f"frame_{timestamp}.png"
    image_path = os.path.join(output_path, image_filename)
    annotated_surface = surface.copy()
    draw_prediction_overlay(annotated_surface, font, prediction)
    pygame.image.save(annotated_surface, image_path)

    label = ""
    index = ""
    confidence = ""
    scores = ["", "", ""]
    best_label = ""
    best_index = ""
    best_confidence = ""
    detection_state = ""
    detector_score = ""
    bbox = ["", "", "", ""]
    crop = ["", "", "", ""]
    if prediction is not None:
        _, best_label_value, best_index_value, best_score = prediction_overlay_lines(prediction)
        label = str(prediction["label"])
        index = str(prediction["index"])
        confidence = f"{float(prediction['confidence']):.6f}"
        scores = [f"{score:.6f}" for score in prediction["scores"]]
        best_label = best_label_value
        best_index = str(best_index_value)
        best_confidence = f"{best_score:.6f}"
        detection_state = str(prediction.get("detection_state", ""))
        detector_score = f"{float(prediction.get('detector_score', 0.0)):.6f}"
        bbox = [str(v) for v in prediction.get("bbox", ("", "", "", ""))]
        crop = [str(v) for v in prediction.get("crop", ("", "", "", ""))]

    csv_path = os.path.join(output_path, "frames.csv")
    write_header = not os.path.exists(csv_path)
    with open(csv_path, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        if write_header:
            writer.writerow(
                [
                    "timestamp",
                    "filename",
                    "label",
                    "index",
                    "confidence",
                    "best_label",
                    "best_index",
                    "best_confidence",
                    *CLASS_NAMES,
                    "detection_state",
                    "detector_score",
                    "bbox_x1",
                    "bbox_y1",
                    "bbox_x2",
                    "bbox_y2",
                    "crop_x",
                    "crop_y",
                    "crop_w",
                    "crop_h",
                ]
            )
        writer.writerow(
            [
                timestamp,
                image_filename,
                label,
                index,
                confidence,
                best_label,
                best_index,
                best_confidence,
                *scores,
                detection_state,
                detector_score,
                *bbox,
                *crop,
            ]
        )

    print(f"Saved {image_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Preview frames streamed from the ESP32 over USB serial.")
    parser.add_argument("--port", default=DEFAULT_PORT, help=f"Serial port (default: {DEFAULT_PORT})")
    parser.add_argument(
        "--output-path",
        default=DEFAULT_OUTPUT_PATH,
        help=f"Directory for saved frames and metadata CSV (default: {DEFAULT_OUTPUT_PATH})",
    )
    parser.add_argument(
        "--save-interval",
        type=float,
        default=DEFAULT_SAVE_INTERVAL_SECONDS,
        help=(
            "Seconds between automatic frame/prediction saves "
            f"(default: {DEFAULT_SAVE_INTERVAL_SECONDS:g})"
        ),
    )
    args = parser.parse_args()

    if args.save_interval <= 0:
        parser.error("--save-interval must be greater than 0")

    preview_stream(args.port, args.output_path, args.save_interval)
