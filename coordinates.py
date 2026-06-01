"""
Universal Meter Reader — YOLO + EasyOCR (100% Local, No API Key)
================================================================
Detects any meter/gauge in a video using YOLOv8,
then reads the value using EasyOCR — best for digital displays.

SETUP (one time only):
    pip install ultralytics opencv-python easyocr numpy Pillow

USAGE:
    python coordinates_ocr.py --video "videos\MVI_3220 - Trim.mp4"
    python coordinates_ocr.py --video "videos\MVI_3220 - Trim.mp4" --interval 3
    python coordinates_ocr.py --video "videos\MVI_3220 - Trim.mp4" --no-frames
"""

import cv2
import numpy as np
import argparse
import os
import sys
import re
from datetime import datetime
from PIL import Image

# ─────────────────────────────────────────────────────────────
#  CONFIGURATION
# ─────────────────────────────────────────────────────────────

# YOLO model — "yolov8n.pt" fastest; "yolov8m.pt" more accurate
YOLO_MODEL = "yolov8n.pt"

# Sample one frame every N seconds of video
FRAME_INTERVAL_SECONDS = 2.0

# Save annotated frames?
SAVE_FRAMES = True
FRAMES_DIR  = "meter_frames_ocr"

# Save readings to CSV?
LOG_TO_CSV = True
CSV_PATH   = "meter_readings_ocr.csv"

# YOLO confidence threshold
YOLO_CONF = 0.25

# Padding around detected meter crop
CROP_PADDING = 0.15

# EasyOCR languages — add more if needed e.g. ['en', 'hi']
OCR_LANGUAGES = ['en']

# Known units to search for in OCR text
KNOWN_UNITS = [
    "psi", "bar", "kpa", "mpa", "pa",
    "°c", "°f", "c", "f",
    "kwh", "kw", "mw", "w",
    "v", "mv", "kv",
    "a", "ma",
    "rpm", "hz",
    "l/min", "m3/h", "gpm",
    "kg/cm2", "kg/cm²",
    "%",
]


# ─────────────────────────────────────────────────────────────
#  YOLO DETECTOR
# ─────────────────────────────────────────────────────────────

class MeterDetector:
    def __init__(self, model_name: str = YOLO_MODEL):
        try:
            from ultralytics import YOLO
            print(f"[YOLO] Loading model: {model_name} ...")
            self.model = YOLO(model_name)
            self.available = True
            print("[YOLO] Model loaded ✓")
        except ImportError:
            print("[WARN] ultralytics not installed — using full frame.")
            print("       Fix: pip install ultralytics")
            self.available = False

    def detect(self, frame: np.ndarray) -> dict:
        if not self.available:
            h, w = frame.shape[:2]
            return {"bbox": (0, 0, w, h), "conf": 1.0, "label": "full_frame"}

        results = self.model(frame, conf=YOLO_CONF, verbose=False)[0]
        best = None
        for box in results.boxes:
            conf = float(box.conf[0])
            if best is None or conf > best["conf"]:
                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                best = {
                    "bbox": (x1, y1, x2, y2),
                    "conf": conf,
                    "label": results.names[int(box.cls[0])],
                }

        if best is None:
            h, w = frame.shape[:2]
            best = {"bbox": (0, 0, w, h), "conf": 0.5, "label": "auto_full"}

        return best


# ─────────────────────────────────────────────────────────────
#  IMAGE PREPROCESSOR
# ─────────────────────────────────────────────────────────────

def preprocess_for_ocr(crop: np.ndarray) -> list[np.ndarray]:
    """
    Returns multiple preprocessed versions of the crop.
    EasyOCR tries all of them and picks the best result.
    """
    # Resize to at least 200px tall for better OCR accuracy
    h, w = crop.shape[:2]
    if h < 200:
        scale = 200 / h
        crop = cv2.resize(crop, (int(w * scale), 200), interpolation=cv2.INTER_CUBIC)

    gray    = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (3, 3), 0)

    # Version 1: plain grayscale
    v1 = gray

    # Version 2: contrast enhanced (CLAHE)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    v2 = clahe.apply(blurred)

    # Version 3: binary threshold (good for digital displays)
    _, v3 = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Version 4: inverted binary (white text on dark background)
    v4 = cv2.bitwise_not(v3)

    # Version 5: adaptive threshold (good for uneven lighting)
    v5 = cv2.adaptiveThreshold(blurred, 255,
                                cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                cv2.THRESH_BINARY, 11, 2)

    return [v1, v2, v3, v4, v5]





# ─────────────────────────────────────────────────────────────
#  HELPERS
# ─────────────────────────────────────────────────────────────

def crop_with_padding(frame: np.ndarray, bbox: tuple, padding: float = CROP_PADDING) -> np.ndarray:
    h, w = frame.shape[:2]
    x1, y1, x2, y2 = bbox
    px = int((x2 - x1) * padding)
    py = int((y2 - y1) * padding)
    return frame[max(0, y1-py):min(h, y2+py), max(0, x1-px):min(w, x2+px)]


def format_timestamp(seconds: float) -> str:
    m, s = divmod(int(seconds), 60)
    h, m = divmod(m, 60)
    return f"{h:02d}:{m:02d}:{s:02d}"


def annotate_frame(frame: np.ndarray, detection: dict, reading: dict,
                   video_ts: str, frame_no: int) -> np.ndarray:
    out = frame.copy()
    x1, y1, x2, y2 = detection["bbox"]

    cv2.rectangle(out, (x1, y1), (x2, y2), (0, 200, 255), 2)
    cv2.putText(out, f"{detection['label']} {detection['conf']:.2f}",
                (x1, max(y1 - 8, 12)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 200, 255), 1)

    val      = reading.get("value")
    unit     = reading.get("unit") or ""
    conf     = reading.get("confidence", 0)
    all_text = (reading.get("all_text") or "")[:55]
    notes    = (reading.get("notes") or "")[:55]

    lines = [
        f"  {val} {unit}" if val is not None else "  Reading: ---",
        f"  OCR text : {all_text}" if all_text else "",
        f"  Conf     : {int(conf * 100)}%",
        f"  {notes}" if notes and val is None else "",
        f"  Time     : {video_ts}  Frame: {frame_no}",
    ]
    lines = [l for l in lines if l]

    line_h, panel_w = 26, 400
    panel_h = len(lines) * line_h + 16
    overlay = out.copy()
    cv2.rectangle(overlay, (8, 8), (8 + panel_w, 8 + panel_h), (20, 20, 20), -1)
    cv2.addWeighted(overlay, 0.65, out, 0.35, 0, out)

    for i, line in enumerate(lines):
        color = (0, 200, 255) if i == 0 and val is not None else (200, 200, 200)
        size  = 0.85 if i == 0 else 0.60
        cv2.putText(out, line, (12, 8 + 18 + i * line_h),
                    cv2.FONT_HERSHEY_SIMPLEX, size, color, 2 if i == 0 else 1)
    return out


# ─────────────────────────────────────────────────────────────
#  CSV LOGGER
# ─────────────────────────────────────────────────────────────

class CSVLogger:
    def __init__(self, path: str):
        self.path = path
        if not os.path.exists(path):
            with open(path, "w") as f:
                f.write("frame,video_timestamp,value,unit,confidence,all_ocr_text,notes\n")

    def log(self, frame_no: int, video_ts: str, reading: dict):
        row = ",".join([
            str(frame_no),
            video_ts,
            str(reading.get("value", "")),
            str(reading.get("unit", "")),
            str(reading.get("confidence", "")),
            str(reading.get("all_text", "")).replace(",", ";"),
            str(reading.get("notes", "")).replace(",", ";"),
        ])
        with open(self.path, "a") as f:
            f.write(row + "\n")


# ─────────────────────────────────────────────────────────────
#  MAIN — VIDEO PROCESSING
# ─────────────────────────────────────────────────────────────

def process_video(video_path: str):
    if not os.path.isfile(video_path):
        print(f"ERROR: File not found: {video_path}")
        sys.exit(1)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"ERROR: Cannot open video: {video_path}")
        sys.exit(1)

    fps      = cap.get(cv2.CAP_PROP_FPS) or 25.0
    total    = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = total / fps
    step     = max(1, int(fps * FRAME_INTERVAL_SECONDS))

    print(f"\n{'─'*55}")
    print(f"  Video    : {os.path.basename(video_path)}")
    print(f"  Duration : {format_timestamp(duration)}  ({total} frames @ {fps:.1f} fps)")
    print(f"  Sampling : every {FRAME_INTERVAL_SECONDS}s  ({step} frames)")
    print(f"  Engine   : EasyOCR (local, no API key)")
    print(f"{'─'*55}\n")

    detector = MeterDetector()
    reader   = OCRReader()
    logger   = CSVLogger(CSV_PATH) if LOG_TO_CSV else None

    if SAVE_FRAMES:
        os.makedirs(FRAMES_DIR, exist_ok=True)

    frame_no = 0
    sampled  = 0
    readings = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_no % step == 0:
            video_ts  = format_timestamp(frame_no / fps)
            detection = detector.detect(frame)
            crop      = crop_with_padding(frame, detection["bbox"])

            print(f"[{video_ts}] Frame {frame_no:5d} — reading...", end=" ", flush=True)
            reading = reader.read(crop)

            val  = reading.get("value")
            unit = reading.get("unit", "")
            conf = reading.get("confidence", 0)

            if val is not None:
                print(f"{val} {unit}  (conf: {int(conf*100)}%)")
            else:
                print(f"--- ({reading.get('notes', 'no reading')[:60]})")

            readings.append({"frame": frame_no, "ts": video_ts, **reading})

            if logger:
                logger.log(frame_no, video_ts, reading)

            if SAVE_FRAMES:
                annotated = annotate_frame(frame, detection, reading, video_ts, frame_no)
                safe_ts   = video_ts.replace(":", "-")
                fname     = os.path.join(FRAMES_DIR, f"frame_{frame_no:06d}_{safe_ts}.jpg")
                cv2.imwrite(fname, annotated)

            sampled += 1

        frame_no += 1

    cap.release()

    # ── Summary ──────────────────────────────────────────────
    valid = [r for r in readings if r.get("value") is not None]
    print(f"\n{'─'*55}")
    print(f"  Frames sampled   : {sampled}")
    print(f"  Successful reads : {len(valid)} / {sampled}")
    if valid:
        values = [r["value"] for r in valid]
        unit   = next((r.get("unit") for r in valid if r.get("unit")), "")
        print(f"  Min reading      : {min(values)} {unit}")
        print(f"  Max reading      : {max(values)} {unit}")
        print(f"  Avg reading      : {sum(values)/len(values):.2f} {unit}")
    if LOG_TO_CSV:
        print(f"  CSV saved to     : {CSV_PATH}")
    if SAVE_FRAMES:
        print(f"  Frames saved to  : {FRAMES_DIR}/")
    print(f"{'─'*55}\n")


# ─────────────────────────────────────────────────────────────
#  ENTRY POINT
# ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Meter Reader — EasyOCR (Local, No API Key, Best for digital displays)"
    )
    parser.add_argument("--video",     type=str, required=True,
                        help="Path to input video file")
    parser.add_argument("--interval",  type=float, default=FRAME_INTERVAL_SECONDS,
                        help=f"Seconds between sampled frames (default: {FRAME_INTERVAL_SECONDS})")
    parser.add_argument("--no-log",    action="store_true", help="Disable CSV output")
    parser.add_argument("--no-frames", action="store_true", help="Disable saving annotated frames")
    args = parser.parse_args()

    FRAME_INTERVAL_SECONDS = args.interval
    if args.no_log:
        LOG_TO_CSV = False
    if args.no_frames:
        SAVE_FRAMES = False

    process_video(args.video)
