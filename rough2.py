import cv2
import numpy as np
import csv, time
from collections import deque

# ─────────────────────────────────────────────────────────────
#  CONFIG  –  adjust these after running the calibration helper
# ─────────────────────────────────────────────────────────────
VIDEO_PATH = r"C:\Users\hp\Desktop\image_processing\videos\MVI_3222.MP4"

PIVOT_X = 468
PIVOT_Y = 369

ANGLE_AT_MIN  = 180
ANGLE_AT_MAX  = 115

FLOW_MIN  =  50.0
FLOW_MAX  = 480.0

CALIBRATE_MODE = True

# ── NEW: only readings inside this range are kept ─────────────
FLOW_VALID_MIN = 200.0   # ← adjust to your expected minimum
FLOW_VALID_MAX = 300.0   # ← adjust to your expected maximum

# ── NEW: median smoothing window (frames) ─────────────────────
SMOOTH_WINDOW = 7

# ─────────────────────────────────────────────────────────────
#  ANGLE → FLOW
# ─────────────────────────────────────────────────────────────
def angle_to_flow(angle_deg):
    total_sweep = (ANGLE_AT_MAX - ANGLE_AT_MIN) % 360
    relative    = (angle_deg    - ANGLE_AT_MIN) % 360
    ratio = relative / total_sweep if total_sweep != 0 else 0.0
    ratio = max(0.0, min(1.0, ratio))
    flow  = round(FLOW_MIN + ratio * (FLOW_MAX - FLOW_MIN), 1)
    return flow


# ─────────────────────────────────────────────────────────────
#  CALIBRATION HELPER
# ─────────────────────────────────────────────────────────────
_click_point = None

def _on_mouse(event, x, y, flags, param):
    global _click_point
    if event == cv2.EVENT_LBUTTONDOWN:
        _click_point = (x, y)
        print(f"  Clicked → PIVOT_X={x}, PIVOT_Y={y}")


def calibrate(frame):
    global _click_point
    frame = cv2.resize(frame, (960, 540))
    overlay = frame.copy()

    cv2.circle(overlay, (PIVOT_X, PIVOT_Y), 8, (0, 255, 255), 2)
    cv2.putText(overlay, f"Current pivot: ({PIVOT_X},{PIVOT_Y})",
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
    cv2.putText(overlay, "Left-click on needle hub to update pivot. Press SPACE to continue.",
                (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 0), 1)

    def ray(angle, color, label):
        rad = np.radians(angle)
        ex  = int(PIVOT_X + 180 * np.cos(rad))
        ey  = int(PIVOT_Y - 180 * np.sin(rad))
        cv2.line(overlay, (PIVOT_X, PIVOT_Y), (ex, ey), color, 2)
        cv2.putText(overlay, label, (ex, ey),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 1)

    ray(ANGLE_AT_MIN, (0, 140, 255), f"MIN {FLOW_MIN:.0f}")
    ray(ANGLE_AT_MAX, (0, 255,   0), f"MAX {FLOW_MAX:.0f}")

    cv2.namedWindow("CALIBRATE – SYN GAS Gauge")
    cv2.setMouseCallback("CALIBRATE – SYN GAS Gauge", _on_mouse)

    while True:
        cv2.imshow("CALIBRATE – SYN GAS Gauge", overlay)
        key = cv2.waitKey(20) & 0xFF
        if key == ord(' '):
            break
        if _click_point:
            print(f"  → Set PIVOT_X={_click_point[0]}, PIVOT_Y={_click_point[1]} in CONFIG")
            _click_point = None

    cv2.destroyWindow("CALIBRATE – SYN GAS Gauge")


# ─────────────────────────────────────────────────────────────
#  NEEDLE DETECTION
# ─────────────────────────────────────────────────────────────
def detect_needle(frame):
    frame  = cv2.resize(frame, (960, 540))
    output = frame.copy()

    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    mask_bright = cv2.inRange(hsv,
                              np.array([0,   0,  160], dtype=np.uint8),
                              np.array([180, 60,  255], dtype=np.uint8))

    needle_mask = mask_bright


    mask_blue_dark = cv2.inRange(hsv,
                                 np.array([90,  30,  0]),
                                 np.array([140, 255, 80]))

    needle_mask = cv2.bitwise_or(mask_bright, mask_blue_dark)

    gauge_mask = np.zeros(needle_mask.shape, dtype=np.uint8)
    cv2.circle(gauge_mask, (PIVOT_X, PIVOT_Y), 250, 255, -1)
    needle_region = cv2.bitwise_and(needle_mask, gauge_mask)

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    needle_region = cv2.morphologyEx(needle_region, cv2.MORPH_CLOSE, kernel)

    edges = cv2.Canny(needle_region, 20, 80)
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180,
                             threshold=25,
                             minLineLength=40,
                             maxLineGap=20)

    if lines is None:
        cv2.putText(output, "Needle NOT detected",
                    (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
        return output, None

    best_line = None
    best_dist = 1e9

    for line in lines:
        x1, y1, x2, y2 = line[0]
        length = np.hypot(x2 - x1, y2 - y1)
        if length < 35:
            continue
        dx, dy = x2 - x1, y2 - y1
        num  = abs(dy * PIVOT_X - dx * PIVOT_Y + x2 * y1 - y2 * x1)
        den  = np.hypot(dx, dy) + 1e-9
        dist = num / den
        if dist < best_dist:
            best_dist = dist
            best_line = line[0]

    if best_line is None or best_dist > 60:
        cv2.putText(output, "Needle NOT detected (dist too large)",
                    (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        return output, None

    x1, y1, x2, y2 = best_line

    d1 = np.hypot(x1 - PIVOT_X, y1 - PIVOT_Y)
    d2 = np.hypot(x2 - PIVOT_X, y2 - PIVOT_Y)
    tip_x, tip_y = (x1, y1) if d1 > d2 else (x2, y2)

    dx        = tip_x - PIVOT_X
    dy        = PIVOT_Y - tip_y
    angle_deg = np.degrees(np.arctan2(dy, dx)) % 360

    flow = angle_to_flow(angle_deg)

    # ── NEW: reject readings outside valid window ─────────────
    if not (FLOW_VALID_MIN <= flow <= FLOW_VALID_MAX):
        cv2.putText(output, f"REJECTED: {flow:.0f} Nm3/h (out of range)",
                    (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        return output, None

    cv2.line(output,   (PIVOT_X, PIVOT_Y), (tip_x, tip_y), (0, 255,   0), 3)
    cv2.circle(output, (PIVOT_X, PIVOT_Y), 7,  (255,   0,   0), -1)
    cv2.circle(output, (tip_x,  tip_y),   5,  (0,     0, 255), -1)

    cv2.putText(output, f"Angle   : {angle_deg:.1f} deg",
                (20,  40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 200, 255), 2)
    cv2.putText(output, f"Flow    : {flow:.1f} Nm3/h",
                (20,  80), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255,   0), 2)
    cv2.putText(output, f"Pivot   : ({PIVOT_X},{PIVOT_Y})",
                (20, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200,   0), 1)

    return output, flow


# ─────────────────────────────────────────────────────────────
#  CSV
# ─────────────────────────────────────────────────────────────
SAVE_CSV = True
CSV_PATH = r"C:\Users\hp\Desktop\image_processing\syngas_flow_log.csv"

def open_csv():
    if not SAVE_CSV:
        return None, None
    f = open(CSV_PATH, "w", newline="")
    w = csv.writer(f)
    w.writerow(["timestamp_s", "frame_no", "raw_flow_nm3h", "smoothed_flow_nm3h"])
    return f, w


# ─────────────────────────────────────────────────────────────
#  VIDEO LOOP
# ─────────────────────────────────────────────────────────────
cap = cv2.VideoCapture(VIDEO_PATH)

if not cap.isOpened():
    print(f"ERROR: Cannot open video → {VIDEO_PATH}")
    exit()

fps      = cap.get(cv2.CAP_PROP_FPS) or 25.0
frame_no = 0
csv_file, csv_writer = open_csv()

# ── NEW: rolling buffer for median smoothing ──────────────────
flow_buffer = deque(maxlen=SMOOTH_WINDOW)

print("SYN GAS Flow Meter Reader  |  Press Q to quit")

if CALIBRATE_MODE:
    ret, first_frame = cap.read()
    if ret:
    
        calibrate(first_frame)
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    print("\nCalibration done. Starting video...\n")
    print("After checking the overlay, update PIVOT_X / PIVOT_Y / ANGLE_AT_MIN / ANGLE_AT_MAX")
    print("then set CALIBRATE_MODE = False for a clean run.\n")

while True:
    ret, frame = cap.read()
    if not ret:
        print("Video ended.")
        cv2.waitKey(0)
        break

    frame_no += 1
    timestamp = frame_no / fps

    output, flow = detect_needle(frame)

    if flow is not None:
        # ── NEW: add to buffer and compute median ─────────────
        flow_buffer.append(flow)
        smoothed = round(float(np.median(flow_buffer)), 1)

        cv2.putText(output, f"Smoothed: {smoothed:.1f} Nm3/h",
                    (20, 160), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 200, 0), 2)

        print(f"[{timestamp:7.2f}s | frame {frame_no:5d}]  raw={flow:6.1f}  smooth={smoothed:6.1f}")
        if csv_writer:
            csv_writer.writerow([f"{timestamp:.3f}", frame_no, flow, smoothed])

    cv2.imshow("SYN GAS Flow Meter", output)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
if csv_file:
    csv_file.close()
    print(f"\nReadings saved to: {CSV_PATH}")
cv2.destroyAllWindows()