import cv2
import numpy as np
import csv
from collections import deque

# ─────────────────────────────────────────────────────────────
#  CONFIG
# ─────────────────────────────────────────────────────────────
VIDEO_PATH = r"C:\Users\hp\Desktop\image_processing\videos\MVI_3222.MP4"
CSV_PATH   = r"C:\Users\hp\Desktop\image_processing\syngas_flow_log.csv"
SAVE_CSV   = True

# Pivot = needle hub centre (measured from raw frame at 960x540)
PIVOT_X = 467
PIVOT_Y = 368

# Angle when needle points at scale MIN and MAX
# (CCW from +X axis, OpenCV convention)
#   48  Nm³/h → bottom-left  ≈ 200 deg
#   480 Nm³/h → top-right    ≈  75 deg
ANGLE_AT_MIN = 200.0
ANGLE_AT_MAX =  75.0

FLOW_MIN = 48.0
FLOW_MAX = 480.0

# Accept only readings in this range
FLOW_VALID_MIN = 240.0
FLOW_VALID_MAX = 290.0

SMOOTH_WINDOW  = 7
DEBUG_MASK     = True   # set False once working
CALIBRATE_MODE = True

# ─────────────────────────────────────────────────────────────
#  ANGLE → FLOW
# ─────────────────────────────────────────────────────────────
def angle_to_flow(angle_deg):
    total_sweep = (ANGLE_AT_MAX - ANGLE_AT_MIN) % 360
    relative    = (angle_deg    - ANGLE_AT_MIN) % 360
    ratio = relative / total_sweep if total_sweep != 0 else 0.0
    ratio = max(0.0, min(1.0, ratio))
    return round(FLOW_MIN + ratio * (FLOW_MAX - FLOW_MIN), 1)

# ─────────────────────────────────────────────────────────────
#  CALIBRATION
# ─────────────────────────────────────────────────────────────
_click_point = None

def _on_mouse(event, x, y, flags, param):
    global _click_point
    if event == cv2.EVENT_LBUTTONDOWN:
        _click_point = (x, y)
        print(f"  Clicked → PIVOT_X={x}, PIVOT_Y={y}")

def calibrate(frame):
    global _click_point
    frame   = cv2.resize(frame, (960, 540))
    overlay = frame.copy()
    cv2.circle(overlay, (PIVOT_X, PIVOT_Y), 8, (0, 255, 255), 2)
    cv2.putText(overlay, f"Pivot: ({PIVOT_X},{PIVOT_Y})",
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
    cv2.putText(overlay, "Click needle hub. SPACE to continue.",
                (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 0), 1)

    def ray(angle, color, label):
        rad = np.radians(angle)
        ex  = int(PIVOT_X + 180 * np.cos(rad))
        ey  = int(PIVOT_Y - 180 * np.sin(rad))
        cv2.line(overlay, (PIVOT_X, PIVOT_Y), (ex, ey), color, 2)
        cv2.putText(overlay, label, (ex, ey), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 1)

    ray(ANGLE_AT_MIN, (0, 140, 255), f"MIN {FLOW_MIN:.0f}")
    ray(ANGLE_AT_MAX, (0, 255,   0), f"MAX {FLOW_MAX:.0f}")

    cv2.namedWindow("CALIBRATE")
    cv2.setMouseCallback("CALIBRATE", _on_mouse)
    while True:
        cv2.imshow("CALIBRATE", overlay)
        key = cv2.waitKey(20) & 0xFF
        if key == ord(' '):
            break
        if _click_point:
            print(f"  → Update PIVOT_X={_click_point[0]}, PIVOT_Y={_click_point[1]}")
            _click_point = None
    cv2.destroyWindow("CALIBRATE")

# ─────────────────────────────────────────────────────────────
#  NEEDLE DETECTION
# ─────────────────────────────────────────────────────────────
def detect_needle(frame):
    frame  = cv2.resize(frame, (960, 540))
    output = frame.copy()

    # Convert to grayscale — simpler and more reliable for this gauge
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Gauge face is bright white; needle is dark black
    # Threshold: keep only dark pixels (needle + text + markings)
    _, dark_mask = cv2.threshold(gray, 80, 200, cv2.THRESH_BINARY_INV)
    # NEW - rectangular mask for square gauge
    gauge_mask = np.zeros_like(dark_mask)

# Define the square gauge face boundaries (in 960x540 frame)
# Adjust these 4 values to tightly fit your gauge face (inside the red border)
    GAUGE_X1 = 200   # left edge
    GAUGE_Y1 = 130   # top edge
    GAUGE_X2 = 730   # right edge
    GAUGE_Y2 = 510   # bottom edge

    cv2.rectangle(gauge_mask, (GAUGE_X1, GAUGE_Y1), (GAUGE_X2, GAUGE_Y2), 255, -1)

# Erode inward to exclude the red metal border
    erode_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    gauge_mask_eroded = cv2.erode(gauge_mask, erode_kernel, iterations=4)
    needle_region = cv2.bitwise_and(dark_mask, gauge_mask_eroded)

    # Morphological cleanup
    close_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    needle_region = cv2.morphologyEx(needle_region, cv2.MORPH_CLOSE, close_kernel)

    if DEBUG_MASK:
        cv2.imshow("DEBUG - needle mask (white=detected)", needle_region)
        cv2.imshow("DEBUG - raw frame", frame)

    # Hough line detection
    edges = cv2.Canny(needle_region, 30, 100)
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180,
                             threshold=30,
                             minLineLength=50,
                             maxLineGap=15)

    if lines is None:
        cv2.putText(output, "Needle NOT detected",
                    (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
        return output, None

    # Pick line closest to pivot
    best_line, best_dist = None, 1e9
    for line in lines:
        x1, y1, x2, y2 = line[0]
        if np.hypot(x2-x1, y2-y1) < 50:
            continue
        dx, dy = x2-x1, y2-y1
        dist = abs(dy*PIVOT_X - dx*PIVOT_Y + x2*y1 - y2*x1) / (np.hypot(dx,dy)+1e-9)
        if dist < best_dist:
            best_dist, best_line = dist, line[0]

    if best_line is None or best_dist > 50:
        cv2.putText(output, "Needle NOT detected (dist too large)",
                    (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        return output, None

    x1, y1, x2, y2 = best_line
    d1 = np.hypot(x1-PIVOT_X, y1-PIVOT_Y)
    d2 = np.hypot(x2-PIVOT_X, y2-PIVOT_Y)
    tip_x, tip_y = (x1, y1) if d1 > d2 else (x2, y2)

    dx        = tip_x - PIVOT_X
    dy        = PIVOT_Y - tip_y   # flip Y for OpenCV
    angle_deg = np.degrees(np.arctan2(dy, dx)) % 360
    flow      = angle_to_flow(angle_deg)

    # Always draw so you can diagnose
    cv2.line(output,   (PIVOT_X, PIVOT_Y), (tip_x, tip_y), (0, 255, 0), 3)
    cv2.circle(output, (PIVOT_X, PIVOT_Y), 8,  (255, 0,   0), -1)  # pivot = red
    cv2.circle(output, (tip_x,   tip_y),   6,  (0,   0, 255), -1)  # tip   = blue
    cv2.putText(output, f"Angle : {angle_deg:.1f} deg",
                (20, 40),  cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 200, 255), 2)
    cv2.putText(output, f"Raw   : {flow:.1f} Nm3/h",
                (20, 80),  cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255,   0), 2)

    # Reject out-of-range AFTER drawing (so you can see what's wrong)
    if not (FLOW_VALID_MIN <= flow <= FLOW_VALID_MAX):
        cv2.putText(output, f"REJECTED  valid={FLOW_VALID_MIN:.0f}-{FLOW_VALID_MAX:.0f}",
                    (20, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        return output, None

    cv2.putText(output, f"Pivot : ({PIVOT_X},{PIVOT_Y})",
                (20, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 0), 1)
    return output, flow

# ─────────────────────────────────────────────────────────────
#  VIDEO LOOP
# ─────────────────────────────────────────────────────────────
cap = cv2.VideoCapture(VIDEO_PATH)
if not cap.isOpened():
    print(f"ERROR: Cannot open → {VIDEO_PATH}")
    exit()

fps      = cap.get(cv2.CAP_PROP_FPS) or 25.0
frame_no = 0
flow_buffer = deque(maxlen=SMOOTH_WINDOW)

csv_file = csv_writer = None
if SAVE_CSV:
    csv_file   = open(CSV_PATH, "w", newline="")
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow(["timestamp_s", "frame_no", "raw_flow_nm3h", "smoothed_flow_nm3h"])

if CALIBRATE_MODE:
    ret, first_frame = cap.read()
    if ret:
        calibrate(first_frame)
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

print("Running... Press Q to quit")

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
        flow_buffer.append(flow)
        smoothed = round(float(np.median(flow_buffer)), 1)
        cv2.putText(output, f"Smoothed: {smoothed:.1f} Nm3/h",
                    (20, 160), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 200, 0), 2)
        print(f"[{timestamp:7.2f}s | f{frame_no:5d}]  raw={flow:6.1f}  smooth={smoothed:6.1f}")
        if csv_writer:
            csv_writer.writerow([f"{timestamp:.3f}", frame_no, flow, smoothed])

    cv2.imshow("SYN GAS Flow Meter", output)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
if csv_file:
    csv_file.close()
    print(f"\nSaved → {CSV_PATH}")
cv2.destroyAllWindows()
