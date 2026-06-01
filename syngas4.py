import cv2
import numpy as np
import csv
from collections import deque

# ─────────────────────────────────────────────────────────────
#  CONFIG  — edit these constants to match your gauge
# ─────────────────────────────────────────────────────────────
VIDEO_PATH = r"C:\Users\hp\Desktop\image_processing\videos\MVI_NEW_3224 - Trim.mp4"
CSV_PATH   = r"C:\Users\hp\Desktop\image_processing\syngas_flow_log.csv"
SAVE_CSV   = True
# ── MOBILE CAMERA CONFIG ─────────────────────────────────────
# Option A: Wi-Fi stream (IP Webcam app on Android)
#   1. Install "IP Webcam" from Play Store
#   2. Open app → tap "Start server"
#   3. Note the URL shown (e.g. http://192.168.1.5:8080)
#   4. Set USE_MOBILE_CAMERA = True and fill MOBILE_STREAM_URL
#
# Option B: USB webcam index (DroidCam or similar)
#   Set MOBILE_CAMERA_INDEX to 1, 2, etc. (0 = built-in webcam)
# ─────────────────────────────────────────────────────────────
USE_MOBILE_CAMERA   = True                          # ← set True to activate
MOBILE_STREAM_URL   = "http://192.168.29.103:8080/video"  # ← your phone's IP:port
MOBILE_CAMERA_INDEX = 1                              # ← used only if URL fails
# ── PIVOT: needle hub centre (960×540 frame) ─────────────────
# From your screenshot the blue dot was at ~(555,430) — that IS
# the real hinge.  The red dot was at ~(370,230) — that was the tip.
# So in previous code tip/tail were FULLY SWAPPED.
PIVOT_X = 642
PIVOT_Y = 425

# ── ANGLE CALIBRATION (CCW from +X, OpenCV math convention) ──
# Gauge face (looking at the screenshot):
#   48  Nm³/h  → needle tip points toward lower-left  ≈ 210°
#   480 Nm³/h  → needle tip points toward upper-right ≈  75°
# Run CALIBRATE_MODE, press P to pause on known-value frames,
# read the "Angle" overlay, then update these two numbers.
ANGLE_AT_MIN = 200.0   # angle when needle = FLOW_MIN
ANGLE_AT_MAX =  80.0   # angle when needle = FLOW_MAX

FLOW_MIN = 48.0
FLOW_MAX = 480.0

FLOW_VALID_MIN =  48.0
FLOW_VALID_MAX = 380.0

SMOOTH_WINDOW  = 7
DEBUG_MASK     = True
CALIBRATE_MODE = True

# ── GAUGE ROI (960×540) ───────────────────────────────────────
GAUGE_X1, GAUGE_Y1 = 215, 140
GAUGE_X2, GAUGE_Y2 = 740, 515

# ─────────────────────────────────────────────────────────────
#  ANGLE → FLOW
# ─────────────────────────────────────────────────────────────
def angle_to_flow(angle_deg):
    """
    Needle sweeps CCW from ANGLE_AT_MIN (low flow) to ANGLE_AT_MAX (high flow).
    e.g. 210° → 48 Nm³/h,  75° → 480 Nm³/h
    CCW means angle DECREASES (210 → 150 → 90 → 75).
    ratio = how far from MIN toward MAX.
    """
    a   = angle_deg % 360
    lo  = ANGLE_AT_MIN % 360   # e.g. 210
    hi  = ANGLE_AT_MAX % 360   # e.g.  75

    # CCW distance from lo to a  (positive = went CCW)
    dist_a  = (lo - a)  % 360
    # Total CCW sweep from lo to hi
    dist_hi = (lo - hi) % 360

    ratio = dist_a / dist_hi if dist_hi > 0 else 0.0
    ratio = max(0.0, min(1.0, ratio))
    return round(FLOW_MIN + ratio * (FLOW_MAX - FLOW_MIN), 1)

# ─────────────────────────────────────────────────────────────
#  CALIBRATION HELPER
# ─────────────────────────────────────────────────────────────
_click_point = None

def _on_mouse(event, x, y, flags, param):
    global _click_point
    if event == cv2.EVENT_LBUTTONDOWN:
        _click_point = (x, y)
        print(f"  Clicked → PIVOT_X={x}, PIVOT_Y={y}")

def draw_ray(img, angle_deg, color, label):
    rad = np.radians(angle_deg)
    ex  = int(PIVOT_X + 190 * np.cos(rad))
    ey  = int(PIVOT_Y - 190 * np.sin(rad))
    cv2.line(img, (PIVOT_X, PIVOT_Y), (ex, ey), color, 2)
    cv2.putText(img, label, (ex + 4, ey + 4),
                cv2.FONT_HERSHEY_SIMPLEX, 0.48, color, 1)

def calibrate(frame):
    global _click_point
    frame   = cv2.resize(frame, (960, 540))
    overlay = frame.copy()

    # Draw pivot
    cv2.circle(overlay, (PIVOT_X, PIVOT_Y), 10, (0, 255, 255), 2)
    cv2.putText(overlay, f"Pivot ({PIVOT_X},{PIVOT_Y}) — click to move",
                (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (0, 255, 255), 2)
    cv2.putText(overlay, "SPACE to continue",
                (10, 56), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 0), 1)

    # Show angle reference rays
    draw_ray(overlay, ANGLE_AT_MIN, (0,  80, 255), f"MIN {FLOW_MIN:.0f} Nm3/h")
    draw_ray(overlay, ANGLE_AT_MAX, (0, 255,  80), f"MAX {FLOW_MAX:.0f} Nm3/h")
    draw_ray(overlay, 145,          (200,200,  0), "mid ~250")

    cv2.namedWindow("CALIBRATE")
    cv2.setMouseCallback("CALIBRATE", _on_mouse)
    while True:
        cv2.imshow("CALIBRATE", overlay)
        key = cv2.waitKey(20) & 0xFF
        if key == ord(' '):
            break
        if _click_point:
            print(f"  → Set PIVOT_X={_click_point[0]}, PIVOT_Y={_click_point[1]}")
            _click_point = None
    cv2.destroyWindow("CALIBRATE")

# ─────────────────────────────────────────────────────────────
#  NEEDLE DETECTION  (core fix: correct tip selection)
# ─────────────────────────────────────────────────────────────
def detect_needle(frame):
    frame  = cv2.resize(frame, (960, 540))
    output = frame.copy()

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Otsu threshold — auto-adapts to lighting
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    dark_mask = cv2.adaptiveThreshold(
    blur, 255,
    cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
    cv2.THRESH_BINARY_INV,
    blockSize=31,
    C=8  
    )       

    # ROI: gauge face only
    gauge_mask = np.zeros_like(dark_mask)
    cv2.rectangle(gauge_mask,
                  (GAUGE_X1, GAUGE_Y1), (GAUGE_X2, GAUGE_Y2), 255, -1)
    erode_k = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    gauge_mask = cv2.erode(gauge_mask, erode_k, iterations=4)

    needle_region = cv2.bitwise_and(dark_mask, gauge_mask)
    needle_region[PIVOT_Y:, PIVOT_X:] = 0 
    # Clean up tick marks and text (short, thin features)
    close_k = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    needle_region = cv2.morphologyEx(needle_region, cv2.MORPH_CLOSE, close_k, iterations=2)

    if DEBUG_MASK:
        cv2.imshow("DEBUG mask", needle_region)

    # Hough lines
    edges = cv2.Canny(needle_region, 20, 80)
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180,
                             threshold=25,
                             minLineLength=60,
                             maxLineGap=20)

    if lines is None:
        _put(output, "Needle NOT detected", (0, 0, 255), 40)
        return output, None

    # ── Pick the Hough segment closest (perpendicularly) to pivot ──
    best_seg, best_dist = None, 1e9
    for seg in lines:
        x1, y1, x2, y2 = seg[0]
        length = np.hypot(x2 - x1, y2 - y1)
        if length < 55:
            continue
        dx, dy = x2 - x1, y2 - y1
        # Perpendicular distance from PIVOT to this infinite line
        dist = abs(dy * PIVOT_X - dx * PIVOT_Y + x2 * y1 - y2 * x1) / (length + 1e-9)
        if dist < best_dist:
            best_dist = dist
            best_seg  = seg[0]

    if best_seg is None or best_dist > 45:
        _put(output, f"No line near pivot (best={best_dist:.0f}px)", (0, 0, 255), 40)
        return output, None

    x1, y1, x2, y2 = best_seg

    # ── KEY FIX: identify tip correctly ──────────────────────────
    # The needle tip points toward the scale arc (upper-left region).
    # The counterweight / tail points away from it (lower-right).
    #
    # Strategy: the endpoint whose direction FROM THE PIVOT falls
    # inside the valid gauge arc [55°, 225°] is the tip.
    # If both or neither qualify, fall back to "farther endpoint".

    def endpoint_angle(ex, ey):
        dx = ex - PIVOT_X
        dy = PIVOT_Y - ey          # flip Y: screen → math
        return np.degrees(np.arctan2(dy, dx)) % 360

    ARC_LO, ARC_HI = 70.0, 230.0  # valid tip angle range

    candidates = []
    for seg in lines:
        x1, y1, x2, y2 = seg[0]
        length = np.hypot(x2 - x1, y2 - y1)
        if length < 55:
            continue
        dx, dy = x2 - x1, y2 - y1
        perp = abs(dy * PIVOT_X - dx * PIVOT_Y + x2*y1 - y2*x1) / (length + 1e-9)
        if perp > 50:
            continue

        a1 = endpoint_angle(x1, y1)
        a2 = endpoint_angle(x2, y2)
        d1 = np.hypot(x1 - PIVOT_X, y1 - PIVOT_Y)
        d2 = np.hypot(x2 - PIVOT_X, y2 - PIVOT_Y)

        # Tip = farther endpoint; tail = nearer endpoint
        if d1 >= d2:
            tip_x, tip_y, tip_a = x1, y1, a1
        else:
            tip_x, tip_y, tip_a = x2, y2, a2

        in_arc = ARC_LO <= tip_a <= ARC_HI
        # Score: in_arc is worth a lot; shorter perp distance is better
        score = (0 if in_arc else 10000) + perp
        candidates.append((score, perp, tip_x, tip_y, tip_a, seg[0]))

    if not candidates:
        _put(output, "No candidates near pivot", (0, 0, 255), 40)
        return output, None

    candidates.sort(key=lambda c: c[0])
    _, best_dist, tip_x, tip_y, angle_deg, best_seg = candidates[0]
    flow = angle_to_flow(angle_deg)

    # ── Draw diagnostics ─────────────────────────────────────────
    cv2.line(output, (PIVOT_X, PIVOT_Y), (tip_x, tip_y), (0, 255, 0), 3)
    cv2.circle(output, (PIVOT_X, PIVOT_Y), 9,  (0,   0, 255), -1)  # pivot = RED  dot
    cv2.circle(output, (tip_x,   tip_y),   6,  (255, 0,   0), -1)  # tip   = BLUE dot

    _put(output, f"Angle  : {angle_deg:.1f} deg",  (0, 200, 255), 40)
    _put(output, f"Raw    : {flow:.1f} Nm3/h",      (0, 255,   0), 80)
    _put(output, f"Pivot  : ({PIVOT_X},{PIVOT_Y})", (200, 200, 0), 120, small=True)
    _put(output, f"PivDist: {best_dist:.1f}px",     (200, 200, 0), 145, small=True)

    if not (FLOW_VALID_MIN <= flow <= FLOW_VALID_MAX):
        _put(output, f"REJECTED  valid={FLOW_VALID_MIN:.0f}-{FLOW_VALID_MAX:.0f}",
             (0, 0, 255), 175)
        return output, None

    return output, flow


def _put(img, text, color, y, small=False):
    scale = 0.62 if small else 0.82
    cv2.putText(img, text, (20, y),
                cv2.FONT_HERSHEY_SIMPLEX, scale, color, 2)

# ─────────────────────────────────────────────────────────────
#  VIDEO LOOP
# ─────────────────────────────────────────────────────────────
cap = cv2.VideoCapture(VIDEO_PATH)
if not cap.isOpened():
    print(f"ERROR: Cannot open → {VIDEO_PATH}")
    exit()

fps         = cap.get(cv2.CAP_PROP_FPS) or 25.0
frame_no    = 0
flow_buffer = deque(maxlen=SMOOTH_WINDOW)

csv_file = csv_writer = None
if SAVE_CSV:
    csv_file   = open(CSV_PATH, "w", newline="")
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow(["timestamp_s", "frame_no",
                         "raw_flow_nm3h", "smoothed_flow_nm3h"])

if CALIBRATE_MODE:
    ret, first_frame = cap.read()
    if ret:
        calibrate(first_frame)
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

print("Running...  Q=quit  P=pause  S=step(while paused)")
paused = False

while True:
    if not paused:
        ret, frame = cap.read()
        if not ret:
            print("Video ended.")
            cv2.waitKey(0)
            break

        frame_no  += 1
        timestamp  = frame_no / fps
        output, flow = detect_needle(frame)

        if flow is not None:
            flow_buffer.append(flow)
            smoothed = round(float(np.median(flow_buffer)), 1)
            _put(output, f"Smoothed: {smoothed:.1f} Nm3/h", (255, 200, 0), 210)
            print(f"[{timestamp:7.2f}s | f{frame_no:5d}]  "
                  f"raw={flow:6.1f}  smooth={smoothed:6.1f}")
            if csv_writer:
                csv_writer.writerow([f"{timestamp:.3f}", frame_no, flow, smoothed])

        cv2.imshow("SYN GAS Flow Meter", output)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key == ord('p'):
        paused = not paused
        print("PAUSED" if paused else "RESUMED")
    elif key == ord('s') and paused:
        ret, frame = cap.read()
        if ret:
            frame_no += 1
            output, _ = detect_needle(frame)
            cv2.imshow("SYN GAS Flow Meter", output)

cap.release()
if csv_file:
    csv_file.close()
    print(f"\nSaved → {CSV_PATH}")
cv2.destroyAllWindows()