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

# ── PIVOT (needle hub) ────────────────────────────────────────
# Looking at the gauge image: the needle appears to pivot near
# the bottom-right of the scale arc, inside the gauge face.
# At 960x540 resolution these are starting estimates — use
# CALIBRATE_MODE=True and click the exact hub to refine them.
PIVOT_X = 522
PIVOT_Y = 415

# ── ANGLE CALIBRATION ────────────────────────────────────────
# OpenCV convention: angle measured CCW from +X axis.
# From the gauge face:
#   48  Nm³/h → needle points toward lower-left  ≈ 215°
#   480 Nm³/h → needle points toward upper-right ≈  70°
# These are approximations. Use CALIBRATE_MODE to verify:
#   1. Pause on a frame where needle is at MIN scale mark
#   2. Note the "Angle" readout → set ANGLE_AT_MIN to that value
#   3. Repeat for MAX
ANGLE_AT_MIN = 215.0   # degrees when needle = 48  Nm³/h
ANGLE_AT_MAX =  70.0   # degrees when needle = 480 Nm³/h

FLOW_MIN = 48.0
FLOW_MAX = 480.0

# Accept readings only in this plausible operating range
FLOW_VALID_MIN =  40.0
FLOW_VALID_MAX = 490.0

SMOOTH_WINDOW  = 7
DEBUG_MASK     = True   # set False once working
CALIBRATE_MODE = True

# ── GAUGE FACE ROI (960×540) ─────────────────────────────────
# Tight rectangle that contains only the white gauge face,
# EXCLUDING the red metal border.
GAUGE_X1, GAUGE_Y1 = 215, 140
GAUGE_X2, GAUGE_Y2 = 740, 515

# ─────────────────────────────────────────────────────────────
#  ANGLE → FLOW  (handles wrap-around correctly)
# ─────────────────────────────────────────────────────────────
def angle_to_flow(angle_deg):
    """
    Maps a CCW angle (from +X) to a flow reading.
    Needle sweeps CCW from ANGLE_AT_MIN down to ANGLE_AT_MAX
    (i.e. from 215° down to 70°, crossing 180°, 90°).
    """
    # Normalise into [0, 360)
    a   = angle_deg % 360
    lo  = ANGLE_AT_MIN % 360  # e.g. 215
    hi  = ANGLE_AT_MAX % 360  # e.g.  70

    # Sweep goes from lo DOWN to hi (CCW sweep of ~145°)
    # Convert so that "distance swept CCW from lo" is our ratio
    # CCW distance from lo to a:
    dist_a  = (lo - a)  % 360
    dist_hi = (lo - hi) % 360   # total sweep

    ratio = dist_a / dist_hi if dist_hi != 0 else 0.0
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

def draw_ray(overlay, angle, color, label):
    rad = np.radians(angle)
    ex  = int(PIVOT_X + 200 * np.cos(rad))
    ey  = int(PIVOT_Y - 200 * np.sin(rad))   # flip Y for screen
    cv2.line(overlay, (PIVOT_X, PIVOT_Y), (ex, ey), color, 2)
    cv2.putText(overlay, label, (ex + 4, ey), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

def calibrate(frame):
    global _click_point
    frame   = cv2.resize(frame, (960, 540))
    overlay = frame.copy()
    cv2.circle(overlay, (PIVOT_X, PIVOT_Y), 10, (0, 255, 255), 2)
    cv2.putText(overlay, f"Pivot: ({PIVOT_X},{PIVOT_Y})",
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
    cv2.putText(overlay, "Click needle hub centre. SPACE to continue.",
                (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 0), 1)
    cv2.putText(overlay, "RED ray = MIN (48), GREEN ray = MAX (480)",
                (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

    draw_ray(overlay, ANGLE_AT_MIN, (0, 0, 255),   f"MIN {FLOW_MIN:.0f}")
    draw_ray(overlay, ANGLE_AT_MAX, (0, 255, 0),   f"MAX {FLOW_MAX:.0f}")
    draw_ray(overlay, 142,          (0, 200, 255), "~142deg")   # where needle was in screenshot

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

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # ── Step 1: isolate dark features on the white gauge face ──
    # Use OTSU for automatic threshold adaptation to lighting changes
    _, dark_mask = cv2.threshold(gray, 0, 255,
                                 cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)

    # ── Step 2: ROI mask — gauge face only ──
    gauge_mask = np.zeros_like(dark_mask)
    cv2.rectangle(gauge_mask,
                  (GAUGE_X1, GAUGE_Y1), (GAUGE_X2, GAUGE_Y2), 255, -1)

    # Erode inward to strip the red metal border
    erode_k = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    gauge_mask = cv2.erode(gauge_mask, erode_k, iterations=4)

    needle_region = cv2.bitwise_and(dark_mask, gauge_mask)

    # ── Step 3: suppress scale tick marks & text ──
    # The needle is a single thick line; ticks are thin and short.
    # Morphological opening with a large kernel removes thin features.
    open_k = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    needle_region = cv2.morphologyEx(needle_region, cv2.MORPH_OPEN,  open_k, iterations=1)
    close_k = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    needle_region = cv2.morphologyEx(needle_region, cv2.MORPH_CLOSE, close_k, iterations=2)

    if DEBUG_MASK:
        cv2.imshow("DEBUG - needle mask", needle_region)
        cv2.imshow("DEBUG - raw frame",   frame)

    # ── Step 4: Hough lines ──
    edges = cv2.Canny(needle_region, 20, 80)
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180,
                             threshold=25,
                             minLineLength=60,
                             maxLineGap=20)

    if lines is None:
        cv2.putText(output, "Needle NOT detected",
                    (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
        return output, None

    # ── Step 5: pick the line that passes THROUGH the pivot ──
    # Score = distance from pivot to the infinite line (lower = better)
    best_line, best_dist = None, 1e9
    for seg in lines:
        x1, y1, x2, y2 = seg[0]
        length = np.hypot(x2 - x1, y2 - y1)
        if length < 55:
            continue
        dx, dy = x2 - x1, y2 - y1
        dist = abs(dy * PIVOT_X - dx * PIVOT_Y + x2 * y1 - y2 * x1) / (length + 1e-9)
        if dist < best_dist:
            best_dist = dist
            best_line = seg[0]

    MAX_PIVOT_DIST = 100   # pixels — raise if still not detecting
    if best_line is None or best_dist > MAX_PIVOT_DIST:
        cv2.putText(output, f"No line near pivot (best dist={best_dist:.1f}px)",
                    (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 255), 2)
        return output, None

    x1, y1, x2, y2 = best_line

    # ── Step 6: determine TIP (far end) vs TAIL ──
    # The needle tip is the endpoint FARTHER from the pivot.
    # BUT on this gauge the needle has a short counterweight tail;
    # the tip points toward the scale marks.
    # We pick the farther endpoint and then verify it is NOT pointing
    # directly back toward the gauge centre-right area (the pivot side).
    d1 = np.hypot(x1 - PIVOT_X, y1 - PIVOT_Y)
    d2 = np.hypot(x2 - PIVOT_X, y2 - PIVOT_Y)
    tip_x, tip_y = (x1, y1) if d1 > d2 else (x2, y2)

    # Direction vector from pivot → tip
    dx = tip_x - PIVOT_X
    dy = PIVOT_Y - tip_y   # flip Y (screen → math convention)
    angle_deg = np.degrees(np.arctan2(dy, dx)) % 360

    # Sanity: the tip should be in the upper-left quadrant of the gauge
    # (scale arc region). If angle is in the "impossible" range (roughly
    # 300°–30°, i.e. pointing right/down-right) the tip/tail may be swapped.
    SWAP_IF_OUT_OF_RANGE = True
    if SWAP_IF_OUT_OF_RANGE:
        # Gauge arc spans roughly 60°–225° (tip should point into this arc)
        arc_lo, arc_hi = 55.0, 230.0
        in_arc = (arc_lo <= angle_deg <= arc_hi)
        if not in_arc:
            # Flip tip/tail
            tip_x, tip_y = (x2, y2) if d1 > d2 else (x1, y1)
            dx = tip_x - PIVOT_X
            dy = PIVOT_Y - tip_y
            angle_deg = np.degrees(np.arctan2(dy, dx)) % 360

    flow = angle_to_flow(angle_deg)

    # ── Draw diagnostics ──
    cv2.line(output, (PIVOT_X, PIVOT_Y), (tip_x, tip_y), (0, 255, 0), 3)
    cv2.circle(output, (PIVOT_X, PIVOT_Y), 8,  (255, 0,   0), -1)  # pivot = red
    cv2.circle(output, (tip_x,   tip_y),   6,  (0,   0, 255), -1)  # tip   = blue
    cv2.putText(output, f"Angle  : {angle_deg:.1f} deg",
                (20, 40),  cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 200, 255), 2)
    cv2.putText(output, f"Raw    : {flow:.1f} Nm3/h",
                (20, 80),  cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255,   0), 2)
    cv2.putText(output, f"Pivot  : ({PIVOT_X},{PIVOT_Y})",
                (20, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 0), 1)
    cv2.putText(output, f"PivDist: {best_dist:.1f}px",
                (20, 145), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 0), 1)

    if not (FLOW_VALID_MIN <= flow <= FLOW_VALID_MAX):
        cv2.putText(output, f"REJECTED  valid={FLOW_VALID_MIN:.0f}-{FLOW_VALID_MAX:.0f}",
                    (20, 175), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        return output, None

    return output, flow

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

print("Running... Press Q to quit | S to skip frame | P to pause")
paused = False

while True:
    if not paused:
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
                        (20, 210), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 200, 0), 2)
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
            output, flow = detect_needle(frame)
            cv2.imshow("SYN GAS Flow Meter", output)

cap.release()
if csv_file:
    csv_file.close()
    print(f"\nSaved → {CSV_PATH}")
cv2.destroyAllWindows()