import cv2
import numpy as np
from collections import deque

# ─────────────────────────────────────────────────────────────
#  CONFIG — tweak these first
# ─────────────────────────────────────────────────────────────
VIDEO_PATH = r"C:\Users\hp\Desktop\image_processing\videos\MVI_3220 - Trim.mp4"

# ── Pivot (needle hub center) in the 960×540 resized frame ───
# Run with CALIBRATE_MODE = True first to click the exact center
PIVOT_X = 507
PIVOT_Y = 292

# ── Gauge face radius in the 960×540 frame ───────────────────
# Should reach the inner edge of the tick marks, NOT the rim
GAUGE_RADIUS = 130

# ── Angle calibration ─────────────────────────────────────────
#   0 psi  → needle points at 226° (CCW from +X, standard math)
# 150 psi  → 226 − 270 = −44 = 316°   (clockwise sweep)
ANGLE_0_PSI = 220.0
TOTAL_SWEEP = 270.0
PSI_MIN     = 0
PSI_MAX     = 150

# ── Set True once to click pivot + gauge edge interactively ───
CALIBRATE_MODE = False

# ── Smoothing window (frames) ─────────────────────────────────
SMOOTH_WINDOW = 7
angle_buffer  = deque(maxlen=SMOOTH_WINDOW)

# ─────────────────────────────────────────────────────────────
#  CALIBRATION  (one-time, sets PIVOT_X/Y and GAUGE_RADIUS)
# ─────────────────────────────────────────────────────────────
_cal_clicks = []

def _mouse_cb(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        _cal_clicks.append((x, y))
        print(f"  Click {len(_cal_clicks)}: ({x}, {y})")

def run_calibration(cap):
    print("\n=== CALIBRATION MODE ===")
    print("Click 1 → needle pivot (hub center)")
    print("Click 2 → any point on the inner tick-mark ring")
    print("Press any key after both clicks.\n")

    ret, frame = cap.read()
    if not ret:
        print("Cannot read frame for calibration.")
        return None, None

    frame = cv2.resize(frame, (960, 540))
    cv2.namedWindow("Calibrate")
    cv2.setMouseCallback("Calibrate", _mouse_cb)

    clone = frame.copy()
    while True:
        disp = clone.copy()
        for i, (cx, cy) in enumerate(_cal_clicks):
            color = (0, 255, 0) if i == 0 else (0, 0, 255)
            cv2.circle(disp, (cx, cy), 6, color, -1)
            cv2.putText(disp, ["Pivot", "Rim"][i], (cx+8, cy-8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        if len(_cal_clicks) == 2:
            px, py = _cal_clicks[0]
            rx, ry = _cal_clicks[1]
            r = int(np.hypot(rx-px, ry-py))
            cv2.circle(disp, (px, py), r, (255, 200, 0), 1)
            cv2.putText(disp, f"radius={r}", (px+r+5, py),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,200,0), 2)
        cv2.imshow("Calibrate", disp)
        if cv2.waitKey(30) & 0xFF != 255:
            break

    cv2.destroyWindow("Calibrate")
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    if len(_cal_clicks) >= 2:
        px, py = _cal_clicks[0]
        rx, ry = _cal_clicks[1]
        r = int(np.hypot(rx-px, ry-py))
        print(f"\n>>> Copy these into CONFIG:\n"
              f"    PIVOT_X = {px}\n"
              f"    PIVOT_Y = {py}\n"
              f"    GAUGE_RADIUS = {r}\n"
              f"    CALIBRATE_MODE = False\n")
        return (px, py), r
    return None, None

# ─────────────────────────────────────────────────────────────
#  HELPERS
# ─────────────────────────────────────────────────────────────
def angle_to_psi(angle_deg):
    ratio = (ANGLE_0_PSI - angle_deg) / TOTAL_SWEEP
    ratio = max(0.0, min(1.0, ratio))
    psi   = round(PSI_MIN + ratio * (PSI_MAX - PSI_MIN), 1)
    return psi, round(psi * 0.0703, 2)


def smooth_angle(new_angle):
    angle_buffer.append(new_angle)
    if len(angle_buffer) < 2:
        return new_angle
    ref = angle_buffer[-1]
    unwrapped = [ref + ((a - ref + 180) % 360 - 180) for a in angle_buffer]
    return float(np.median(unwrapped)) % 360


def build_mask(shape, pivot, radius):
    """
    Annular mask tuned to exclude:
      - Outer rim / tick marks  (outer 20 px of gauge radius)
      - Inner hub               (inner 25 px around pivot)
      - Bottom stem             (anything below pivot + radius*0.55)
        → This is the critical fix: the mounting pipe is directly below
    """
    px, py = pivot
    mask = np.zeros(shape[:2], dtype=np.uint8)

    # 1. Full inner circle (no tick-mark ring)
    cv2.circle(mask, (px, py), radius - 20, 255, -1)

    # 2. Black out hub
    cv2.circle(mask, (px, py), 25, 0, -1)

    # 3. BLACK OUT the bottom half below pivot+offset
    #    Cuts off the mounting stem that confused PCA
    cutoff_y = py + int(radius * 0.5)   # 50% below pivot
    if cutoff_y < shape[0]:
        mask[cutoff_y:, :] = 0

    return mask


def extract_needle_mask(frame, pivot, radius):
    gray   = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Adaptive threshold — robust to uneven lighting
    thresh = cv2.adaptiveThreshold(
        gray, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        blockSize=21, C=7
    )

    ann  = build_mask(frame.shape, pivot, radius)
    roi  = cv2.bitwise_and(thresh, ann)

    # Morphology: remove speckle, connect needle body
    k3 = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    roi = cv2.morphologyEx(roi, cv2.MORPH_OPEN,  k3, iterations=1)
    roi = cv2.morphologyEx(roi, cv2.MORPH_CLOSE, k3, iterations=2)
    return roi


def find_longest_dark_blob(mask, pivot):
    """
    Find the largest connected component that passes closest to the pivot.
    Returns the component mask or None.
    """
    n, labels, stats, centroids = cv2.connectedComponentsWithStats(mask)
    if n < 2:
        return None

    px, py = pivot
    best_label = -1
    best_score = -1

    for lbl in range(1, n):
        area = stats[lbl, cv2.CC_STAT_AREA]
        if area < 60:
            continue
        # Component bounding box
        lx = stats[lbl, cv2.CC_STAT_LEFT]
        ly = stats[lbl, cv2.CC_STAT_TOP]
        lw = stats[lbl, cv2.CC_STAT_WIDTH]
        lh = stats[lbl, cv2.CC_STAT_HEIGHT]
        # Length of the component (max dimension)
        length = max(lw, lh)
        cx, cy = centroids[lbl]
        # Distance of centroid from pivot (needle centroid should be near pivot)
        dist_from_pivot = np.hypot(cx - px, cy - py)
        # Score = length / (1 + dist_from_pivot)  — long AND near pivot wins
        score = length / (1.0 + dist_from_pivot * 0.5)
        if score > best_score:
            best_score = score
            best_label = lbl

    if best_label < 0:
        return None

    comp_mask = (labels == best_label).astype(np.uint8) * 255
    return comp_mask


def pca_angle_from_mask(comp_mask):
    """PCA on component pixels → principal axis angle (ambiguous ±180°)."""
    pts = np.column_stack(np.where(comp_mask > 0))  # (row=y, col=x)
    if len(pts) < 40:
        return None, None, None
    xy   = pts[:, ::-1].astype(np.float32)
    mean = xy.mean(axis=0)
    cov  = np.cov(xy.T)
    _, vecs = np.linalg.eigh(cov)
    principal = vecs[:, -1]
    angle = np.degrees(np.arctan2(-principal[1], principal[0])) % 360
    return angle, float(mean[0]), float(mean[1])


def disambiguate_with_pivot(angle, mean_x, mean_y, pivot):
    """
    The needle tip is the end FARTHER from the pivot.
    The tail/hub end is closer to the pivot.
    Use the vector from pivot→centroid to pick the right half.
    """
    px, py = pivot
    # Vector from pivot to centroid
    vec_to_centroid = np.array([mean_x - px, -(mean_y - py)])   # flip y

    # The two candidate directions
    rad_a = np.radians(angle)
    rad_b = np.radians((angle + 180) % 360)
    dir_a = np.array([np.cos(rad_a),  np.sin(rad_a)])
    dir_b = np.array([np.cos(rad_b),  np.sin(rad_b)])

    # The correct tip direction should be on the SAME side as centroid→away-from-pivot
    dot_a = np.dot(vec_to_centroid, dir_a)
    dot_b = np.dot(vec_to_centroid, dir_b)

    return angle if dot_a >= dot_b else (angle + 180) % 360


def walk_to_tip(tip_angle, mean_x, mean_y, comp_mask):
    """Walk from centroid in tip direction to find the farthest white pixel."""
    rad = np.radians(tip_angle)
    dx, dy_img = np.cos(rad), -np.sin(rad)   # dy_img: flip for OpenCV
    h, w = comp_mask.shape
    tip_x, tip_y = mean_x, mean_y
    for i in range(1, 350):
        x = int(round(mean_x + i * dx))
        y = int(round(mean_y + i * dy_img))
        if not (0 <= x < w and 0 <= y < h):
            break
        if comp_mask[y, x] > 0:
            tip_x, tip_y = x, y
    return int(tip_x), int(tip_y)


# ─────────────────────────────────────────────────────────────
#  MAIN DETECTION
# ─────────────────────────────────────────────────────────────
def detect_needle(frame, pivot, radius):
    frame  = cv2.resize(frame, (960, 540))
    output = frame.copy()
    px, py = pivot

    needle_mask = extract_needle_mask(frame, pivot, radius)
    comp_mask   = find_longest_dark_blob(needle_mask, pivot)

    if comp_mask is None:
        cv2.putText(output, "Needle NOT detected",
                    (20, 45), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
        return output, None, None, needle_mask

    pca_angle, mean_x, mean_y = pca_angle_from_mask(comp_mask)
    if pca_angle is None:
        cv2.putText(output, "PCA failed",
                    (20, 45), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 165, 255), 2)
        return output, None, None, needle_mask

    # Disambiguate using pivot geometry
    tip_angle = disambiguate_with_pivot(pca_angle, mean_x, mean_y, pivot)

    # Walk to actual tip pixel
    tip_x, tip_y = walk_to_tip(tip_angle, mean_x, mean_y, comp_mask)

    # Final angle: pivot → tip  (most accurate)
    dx  = tip_x - px
    dy  = py - tip_y                    # flip Y for math convention
    raw_angle = np.degrees(np.arctan2(dy, dx)) % 360
    smoothed  = smooth_angle(raw_angle)
    psi, kgcm2 = angle_to_psi(smoothed)

    # ── Draw ──────────────────────────────────────────────────
    # Faint needle-mask overlay (blue tint)
    overlay = output.copy()
    overlay[comp_mask > 0] = [255, 180, 50]
    output = cv2.addWeighted(output, 0.75, overlay, 0.25, 0)

    cv2.line(output,   (px, py),      (tip_x, tip_y), (0, 255, 0),  3)
    cv2.circle(output, (px, py),       7,              (255, 0,   0), -1)
    cv2.circle(output, (tip_x, tip_y), 5,              (0,   0, 255), -1)
    cv2.circle(output, (px, py), radius - 20, (0, 200, 200), 1)  # mask boundary

    cv2.putText(output, f"Angle (raw)    : {raw_angle:.1f} deg",
                (20, 40),  cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 200, 255), 2)
    cv2.putText(output, f"Angle (smooth) : {smoothed:.1f} deg",
                (20, 75),  cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 200, 255), 2)
    cv2.putText(output, f"PSI            : {psi} psi",
                (20, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255,   0), 2)
    cv2.putText(output, f"kg/cm2         : {kgcm2}",
                (20, 145), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 255), 2)

    return output, psi, kgcm2, needle_mask


# ─────────────────────────────────────────────────────────────
#  VIDEO LOOP
# ─────────────────────────────────────────────────────────────
cap = cv2.VideoCapture(VIDEO_PATH)
if not cap.isOpened():
    print(f"ERROR: Cannot open video → {VIDEO_PATH}")
    exit()

pivot  = (PIVOT_X, PIVOT_Y)
radius = GAUGE_RADIUS

if CALIBRATE_MODE:
    cal_pivot, cal_radius = run_calibration(cap)
    if cal_pivot:
        pivot  = cal_pivot
        radius = cal_radius

print("Pressure Gauge v4  |  Q=quit  D=debug mask  S=snapshot  C=recalibrate")
show_debug = False

while True:
    ret, frame = cap.read()
    if not ret:
        print("Video ended.")
        cv2.waitKey(0)
        break

    output, psi, kgcm2, raw_mask = detect_needle(frame, pivot, radius)

    if show_debug:
        frame_r = cv2.resize(frame, (960, 540))
        debug   = cv2.cvtColor(raw_mask, cv2.COLOR_GRAY2BGR)
        # Show cutoff line on debug
        cutoff_y = pivot[1] + int(radius * 0.5)
        cv2.line(debug, (0, cutoff_y), (960, cutoff_y), (0, 0, 255), 1)
        cv2.circle(debug, pivot, radius - 20, (0, 200, 0), 1)
        cv2.circle(debug, pivot, 25,           (0, 0, 200), 1)
        cv2.circle(debug, pivot, 4,            (255, 255, 0), -1)
        cv2.putText(debug, "GREEN=outer mask  BLUE=hub  RED=stem cutoff",
                    (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200,200,200), 1)
        combined = np.hstack([output, debug])
        cv2.imshow("Pressure Gauge v4", combined)
    else:
        cv2.imshow("Pressure Gauge v4", output)

    if psi is not None:
        print(f"PSI: {psi:6.1f}  |  kg/cm²: {kgcm2:.2f}")

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key == ord('d'):
        show_debug = not show_debug
        print(f"Debug: {'ON' if show_debug else 'OFF'}")
    elif key == ord('s'):
        fname = f"snap_{int(cv2.getTickCount())}.png"
        cv2.imwrite(fname, output)
        print(f"Saved: {fname}")
    elif key == ord('c'):
        _cal_clicks.clear()
        angle_buffer.clear()
        cal_pivot, cal_radius = run_calibration(cap)
        if cal_pivot:
            pivot  = cal_pivot
            radius = cal_radius
            print(f"Recalibrated → pivot={pivot}  radius={radius}")

cap.release()
cv2.destroyAllWindows()
