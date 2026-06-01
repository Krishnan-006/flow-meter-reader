import cv2
import numpy as np
from collections import deque

# ─────────────────────────────────────────────────────────────
#  CONFIG  –  only change these
# ─────────────────────────────────────────────────────────────
VIDEO_PATH = r"C:\Users\hp\Desktop\image_processing\videos\MVI_3220 - Trim.mp4"

PIVOT_X = 510
PIVOT_Y = 310
GAUGE_RADIUS = 360   # radius of the circular mask around the gauge face

# Angle calibration (CCW from +X axis, standard math convention)
# 0 psi   → needle points at 226 deg
# 150 psi → needle points at 226 - 270 = -44 deg = 316 deg
ANGLE_0_PSI  = 226
TOTAL_SWEEP  = 270.0   # clockwise degrees from 0 → 150 psi

# Scale range
PSI_MIN = 0
PSI_MAX = 150

FLOW_VALID_MIN = 20.0
FLOW_VALID_MAX = 80.0

# ── Smoothing: median over last N valid readings ──────────────
SMOOTH_WINDOW = 7      # number of frames to buffer (increase for more stability)
angle_buffer  = deque(maxlen=SMOOTH_WINDOW)

# ─────────────────────────────────────────────────────────────
#  ANGLE → VALUE
# ─────────────────────────────────────────────────────────────
def angle_to_psi(angle_deg):
    ratio = (ANGLE_0_PSI - angle_deg) / TOTAL_SWEEP
    ratio = max(0.0, min(1.0, ratio))
    psi   = round(PSI_MIN + ratio * (PSI_MAX - PSI_MIN), 1)
    kgcm2 = round(psi * 0.0703, 2)
    return psi, kgcm2


# ─────────────────────────────────────────────────────────────
#  SCORE A LINE SEGMENT
#  Lower score = more needle-like
# ─────────────────────────────────────────────────────────────
def score_line(x1, y1, x2, y2):
    """
    Returns a score (lower = more likely to be the needle).
    Combines:
      1. Distance of the line from the pivot (should be ~0)
      2. How radial the line is (needle passes through pivot)
    """
    length = np.hypot(x2 - x1, y2 - y1)
    if length < 50:
        return 1e9

    dx, dy = x2 - x1, y2 - y1

    # Distance of pivot from the infinite line
    num  = abs(dy * PIVOT_X - dx * PIVOT_Y + x2 * y1 - y2 * x1)
    den  = length + 1e-9
    pivot_dist = num / den

    # "Radialness": the midpoint of the segment should be near the
    # direction from the pivot to the segment midpoint.
    # Penalise segments whose midpoint is far from the pivot
    mx, my = (x1 + x2) / 2, (y1 + y2) / 2
    mid_dist = np.hypot(mx - PIVOT_X, my - PIVOT_Y)

    # Combine: primary = pivot distance, secondary = midpoint distance
    score = pivot_dist * 3 + mid_dist * 0.1
    return score


# ─────────────────────────────────────────────────────────────
#  DISAMBIGUATE TIP vs TAIL
#  The needle tip is the end that is farther from the pivot AND
#  lands in the active arc (ANGLE_0_PSI-TOTAL_SWEEP … ANGLE_0_PSI).
# ─────────────────────────────────────────────────────────────
def pick_tip(x1, y1, x2, y2):
    """
    Return (tip_x, tip_y) — the endpoint that is the needle tip.
    Strategy:
      - Prefer the endpoint farther from pivot (standard).
      - Tie-break: prefer the endpoint whose angle falls inside the
        valid gauge arc, avoiding reading the tail instead of the tip.
    """
    d1 = np.hypot(x1 - PIVOT_X, y1 - PIVOT_Y)
    d2 = np.hypot(x2 - PIVOT_X, y2 - PIVOT_Y)

    def in_arc(angle):
        """True if angle (CCW from +X) is within the valid sweep."""
        lo = (ANGLE_0_PSI - TOTAL_SWEEP) % 360   # ~316 deg for this gauge
        hi = ANGLE_0_PSI                          # 226 deg
        # Arc crosses 0/360? Handle wrap-around
        if lo > hi:
            return angle >= lo or angle <= hi
        else:
            return lo <= angle <= hi

    ang1 = np.degrees(np.arctan2(PIVOT_Y - y1, x1 - PIVOT_X)) % 360
    ang2 = np.degrees(np.arctan2(PIVOT_Y - y2, x2 - PIVOT_X)) % 360

    both_in  = in_arc(ang1) and in_arc(ang2)
    one1_in  = in_arc(ang1)
    one2_in  = in_arc(ang2)

    if not both_in:
        # Prefer whichever endpoint falls inside the valid arc
        if one1_in and not one2_in:
            return x1, y1
        if one2_in and not one1_in:
            return x2, y2

    # Both in arc (or neither): fall back to farther endpoint
    if d1 >= d2:
        return x1, y1
    return x2, y2


# ─────────────────────────────────────────────────────────────
#  BUILD A TIGHT NEEDLE MASK
#  Removes tick marks (short radial marks) from the image before
#  Hough by using a tighter annular region that excludes the rim.
# ─────────────────────────────────────────────────────────────
def make_needle_mask(shape):
    """
    Annular mask: exclude the outer ring where tick marks live,
    and a small dead-zone around the pivot where hub lines cluster.
    """
    mask = np.zeros(shape[:2], dtype=np.uint8)
    # Inner filled circle (exclude hub ≈ 30 px radius)
    cv2.circle(mask, (PIVOT_X, PIVOT_Y), GAUGE_RADIUS - 25, 255, -1)
    cv2.circle(mask, (PIVOT_X, PIVOT_Y), 28, 0, -1)   # black out hub
    return mask


# ─────────────────────────────────────────────────────────────
#  NEEDLE DETECTION ON A SINGLE FRAME
# ─────────────────────────────────────────────────────────────
def detect_needle(frame):
    frame = cv2.resize(frame, (960, 540))
    output = frame.copy()

    # ── 1. Isolate dark needle pixels ────────────────────────
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Adaptive threshold → picks up the dark needle on a bright face
    # even when lighting varies across the gauge
    thresh = cv2.adaptiveThreshold(
        gray, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        blockSize=25,   # neighbourhood size (must be odd)
        C=8             # constant subtracted from mean
    )

    # ── 2. Apply annular mask ─────────────────────────────────
    needle_mask = make_needle_mask(frame.shape)
    roi = cv2.bitwise_and(thresh, needle_mask)

    # Morphological opening: removes isolated dots / noise specks
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    roi    = cv2.morphologyEx(roi, cv2.MORPH_OPEN, kernel, iterations=1)

    # ── 3. Edge detection + Hough ─────────────────────────────
    edges = cv2.Canny(roi, 30, 100)
    lines = cv2.HoughLinesP(
        edges, 1, np.pi / 180,
        threshold=25,
        minLineLength=55,
        maxLineGap=20
    )

    if lines is None:
        cv2.putText(output, "Needle NOT detected",
                    (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
        return output, None, None

    # ── 4. Score all lines, keep the best N, vote on angle ────
    # Score every line and sort ascending (lower = better)
    scored = []
    for line in lines:
        x1, y1, x2, y2 = line[0]
        s = score_line(x1, y1, x2, y2)
        scored.append((s, x1, y1, x2, y2))

    scored.sort(key=lambda t: t[0])

    # Keep top-5 candidates and draw them (debug, green = best)
    TOP_N = 5
    candidates = scored[:TOP_N]

    # Collect their angles (after tip disambiguation)
    candidate_angles = []
    for rank, (s, x1, y1, x2, y2) in enumerate(candidates):
        tx, ty = pick_tip(x1, y1, x2, y2)
        dx  = tx - PIVOT_X
        dy  = PIVOT_Y - ty          # flip Y for math convention
        ang = np.degrees(np.arctan2(dy, dx)) % 360
        candidate_angles.append(ang)

    if not candidate_angles:
        cv2.putText(output, "Needle NOT detected",
                    (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
        return output, None, None

    # ── 5. Choose the best angle from candidates ─────────────
    # Use the single best-scored line's angle (most radial through pivot)
    best_angle = candidate_angles[0]

    # ── 6. Temporal smoothing: median of recent angles ────────
    # Handle angle wrap-around (e.g. 359 → 1) before averaging
    angle_buffer.append(best_angle)

    if len(angle_buffer) >= 2:
        # Unwrap angles relative to the current reading to handle wrap-around
        ref     = angle_buffer[-1]
        unwrapped = []
        for a in angle_buffer:
            diff = (a - ref + 180) % 360 - 180
            unwrapped.append(ref + diff)
        smoothed_angle = float(np.median(unwrapped)) % 360
    else:
        smoothed_angle = best_angle

    # ── 7. Map to PSI ─────────────────────────────────────────
    psi, kgcm2 = angle_to_psi(smoothed_angle)

    # ── 8. Draw the best line & annotate ─────────────────────
    s, x1, y1, x2, y2 = candidates[0]
    tip_x, tip_y = pick_tip(x1, y1, x2, y2)

    # Draw secondary candidates (dim blue) for debugging
    for rank, (_, lx1, ly1, lx2, ly2) in enumerate(candidates[1:], 1):
        cv2.line(output, (lx1, ly1), (lx2, ly2), (255, 150, 0), 1)

    # Draw the best line (bright green)
    cv2.line(output,  (PIVOT_X, PIVOT_Y), (tip_x, tip_y), (0, 255, 0), 3)
    cv2.circle(output, (PIVOT_X, PIVOT_Y), 7,  (255, 0,   0), -1)
    cv2.circle(output, (tip_x,  tip_y),   5,  (0,   0, 255), -1)

    cv2.putText(output, f"Angle (raw)    : {best_angle:.1f} deg",
                (20, 40),  cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 200, 255), 2)
    cv2.putText(output, f"Angle (smooth) : {smoothed_angle:.1f} deg",
                (20, 75),  cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 200, 255), 2)
    cv2.putText(output, f"PSI            : {psi} psi",
                (20, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255,   0), 2)
    cv2.putText(output, f"kg/cm2         : {kgcm2}",
                (20, 145), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 255), 2)

    return output, psi, kgcm2


# ─────────────────────────────────────────────────────────────
#  VIDEO LOOP
# ─────────────────────────────────────────────────────────────
cap = cv2.VideoCapture(VIDEO_PATH)

if not cap.isOpened():
    print(f"ERROR: Cannot open video → {VIDEO_PATH}")
    exit()

print("Reading pressure gauge from video...")
print("Press Q to quit | Press S to save a snapshot | Press D to toggle debug ROI view")

show_debug = False

while True:
    ret, frame = cap.read()

    if not ret:
        print("Video ended. Press any key to close.")
        cv2.waitKey(0)
        break

    output, psi, kgcm2 = detect_needle(frame)

    # ── Optional debug view: show the thresholded ROI ─────────
    if show_debug:
        frame_r = cv2.resize(frame, (960, 540))
        gray    = cv2.cvtColor(frame_r, cv2.COLOR_BGR2GRAY)
        thresh  = cv2.adaptiveThreshold(gray, 255,
                    cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                    cv2.THRESH_BINARY_INV, 25, 8)
        roi     = cv2.bitwise_and(thresh, make_needle_mask(frame_r.shape))
        roi_bgr = cv2.cvtColor(roi, cv2.COLOR_GRAY2BGR)
        combined = np.hstack([output, roi_bgr])
        cv2.imshow("Pressure Gauge | [D] debug  [Q] quit  [S] snap", combined)
    else:
        cv2.imshow("Pressure Gauge | [D] debug  [Q] quit  [S] snap", output)

    if psi is not None:
        print(f"PSI: {psi:6.1f}  |  kg/cm²: {kgcm2:.2f}")

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key == ord('d'):
        show_debug = not show_debug
        print(f"Debug view: {'ON' if show_debug else 'OFF'}")
    elif key == ord('s'):
        fname = f"snapshot_{int(cv2.getTickCount())}.png"
        cv2.imwrite(fname, output)
        print(f"Saved snapshot: {fname}")

cap.release()
cv2.destroyAllWindows()