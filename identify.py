"""
STEP 2 — Main gauge reader.
Paste values from step1_calibrate.py output into the CONFIG block below.
"""
import cv2
import numpy as np
from collections import deque

# ═════════════════════════════════════════════════════════════
#  CONFIG  — paste step1 output here
# ═════════════════════════════════════════════════════════════
VIDEO_PATH   = r"C:\Users\hp\Desktop\image_processing\videos\MVI_3220 - Trim.mp4"

PIVOT_X      = 507
PIVOT_Y      = 291
GAUGE_RADIUS = 102
ANGLE_0_PSI  = 223.8
TOTAL_SWEEP  = 270.0
PSI_MIN      = 0
PSI_MAX      = 150
SMOOTH_WINDOW = 9       # increase for smoother (adds slight lag)
# ═════════════════════════════════════════════════════════════

angle_buf = deque(maxlen=SMOOTH_WINDOW)

# ── Maths ─────────────────────────────────────────────────────
def angle_to_psi(a):
    ratio = (ANGLE_0_PSI - a) / TOTAL_SWEEP
    ratio = max(0.0, min(1.0, ratio))
    psi   = round(PSI_MIN + ratio * (PSI_MAX - PSI_MIN), 1)
    return psi, round(psi * 0.0703, 2)

def smooth(a):
    angle_buf.append(a)
    ref = angle_buf[-1]
    uw  = [ref + ((x - ref + 180) % 360 - 180) for x in angle_buf]
    return float(np.median(uw)) % 360

# ── Mask ──────────────────────────────────────────────────────
def make_mask(shape, pivot, radius):
    px, py = pivot
    m = np.zeros(shape[:2], dtype=np.uint8)
    cv2.circle(m, (px, py), radius - 15, 255, -1)  # inside tick ring
    cv2.circle(m, (px, py), 22,          0,   -1)  # remove hub
    # Cut off bottom stem
    cutoff = py + int(radius * 0.55)
    if cutoff < shape[0]:
        m[cutoff:, :] = 0
    return m

# ── Detection ─────────────────────────────────────────────────
def detect(frame, pivot, radius):
    frame  = cv2.resize(frame, (960, 540))
    output = frame.copy()
    px, py = pivot

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Adaptive threshold to handle variable lighting
    thr = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV, 21, 7)

    ann = make_mask(frame.shape, pivot, radius)
    roi = cv2.bitwise_and(thr, ann)

    k = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    roi = cv2.morphologyEx(roi, cv2.MORPH_OPEN,  k, iterations=1)
    roi = cv2.morphologyEx(roi, cv2.MORPH_CLOSE, k, iterations=2)

    # ── Find best blob (passes through pivot, is long) ────────
    n, labels, stats, centroids = cv2.connectedComponentsWithStats(roi)
    best_lbl, best_score = -1, -1.0

    for lbl in range(1, n):
        area = stats[lbl, cv2.CC_STAT_AREA]
        if area < 40:
            continue
        pts = np.column_stack(np.where(labels == lbl))[:, ::-1].astype(np.float32)
        if len(pts) < 20:
            continue
        vx, vy, x0, y0 = cv2.fitLine(pts, cv2.DIST_L2, 0, 0.01, 0.01)
        vx, vy, x0, y0 = float(vx), float(vy), float(x0), float(y0)

        # Distance of pivot from this fitted line
        dpiv = abs((x0 - px)*vy - (y0 - py)*vx)

        # Length of blob along its axis
        projs = (pts[:,0]-x0)*vx + (pts[:,1]-y0)*vy
        length = projs.max() - projs.min()

        # Score: long + passes close to pivot
        score = length / (1.0 + dpiv * 1.5)
        if score > best_score:
            best_score = score
            best_lbl   = lbl

    if best_lbl < 0:
        cv2.putText(output, "Needle NOT detected",
                    (20, 45), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
        return output, None, None, roi

    blob = (labels == best_lbl).astype(np.uint8) * 255

    # ── Fit line → get two extreme endpoints ──────────────────
    pts = np.column_stack(np.where(blob > 0))[:, ::-1].astype(np.float32)
    vx, vy, x0, y0 = cv2.fitLine(pts, cv2.DIST_L2, 0, 0.01, 0.01)
    vx, vy, x0, y0 = float(vx), float(vy), float(x0), float(y0)
    projs  = (pts[:,0]-x0)*vx + (pts[:,1]-y0)*vy
    t_min, t_max = projs.min(), projs.max()
    end_a  = (x0 + t_min*vx, y0 + t_min*vy)
    end_b  = (x0 + t_max*vx, y0 + t_max*vy)

    # ── Pick tip = endpoint FARTHER from pivot ─────────────────
    # AND outside the hub dead-zone (tail/counterweight is near hub)
    HUB_R = 24
    da = np.hypot(end_a[0]-px, end_a[1]-py)
    db = np.hypot(end_b[0]-px, end_b[1]-py)

    # If one end is inside hub → that's the tail, other is tip
    a_in_hub = da < HUB_R
    b_in_hub = db < HUB_R

    if a_in_hub and not b_in_hub:
        tip = end_b
    elif b_in_hub and not a_in_hub:
        tip = end_a
    else:
        # Both outside hub: pick farther one
        tip = end_b if db >= da else end_a

    tip_x, tip_y = int(round(tip[0])), int(round(tip[1]))

    # ── Angle and PSI ─────────────────────────────────────────
    raw_a    = np.degrees(np.arctan2(py - tip_y, tip_x - px)) % 360
    smooth_a = smooth(raw_a)
    psi, kgcm2 = angle_to_psi(smooth_a)

    # ── Draw ──────────────────────────────────────────────────
    # Highlight blob
    ov = output.copy()
    ov[blob > 0] = [40, 220, 40]
    output = cv2.addWeighted(output, 0.78, ov, 0.22, 0)

    # Both endpoints as small circles
    cv2.circle(output, (int(end_a[0]), int(end_a[1])), 4, (180, 180, 0), -1)
    cv2.circle(output, (int(end_b[0]), int(end_b[1])), 4, (180, 180, 0), -1)

    # Pivot → tip line
    cv2.line(output,   (px, py), (tip_x, tip_y), (0, 255, 0), 3)
    cv2.circle(output, (px, py),    8, (0,   0, 255), -1)   # pivot = blue
    cv2.circle(output, (tip_x, tip_y), 6, (0, 255, 255), -1) # tip = cyan

    # Mask circle
    cv2.circle(output, (px, py), radius - 15, (0, 200, 200), 1)

    cv2.putText(output, f"Angle (raw)    : {raw_a:.1f} deg",
                (20, 40),  cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 200, 255), 2)
    cv2.putText(output, f"Angle (smooth) : {smooth_a:.1f} deg",
                (20, 75),  cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 200, 255), 2)
    cv2.putText(output, f"PSI            : {psi} psi",
                (20, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.9,  (0, 255, 0),   2)
    cv2.putText(output, f"kg/cm2         : {kgcm2}",
                (20, 148), cv2.FONT_HERSHEY_SIMPLEX, 0.9,  (0, 255, 255), 2)

    return output, psi, kgcm2, roi

# ── Video loop ────────────────────────────────────────────────
cap = cv2.VideoCapture(VIDEO_PATH)
if not cap.isOpened():
    print(f"ERROR: Cannot open → {VIDEO_PATH}")
    exit()

pivot  = (PIVOT_X, PIVOT_Y)
radius = GAUGE_RADIUS
show_debug = False

print("Pressure Gauge v5  |  Q=quit  D=debug mask  S=snapshot")

while True:
    ret, frame = cap.read()
    if not ret:
        print("Video ended.")
        cv2.waitKey(0)
        break

    output, psi, kgcm2, mask = detect(frame, pivot, radius)

    if show_debug:
        dbg = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
        cutoff = PIVOT_Y + int(radius * 0.55)
        cv2.line(dbg, (0, cutoff), (960, cutoff), (0, 0, 255), 1)
        cv2.circle(dbg, pivot, radius - 15, (0, 200, 0), 1)
        cv2.circle(dbg, pivot, 22, (100, 100, 255), 1)
        cv2.circle(dbg, pivot, 4, (0, 255, 255), -1)
        cv2.putText(dbg, "green=mask ring  blue=hub zone  red=stem cutoff",
                    (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180,180,180), 1)
        cv2.imshow("Pressure Gauge", np.hstack([output, dbg]))
    else:
        cv2.imshow("Pressure Gauge", output)

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

cap.release()
cv2.destroyAllWindows()
