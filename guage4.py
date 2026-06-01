import cv2
import csv
import numpy as np
import os
from collections import deque
from datetime import datetime

# ─────────────────────────────────────────────────────────────
#  CONFIG — tweak these first
# ─────────────────────────────────────────────────────────────
VIDEO_PATH = r"C:\Users\hp\Desktop\image_processing\videos\MVI_3220 - Trim.mp4"

CSV_OUTPUT_PATH = None   # None = auto-generate next to script

PIVOT_X      = 507
PIVOT_Y      = 292
GAUGE_RADIUS = 130

# ── Angle calibration ─────────────────────────────────────────
# HOW TO CALIBRATE LIVE:
#   1. Run the script — a "Calibration" slider window appears.
#   2. Pause the video on a frame where the needle is at 0 PSI  (press SPACE).
#   3. Drag "Angle @ 0 PSI" until the displayed PSI reads 0.
#   4. Resume (SPACE), pause on a frame where needle is at max PSI.
#   5. Drag "Total Sweep" until PSI reads your gauge maximum (150).
#   6. The console prints the final values — copy them into ANGLE_0_PSI /
#      TOTAL_SWEEP below so you don't need to re-calibrate next run.
ANGLE_0_PSI  = 220.0
TOTAL_SWEEP  = 270.0
PSI_MIN      = 0
PSI_MAX      = 150

CALIBRATE_MODE = False   # set True once to click pivot + rim

SMOOTH_WINDOW = 7
angle_buffer  = deque(maxlen=SMOOTH_WINDOW)

# ─────────────────────────────────────────────────────────────
#  CSV LOGGING
# ─────────────────────────────────────────────────────────────
_csv_file    = None
_csv_writer  = None
_csv_enabled = True
_frame_count = 0

def open_csv(path=None):
    global _csv_file, _csv_writer
    if path is None:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        ts   = datetime.now().strftime("%Y%m%d_%H%M%S")
        path = os.path.join(script_dir, f"gauge_log_{ts}.csv")
    _csv_file = open(path, "w", newline="", encoding="utf-8")
    _csv_writer = csv.writer(_csv_file)
    _csv_writer.writerow(["timestamp_s", "frame", "raw_angle_deg",
                          "smoothed_angle_deg", "psi", "kg_cm2"])
    print(f"[CSV] Logging to: {path}")
    return path

def log_csv(ts, frame_no, raw_a, smooth_a, psi, kgcm2):
    if _csv_writer is None or not _csv_enabled:
        return
    _csv_writer.writerow([
        f"{ts:.4f}", frame_no,
        f"{raw_a:.2f}"    if raw_a    is not None else "",
        f"{smooth_a:.2f}" if smooth_a is not None else "",
        f"{psi:.1f}"      if psi      is not None else "",
        f"{kgcm2:.3f}"    if kgcm2    is not None else "",
    ])
    _csv_file.flush()

def close_csv():
    global _csv_file, _csv_writer
    if _csv_file:
        _csv_file.close()
        _csv_file = _csv_writer = None

# ─────────────────────────────────────────────────────────────
#  LIVE CALIBRATION SLIDERS
# ─────────────────────────────────────────────────────────────
CAL_WIN = "Calibration (drag sliders, then press P to print values)"

def create_cal_window():
    """
    Creates a separate window with trackbars for live tuning.
    Trackbar ranges:
      Angle @ 0 PSI : 0–359  (maps directly to degrees)
      Total Sweep   : 180–360
      PSI Max       : 50–300
      Hub mask px   : 10–60   (inner dead-zone radius)
      Stem cutoff % : 0–90    (% below pivot to cut off)
    """
    cv2.namedWindow(CAL_WIN, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(CAL_WIN, 500, 200)
    cv2.createTrackbar("Angle @ 0 PSI", CAL_WIN, int(ANGLE_0_PSI), 359, lambda v: None)
    cv2.createTrackbar("Total Sweep",   CAL_WIN, int(TOTAL_SWEEP),  360, lambda v: None)
    cv2.createTrackbar("PSI Max",       CAL_WIN, int(PSI_MAX),      300, lambda v: None)
    cv2.createTrackbar("Hub mask px",   CAL_WIN, 25,                 60, lambda v: None)
    cv2.createTrackbar("Stem cutoff %", CAL_WIN, 50,                 90, lambda v: None)

def read_cal_sliders():
    a0    = cv2.getTrackbarPos("Angle @ 0 PSI", CAL_WIN)
    sweep = max(1, cv2.getTrackbarPos("Total Sweep",   CAL_WIN))
    pmax  = max(1, cv2.getTrackbarPos("PSI Max",       CAL_WIN))
    hub   = cv2.getTrackbarPos("Hub mask px",   CAL_WIN)
    stem  = cv2.getTrackbarPos("Stem cutoff %", CAL_WIN)
    return float(a0), float(sweep), float(pmax), hub, stem / 100.0

# ─────────────────────────────────────────────────────────────
#  CALIBRATION (pivot + rim click)
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
            px, py = _cal_clicks[0]; rx, ry = _cal_clicks[1]
            r = int(np.hypot(rx-px, ry-py))
            cv2.circle(disp, (px, py), r, (255, 200, 0), 1)
            cv2.putText(disp, f"r={r}", (px+r+5, py),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,200,0), 2)
        cv2.imshow("Calibrate", disp)
        if cv2.waitKey(30) & 0xFF != 255:
            break
    cv2.destroyWindow("Calibrate")
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    if len(_cal_clicks) >= 2:
        px, py = _cal_clicks[0]; rx, ry = _cal_clicks[1]
        r = int(np.hypot(rx-px, ry-py))
        print(f"\n>>> Copy into CONFIG:\n    PIVOT_X={px}\n    PIVOT_Y={py}\n    GAUGE_RADIUS={r}\n    CALIBRATE_MODE=False\n")
        return (px, py), r
    return None, None

# ─────────────────────────────────────────────────────────────
#  HELPERS
# ─────────────────────────────────────────────────────────────
def angle_to_psi(angle_deg, a0, sweep, pmax):
    ratio = (a0 - angle_deg) / sweep
    ratio = max(0.0, min(1.0, ratio))
    psi   = round(PSI_MIN + ratio * (pmax - PSI_MIN), 1)
    return psi, round(psi * 0.0703, 3)

def smooth_angle(new_angle):
    angle_buffer.append(new_angle)
    if len(angle_buffer) < 2:
        return new_angle
    ref = angle_buffer[-1]
    unwrapped = [ref + ((a - ref + 180) % 360 - 180) for a in angle_buffer]
    return float(np.median(unwrapped)) % 360

def build_mask(shape, pivot, radius, hub_px, stem_frac):
    px, py = pivot
    mask = np.zeros(shape[:2], dtype=np.uint8)
    cv2.circle(mask, (px, py), radius - 20, 255, -1)
    cv2.circle(mask, (px, py), hub_px, 0, -1)
    cutoff_y = py + int(radius * stem_frac)
    if cutoff_y < shape[0]:
        mask[cutoff_y:, :] = 0
    return mask

def extract_needle_mask(frame, pivot, radius, hub_px, stem_frac):
    gray   = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    thresh = cv2.adaptiveThreshold(gray, 255,
                cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                cv2.THRESH_BINARY_INV, blockSize=21, C=7)
    ann = build_mask(frame.shape, pivot, radius, hub_px, stem_frac)
    roi = cv2.bitwise_and(thresh, ann)
    k3  = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    roi = cv2.morphologyEx(roi, cv2.MORPH_OPEN,  k3, iterations=1)
    roi = cv2.morphologyEx(roi, cv2.MORPH_CLOSE, k3, iterations=2)
    return roi

def find_longest_dark_blob(mask, pivot):
    n, labels, stats, centroids = cv2.connectedComponentsWithStats(mask)
    if n < 2:
        return None
    px, py = pivot
    best_label, best_score = -1, -1
    for lbl in range(1, n):
        area = stats[lbl, cv2.CC_STAT_AREA]
        if area < 60:
            continue
        length = max(stats[lbl, cv2.CC_STAT_WIDTH], stats[lbl, cv2.CC_STAT_HEIGHT])
        cx, cy = centroids[lbl]
        score = length / (1.0 + np.hypot(cx - px, cy - py) * 0.5)
        if score > best_score:
            best_score, best_label = score, lbl
    return None if best_label < 0 else (labels == best_label).astype(np.uint8) * 255

def pca_angle_from_mask(comp_mask):
    pts = np.column_stack(np.where(comp_mask > 0))
    if len(pts) < 40:
        return None, None, None
    xy   = pts[:, ::-1].astype(np.float32)
    mean = xy.mean(axis=0)
    _, vecs = np.linalg.eigh(np.cov(xy.T))
    principal = vecs[:, -1]
    angle = np.degrees(np.arctan2(-principal[1], principal[0])) % 360
    return angle, float(mean[0]), float(mean[1])

def disambiguate_with_pivot(angle, pivot, comp_mask):
    px, py = pivot
    ys, xs = np.where(comp_mask > 0)
    if len(xs) == 0:
        return angle
    dists   = np.hypot(xs - px, ys - py)
    far_idx = np.argmax(dists)
    vec_far = np.array([float(xs[far_idx]) - px, -(float(ys[far_idx]) - py)])
    norm    = np.linalg.norm(vec_far)
    if norm < 1e-6:
        return angle
    vec_far /= norm
    rad_a = np.radians(angle)
    dir_a = np.array([np.cos(rad_a), np.sin(rad_a)])
    return angle if np.dot(vec_far, dir_a) >= np.dot(vec_far, -dir_a) else (angle + 180) % 360

def walk_to_tip(tip_angle, mean_x, mean_y, comp_mask):
    rad = np.radians(tip_angle)
    dx, dy_img = np.cos(rad), -np.sin(rad)
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
def detect_needle(frame, pivot, radius, a0, sweep, pmax, hub_px, stem_frac):
    frame  = cv2.resize(frame, (960, 540))
    output = frame.copy()
    px, py = pivot

    needle_mask = extract_needle_mask(frame, pivot, radius, hub_px, stem_frac)
    comp_mask   = find_longest_dark_blob(needle_mask, pivot)

    if comp_mask is None:
        cv2.putText(output, "Needle NOT detected",
                    (20, 45), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
        return output, None, None, None, needle_mask

    pca_angle, mean_x, mean_y = pca_angle_from_mask(comp_mask)
    if pca_angle is None:
        cv2.putText(output, "PCA failed",
                    (20, 45), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 165, 255), 2)
        return output, None, None, None, needle_mask

    tip_angle        = disambiguate_with_pivot(pca_angle, pivot, comp_mask)
    tip_x, tip_y     = walk_to_tip(tip_angle, mean_x, mean_y, comp_mask)

    dx        = tip_x - px
    dy        = py - tip_y
    raw_angle = np.degrees(np.arctan2(dy, dx)) % 360
    smoothed  = smooth_angle(raw_angle)
    psi, kgcm2 = angle_to_psi(smoothed, a0, sweep, pmax)

    # ── Draw ──────────────────────────────────────────────────
    overlay = output.copy()
    overlay[comp_mask > 0] = [255, 180, 50]
    output = cv2.addWeighted(output, 0.75, overlay, 0.25, 0)

    cv2.line(output,   (px, py),      (tip_x, tip_y), (0, 255, 0), 3)
    cv2.circle(output, (px, py),       7,              (255, 0,   0), -1)
    cv2.circle(output, (tip_x, tip_y), 5,              (0,   0, 255), -1)
    cv2.circle(output, (px, py), radius - 20, (0, 200, 200), 1)

    # Stem cutoff line (visual aid)
    cutoff_y = py + int(radius * stem_frac)
    cv2.line(output, (px - radius, cutoff_y), (px + radius, cutoff_y),
             (0, 80, 200), 1)

    csv_lbl = "LOG:ON" if _csv_enabled else "LOG:OFF"
    cv2.putText(output, csv_lbl, (20, 215),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65,
                (0, 255, 100) if _csv_enabled else (80, 80, 80), 2)

    cv2.putText(output, f"Angle (raw)    : {raw_angle:.1f} deg",
                (20,  40), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 200, 255), 2)
    cv2.putText(output, f"Angle (smooth) : {smoothed:.1f} deg",
                (20,  75), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 200, 255), 2)
    cv2.putText(output, f"PSI            : {psi} psi",
                (20, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255,   0), 2)
    cv2.putText(output, f"kg/cm2         : {kgcm2}",
                (20, 145), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 255), 2)
    cv2.putText(output, f"a0={a0:.0f} sweep={sweep:.0f} pmax={pmax:.0f}",
                (20, 180), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (180, 180, 60), 1)

    return output, raw_angle, psi, kgcm2, needle_mask

# ─────────────────────────────────────────────────────────────
#  VIDEO LOOP
# ─────────────────────────────────────────────────────────────
cap = cv2.VideoCapture(VIDEO_PATH)
if not cap.isOpened():
    print(f"ERROR: Cannot open → {VIDEO_PATH}")
    exit()

pivot  = (PIVOT_X, PIVOT_Y)
radius = GAUGE_RADIUS

if not CALIBRATE_MODE and (PIVOT_X == 0 or GAUGE_RADIUS == 0):
    print("[WARN] Pivot or radius is 0 — forcing calibration.")
    CALIBRATE_MODE = True

if CALIBRATE_MODE:
    cal_pivot, cal_radius = run_calibration(cap)
    if cal_pivot:
        pivot, radius = cal_pivot, cal_radius

create_cal_window()
csv_path = open_csv(CSV_OUTPUT_PATH)
fps      = cap.get(cv2.CAP_PROP_FPS) or 30.0

print("Pressure Gauge v6")
print("  Q=quit  D=debug  S=snapshot  C=recalibrate  W=toggle CSV  SPACE=pause")
print("  P=print current slider values to console")
print(f"FPS: {fps:.2f}  |  CSV: {csv_path}")

show_debug = False
paused     = False
last_frame = None
last_mask  = None

while True:
    if not paused:
        ret, frame = cap.read()
        if not ret:
            print("Video ended.")
            break
        last_frame = frame.copy()
        _frame_count += 1

    # Always re-read sliders even when paused — live update
    a0, sweep, pmax, hub_px, stem_frac = read_cal_sliders()

    timestamp_s = _frame_count / fps
    output, raw_angle, psi, kgcm2, raw_mask = detect_needle(
        last_frame, pivot, radius, a0, sweep, pmax, hub_px, stem_frac)
    last_mask = raw_mask

    if not paused:
        smoothed_angle = angle_buffer[-1] if angle_buffer else raw_angle
        log_csv(timestamp_s, _frame_count, raw_angle, smoothed_angle, psi, kgcm2)
        if psi is not None:
            print(f"Frame {_frame_count:5d} | t={timestamp_s:7.2f}s | "
                  f"PSI: {psi:6.1f} | kg/cm²: {kgcm2:.3f} | "
                  f"raw°: {raw_angle:.1f}")

    if show_debug and last_mask is not None:
        debug    = cv2.cvtColor(last_mask, cv2.COLOR_GRAY2BGR)
        cutoff_y = pivot[1] + int(radius * stem_frac)
        cv2.line(debug, (0, cutoff_y), (960, cutoff_y), (0, 0, 255), 1)
        cv2.circle(debug, pivot, radius - 20, (0, 200, 0), 1)
        cv2.circle(debug, pivot, hub_px,       (0, 0, 200), 1)
        cv2.circle(debug, pivot, 4,            (255, 255, 0), -1)
        cv2.putText(debug, "GREEN=outer  BLUE=hub  RED=stem cutoff",
                    (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 200), 1)
        combined = np.hstack([output, debug])
        cv2.imshow("Pressure Gauge v6", combined)
    else:
        cv2.imshow("Pressure Gauge v6", output)

    key = cv2.waitKey(1 if not paused else 30) & 0xFF
    if key == ord('q'):
        break
    elif key == ord(' '):
        paused = not paused
        print(f"{'PAUSED — drag sliders to tune live' if paused else 'Resumed'}")
    elif key == ord('p'):
        print(f"\n>>> Calibrated values — copy into CONFIG:\n"
              f"    ANGLE_0_PSI = {a0:.1f}\n"
              f"    TOTAL_SWEEP = {sweep:.1f}\n"
              f"    PSI_MAX     = {pmax:.1f}\n")
    elif key == ord('d'):
        show_debug = not show_debug
        print(f"Debug: {'ON' if show_debug else 'OFF'}")
    elif key == ord('s'):
        fname = f"snap_{int(cv2.getTickCount())}.png"
        cv2.imwrite(fname, output)
        print(f"Saved: {fname}")
    elif key == ord('w'):
        _csv_enabled = not _csv_enabled
        print(f"CSV logging: {'ON' if _csv_enabled else 'OFF'}")
    elif key == ord('c'):
        _cal_clicks.clear()
        angle_buffer.clear()
        cal_pivot, cal_radius = run_calibration(cap)
        if cal_pivot:
            pivot, radius = cal_pivot, cal_radius
            print(f"Recalibrated → pivot={pivot}  radius={radius}")

cap.release()
cv2.destroyAllWindows()
close_csv()
print(f"\nDone. CSV saved to: {csv_path}")