import cv2
import numpy as np

# ─────────────────────────────────────────────────────────────
#  CONFIG  –  only change these
# ─────────────────────────────────────────────────────────────
VIDEO_PATH = r"C:\Users\hp\Desktop\image_processing\videos\MVI_3220 - Trim.mp4"

PIVOT_X = 485
PIVOT_Y = 280

# Angle calibration (CCW from +X axis)
# 0 psi  → needle at 238 deg (lower-left, ~8 o'clock)
# 150 psi → needle at 328 deg (lower-right, ~4 o'clock)  — 270 deg CW sweep
ANGLE_0_PSI   = 226  # angle when needle points at 0
TOTAL_SWEEP   = 270.0   # total clockwise degrees from 0 → 150 psi

# Scale range
PSI_MIN =0
PSI_MAX =150

# ─────────────────────────────────────────────────────────────
#  ANGLE → VALUE
# ─────────────────────────────────────────────────────────────
def angle_to_psi(angle_deg):
    # Gauge is CW → as angle decreases (CCW conv.) value increases
    ratio = (ANGLE_0_PSI - angle_deg) / TOTAL_SWEEP
    ratio = max(0.0, min(1.0, ratio))
    psi   = round(PSI_MIN + ratio * (PSI_MAX - PSI_MIN), 1)
    kgcm2 = round(psi * 0.0703, 2)
    return psi, kgcm2


# ─────────────────────────────────────────────────────────────
#  NEEDLE DETECTION ON A SINGLE FRAME
# ─────────────────────────────────────────────────────────────
def detect_needle(frame):
    """
    Returns (annotated_frame, psi, kgcm2)
    """
    frame = cv2.resize(frame, (960, 540))
    output = frame.copy()

    # ── Isolate dark needle pixels inside gauge face ──────
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mask_dark = cv2.inRange(hsv,
                            np.array([0,   0,   0]),
                            np.array([180, 255, 80]))

    # Circular mask around gauge face
    gauge_mask = np.zeros(mask_dark.shape, dtype=np.uint8)
    cv2.circle(gauge_mask, (PIVOT_X, PIVOT_Y), 210, 255, -1)
    needle_region = cv2.bitwise_and(mask_dark, gauge_mask)

    # ── Edge + Hough lines ────────────────────────────────
    edges = cv2.Canny(needle_region, 30, 100)
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180,
                             threshold=30,
                             minLineLength=50,
                             maxLineGap=15)

    if lines is None:
        cv2.putText(output, "Needle NOT detected",
                    (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
        return output, None, None

    # ── Pick line closest to pivot ────────────────────────
    best_line  = None
    best_dist  = 1e9

    for line in lines:
        x1, y1, x2, y2 = line[0]
        length = np.hypot(x2 - x1, y2 - y1)
        if length < 50:
            continue
        dx, dy = x2 - x1, y2 - y1
        num  = abs(dy * PIVOT_X - dx * PIVOT_Y + x2 * y1 - y2 * x1)
        den  = np.hypot(dx, dy) + 1e-9
        dist = num / den
        if dist < best_dist:
            best_dist = dist
            best_line = line[0]

    if best_line is None:
        cv2.putText(output, "Needle NOT detected",
                    (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
        return output, None, None

    x1, y1, x2, y2 = best_line

    # ── Tip = endpoint farther from pivot ─────────────────
    d1 = np.hypot(x1 - PIVOT_X, y1 - PIVOT_Y)
    d2 = np.hypot(x2 - PIVOT_X, y2 - PIVOT_Y)
    tip_x, tip_y = (x1, y1) if d1 > d2 else (x2, y2)

    # ── Compute CCW angle from pivot to tip ───────────────
    dx = tip_x - PIVOT_X
    dy = PIVOT_Y - tip_y          # flip Y (OpenCV Y grows downward)
    angle_deg = np.degrees(np.arctan2(dy, dx)) % 360

    # ── Map to PSI / kg/cm² ───────────────────────────────
    psi, kgcm2 = angle_to_psi(angle_deg)

    # ── Annotate ──────────────────────────────────────────
    cv2.line(output, (PIVOT_X, PIVOT_Y), (tip_x, tip_y), (0, 255, 0), 3)
    cv2.circle(output, (PIVOT_X, PIVOT_Y), 7, (255, 0,   0), -1)
    cv2.circle(output, (tip_x,  tip_y),   5, (0,   0, 255), -1)

    cv2.putText(output, f"Angle  : {angle_deg:.1f} deg",
                (20, 40),  cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 200, 255), 2)
    cv2.putText(output, f"PSI    : {psi} psi",
                (20, 80),  cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255,   0), 2)
    cv2.putText(output, f"kg/cm2 : {kgcm2}",
                (20, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

    return output, psi, kgcm2


# ─────────────────────────────────────────────────────────────
#  VIDEO LOOP
# ─────────────────────────────────────────────────────────────
cap = cv2.VideoCapture(VIDEO_PATH)

if not cap.isOpened():
    print(f"ERROR: Cannot open video → {VIDEO_PATH}")
    exit()

print("Reading pressure gauge from video... Press Q to quit.")

while True:

    ret, frame = cap.read()

    if not ret:
        print("Video ended.")
        # Hold last frame until Q pressed
        cv2.waitKey(0)
        break

    output, psi, kgcm2 = detect_needle(frame)

    if psi is not None:
        print(f"PSI: {psi:6.1f}  |  kg/cm²: {kgcm2:.2f}")

    cv2.imshow("Pressure Gauge", output)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()