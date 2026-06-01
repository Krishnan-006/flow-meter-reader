"""
STEP 1 — Run this ONCE to calibrate your gauge.
It will open the first video frame and ask you to click 3 points.
At the end it prints the exact config values to paste into step2_gauge.py
"""
import cv2
import numpy as np

VIDEO_PATH = r"C:\Users\hp\Desktop\image_processing\videos\MVI_3220 - Trim.mp4"

clicks = []

def mouse_cb(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN and len(clicks) < 5:
        clicks.append((x, y))

cap = cv2.VideoCapture(VIDEO_PATH)
ret, frame = cap.read()
cap.release()
if not ret:
    print("ERROR: Cannot read video frame.")
    exit()

frame = cv2.resize(frame, (960, 540))
orig  = frame.copy()

cv2.namedWindow("CALIBRATE — follow instructions on screen")
cv2.setMouseCallback("CALIBRATE — follow instructions on screen", mouse_cb)

INSTRUCTIONS = [
    "Click 1/3 : Needle hub CENTER (red dot will appear)",
    "Click 2/3 : Any point on the inner TICK RING edge",
    "Click 3/3 : Needle TIP in this frame (blue arrow appears)",
    "Press ENTER to confirm  |  R to reset  |  Q to quit",
]

print("\n" + "="*55)
print("  GAUGE CALIBRATION")
print("="*55)
for ln in INSTRUCTIONS[:3]:
    print(" ", ln)
print()

while True:
    disp = orig.copy()

    # Instruction overlay
    step = min(len(clicks), 3)
    msg  = INSTRUCTIONS[step] if step < 4 else INSTRUCTIONS[3]
    cv2.rectangle(disp, (0, 0), (960, 36), (30, 30, 30), -1)
    cv2.putText(disp, msg, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 230, 230), 2)

    # Draw clicks
    if len(clicks) >= 1:
        px, py = clicks[0]
        cv2.circle(disp, (px, py), 7, (0, 0, 255), -1)
        cv2.putText(disp, "PIVOT", (px+10, py-6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 255), 2)

    if len(clicks) >= 2:
        px, py = clicks[0]
        rx, ry = clicks[1]
        radius = int(np.hypot(rx-px, ry-py))
        cv2.circle(disp, (rx, ry), 6, (0, 200, 0), -1)
        cv2.circle(disp, (px, py), radius, (0, 200, 0), 2)
        cv2.putText(disp, f"r={radius}", (px+radius+5, py),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 200, 0), 2)

    if len(clicks) >= 3:
        px, py  = clicks[0]
        tx, ty  = clicks[2]
        # Draw the needle line
        cv2.line(disp,   (px, py), (tx, ty), (0, 255, 0), 3)
        cv2.circle(disp, (tx, ty), 6, (255, 80, 0), -1)
        cv2.putText(disp, "TIP", (tx+8, ty-6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 80, 0), 2)

        # Compute angle
        dx  = tx - px
        dy  = py - ty   # flip Y
        ang = np.degrees(np.arctan2(dy, dx)) % 360
        cv2.putText(disp, f"Tip angle = {ang:.1f} deg",
                    (10, 520), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0,255,255), 2)

        # Ask for PSI at this position
        cv2.putText(disp, "What PSI does the needle show? (read gauge, press ENTER)",
                    (10, 495), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200,200,0), 1)

    cv2.imshow("CALIBRATE — follow instructions on screen", disp)
    key = cv2.waitKey(20) & 0xFF

    if key == ord('r') or key == ord('R'):
        clicks.clear()
        print("Reset — click again from step 1")

    elif key == 13 and len(clicks) >= 3:   # ENTER
        break

    elif key == ord('q'):
        cv2.destroyAllWindows()
        exit()

cv2.destroyAllWindows()

# ── Compute calibration values ────────────────────────────────
px, py  = clicks[0]
rx, ry  = clicks[1]
tx, ty  = clicks[2]
radius  = int(np.hypot(rx-px, ry-py))
dx      = tx - px
dy      = py - ty
tip_ang = np.degrees(np.arctan2(dy, dx)) % 360

print("\n" + "="*55)
print("  CALIBRATION RESULT")
print("="*55)
print(f"  Pivot  : ({px}, {py})")
print(f"  Radius : {radius}")
print(f"  Tip angle in this frame : {tip_ang:.1f} deg")
print()

# Ask user for PSI reading at the calibration frame
psi_str = input("  Enter the PSI value the needle is showing right now: ").strip()
try:
    known_psi = float(psi_str)
except ValueError:
    known_psi = None

print()
if known_psi is not None:
    # Back-calculate ANGLE_0_PSI
    # angle_at_0 = tip_ang + (known_psi / PSI_RANGE) * TOTAL_SWEEP
    # (because clockwise sweep means angle decreases as PSI increases)
    PSI_MAX     = 150.0
    TOTAL_SWEEP = 270.0
    angle_0 = (tip_ang + (known_psi / PSI_MAX) * TOTAL_SWEEP) % 360
    print(f"  Calculated ANGLE_0_PSI = {angle_0:.1f}")
    print()
    print("="*55)
    print("  PASTE THESE INTO step2_gauge.py:")
    print("="*55)
    print(f"  PIVOT_X      = {px}")
    print(f"  PIVOT_Y      = {py}")
    print(f"  GAUGE_RADIUS = {radius}")
    print(f"  ANGLE_0_PSI  = {angle_0:.1f}")
    print(f"  TOTAL_SWEEP  = {TOTAL_SWEEP}")
    print(f"  PSI_MIN      = 0")
    print(f"  PSI_MAX      = 150")
else:
    print("  (No PSI entered — using default ANGLE_0_PSI=226)")
    print()
    print("="*55)
    print("  PASTE THESE INTO step2_gauge.py:")
    print("="*55)
    print(f"  PIVOT_X      = {px}")
    print(f"  PIVOT_Y      = {py}")
    print(f"  GAUGE_RADIUS = {radius}")

print()
input("Press ENTER to close.")