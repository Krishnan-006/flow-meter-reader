import cv2
import numpy as np
# ─────────────────────────────────────────────────────────────
#  CONFIG
# ─────────────────────────────────────────────────────────────
IMAGE_PATH = r"C:\Users\hp\Desktop\image_processing\images\sample2.jpeg"

# Pivot point of the needle in the RESIZED frame (960x540)
# The black pivot/knob is roughly at bottom-right of the arc
# Tune these if needed after running once
PIVOT_X = 720
PIVOT_Y = 400

# Angle calibration (measured from positive X-axis, counter-clockwise)
# Run with DEBUG = True first to find these angles visually
ANGLE_AT_48  = 210  # degrees — needle pointing at 48  (bottom-left)
ANGLE_AT_480 =  90.0   # degrees — needle pointing at 480 (top)

# Scale min / max
VALUE_MIN = 48
VALUE_MAX = 480

# Set True to show intermediate debug windows
DEBUG = True

# ─────────────────────────────────────────────────────────────
#  HELPERS
# ─────────────────────────────────────────────────────────────

def angle_to_value(angle_deg):
    """Map needle angle (degrees) to scale value."""
    ratio = (angle_deg - ANGLE_AT_48) / (ANGLE_AT_480 - ANGLE_AT_48)
    ratio = max(0.0, min(1.0, ratio))
    return round(VALUE_MIN + ratio * (VALUE_MAX - VALUE_MIN), 1)


def best_needle_line(lines, pivot, img_shape):
    """
    From all Hough lines pick the one that:
      - passes closest to the pivot point
      - is long enough to be the needle (not a scale tick)
    Returns (x1,y1,x2,y2) or None.
    """
    h, w = img_shape[:2]
    best_line = None
    best_score = 1e9

    for line in lines:
        x1, y1, x2, y2 = line[0]
        length = np.hypot(x2 - x1, y2 - y1)
        if length < 80:          # ignore short ticks
            continue

        # Distance from pivot to the infinite line
        dx, dy = x2 - x1, y2 - y1
        num = abs(dy * pivot[0] - dx * pivot[1] + x2 * y1 - y2 * x1)
        den = np.hypot(dx, dy) + 1e-9
        dist = num / den

        # Score: prioritise lines close to pivot AND long
        score = dist - length * 0.1
        if score < best_score:
            best_score = score
            best_line  = (x1, y1, x2, y2)

    return best_line


# ─────────────────────────────────────────────────────────────
#  MAIN
# ─────────────────────────────────────────────────────────────

def extract_value(image_path):

    # ── Load ───────────────────────────────────────────────
    frame = cv2.imread(image_path)
    if frame is None:
        print(f"ERROR: Cannot load → {image_path}")
        return

    frame = cv2.resize(frame, (1024,576))
    output = frame.copy()

    # ── Grayscale + blur ───────────────────────────────────
    gray    = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # ── Edge detection ─────────────────────────────────────
    edges = cv2.Canny(blurred, 50, 150, apertureSize=3)

    if DEBUG:
        cv2.imshow("Edges", edges)

    # ── Hough line detection ───────────────────────────────
    lines = cv2.HoughLinesP(
        edges,
        rho=1,
        theta=np.pi / 180,
        threshold=60,
        minLineLength=80,
        maxLineGap=20
    )

    if lines is None:
        print("No lines detected. Try lowering the Canny/Hough thresholds.")
        cv2.imshow("Result", output)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        return

    # ── Pick best needle line ──────────────────────────────
    pivot = (PIVOT_X, PIVOT_Y)
    needle = best_needle_line(lines, pivot, frame.shape)

    if needle is None:
        print("Needle not found among detected lines.")
        cv2.imshow("Result", output)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        return

    x1, y1, x2, y2 = needle

    # ── Choose the tip (point farther from pivot) ──────────
    d1 = np.hypot(x1 - PIVOT_X, y1 - PIVOT_Y)
    d2 = np.hypot(x2 - PIVOT_X, y2 - PIVOT_Y)
    tip_x, tip_y = (x1, y1) if d1 > d2 else (x2, y2)

    # ── Compute angle from pivot to tip ────────────────────
    dx = tip_x - PIVOT_X
    dy = PIVOT_Y - tip_y          # flip Y (OpenCV Y grows downward)
    angle_deg = np.degrees(np.arctan2(dy, dx)) % 360

    # ── Map angle → value ──────────────────────────────────
    value = angle_to_value(angle_deg)

    print(f"Needle tip  : ({tip_x}, {tip_y})")
    print(f"Angle       : {angle_deg:.1f}°")
    print(f"Flow Rate   : {value} Nm³/h")

    # ── Annotate output ────────────────────────────────────
    cv2.line(output, (PIVOT_X, PIVOT_Y), (tip_x, tip_y), (0, 255, 0), 3)
    cv2.circle(output, (PIVOT_X, PIVOT_Y), 8, (255, 0, 0), -1)
    cv2.circle(output, (tip_x,  tip_y),   6, (0, 0, 255), -1)

    cv2.putText(output, f"Angle : {angle_deg:.1f} deg",
                (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 200, 255), 2)
    cv2.putText(output, f"Value : {value} Nm3/h",
                (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0),   2)

    cv2.imshow("Result", output)
    cv2.waitKey(0)



# ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    extract_value(IMAGE_PATH)
