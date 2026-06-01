import cv2
import sys

# ─────────────────────────────────────────────
# CONFIG  –  edit these paths before running
# ─────────────────────────────────────────────
IMAGE_PATH = r"C:\Users\hp\Desktop\image_processing\images\sample2.jpeg"
VIDEO_PATH = r"C:\Users\hp\Desktop\image_processing\videos\MVI_3223.MP4"

MODE = "image"   # "image"  or  "video"

# ─────────────────────────────────────────────
# VALUE MAPPING
# Adjust min/max pixel & value to match your
# rotameter's scale in the resized frame (960x540)
# ─────────────────────────────────────────────
def pixel_to_value(pixel_y,
                   min_pixel=100, max_pixel=500,
                   min_value=48,  max_value=480):
    """
    Map a Y pixel position to a flow-rate value.
    NOTE: Y increases downward in OpenCV, so a
    higher pixel_y → lower on the tube → lower reading.
    The ratio is therefore inverted below.
    """
    ratio = (pixel_y - min_pixel) / (max_pixel - min_pixel)
    # Invert: top of tube (small y) = high reading
    value = max_value - ratio * (max_value - min_value)
    return round(value, 2)


# ─────────────────────────────────────────────
# FLOAT DETECTION  (works on a single frame)
# ─────────────────────────────────────────────
def detect_float(frame):
    """
    Detects the float in a single BGR frame.
    Returns (annotated_frame, center_y) or (frame, None).
    """
    # Resize to a standard resolution
    frame = cv2.resize(frame, (960, 540))

    # Convert to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # ── Preprocessing: blur to reduce noise ──
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # ── Threshold: try OTSU for automatic level ──
    _, thresh = cv2.threshold(
        blurred, 0, 255,
        cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
    )

    # ── Morphological cleanup ──
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN,  kernel)

    # ── Find contours ──
    contours, _ = cv2.findContours(
        thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    if not contours:
        print("No contours found – try adjusting the threshold.")
        return frame, None

    # ── Pick the largest contour ──
    largest = max(contours, key=cv2.contourArea)
    area    = cv2.contourArea(largest)

    # Ignore contours that are too small (noise)
    if area < 200:
        print(f"Largest contour too small (area={area:.0f}) – possible noise.")
        return frame, None

    x, y, w, h = cv2.boundingRect(largest)
    center_y   = y + h // 2
    center_x   = x + w // 2

    # ── Annotate frame ──
    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
    cv2.circle(frame, (center_x, center_y), 5, (255, 0, 0), -1)
    cv2.putText(frame,
                f"Y Pixel: {center_y}",
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

    return frame, center_y


# ─────────────────────────────────────────────
# ANNOTATE FLOW RATE ON FRAME
# ─────────────────────────────────────────────
def annotate_value(frame, y_position):
    value = pixel_to_value(y_position)
    print(f"  Y Pixel → {y_position}   |   Flow Rate → {value} LPM")
    cv2.putText(frame,
                f"Flow Rate: {value} LPM",
                (20, 80),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
    return frame, value


# ─────────────────────────────────────────────
# IMAGE MODE
# ─────────────────────────────────────────────
def run_image(path):
    frame = cv2.imread(path)
    if frame is None:
        print(f"ERROR: Could not load image → {path}")
        sys.exit(1)

    print("Running float detection on image …")
    output, y_position = detect_float(frame)

    if y_position is not None:
        output, _ = annotate_value(output, y_position)
    else:
        print("Float not detected. Check lighting / threshold settings.")

    cv2.imshow("Rotameter – Image Mode", output)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


# ─────────────────────────────────────────────
# VIDEO MODE
# ─────────────────────────────────────────────
def run_video(path):
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        print(f"ERROR: Could not open video → {path}")
        sys.exit(1)

    print("Running float detection on video … Press Q to quit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Video ended.")
            break

        output, y_position = detect_float(frame)

        if y_position is not None:
            output, _ = annotate_value(output, y_position)

        cv2.imshow("Rotameter – Video Mode", output)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


# ─────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────
if __name__ == "__main__":
    if MODE == "image":
        run_image(IMAGE_PATH)
    elif MODE == "video":
        run_video(VIDEO_PATH)
    else:
        print("ERROR: MODE must be 'image' or 'video'")
        sys.exit(1)