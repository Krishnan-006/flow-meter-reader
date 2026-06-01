import cv2
import numpy as np
import math

# -----------------------------
# Gauge Settings
# -----------------------------

MIN_ANGLE = 0     # Needle angle at minimum value
MAX_ANGLE = 360     # Needle angle at maximum value

MIN_VALUE = 0        # Gauge minimum reading
MAX_VALUE = 150      # Gauge maximum reading

# -----------------------------
# Video Capture
# -----------------------------

cap = cv2.VideoCapture(r"C:\Users\hp\Desktop\image_processing\videos\MVI_3220 - Trim.mp4")

while True:

    ret, frame = cap.read()

    if not ret:
        break

    # Resize
    frame = cv2.resize(frame, (800, 600))

    output = frame.copy()

    # Convert to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Blur
    blur = cv2.GaussianBlur(gray, (5, 5), 0)

    # Detect circles (Gauge)
    circles = cv2.HoughCircles(
        blur,
        cv2.HOUGH_GRADIENT,
        1,
        100,
        param1=100,
        param2=30,
        minRadius=100,
        maxRadius=300
    )

    if circles is not None:

        circles = np.round(circles[0, :]).astype("int")

        for (x, y, r) in circles:

            # Draw gauge circle
            cv2.circle(output, (x, y), r, (0, 255, 0), 2)

            # ROI of gauge
            roi = frame[y-r:y+r, x-r:x+r]

            if roi.size == 0:
                continue

            roi_gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)

            # Edge detection
            edges = cv2.Canny(roi_gray, 50, 150)

            # Detect lines (needle)
            lines = cv2.HoughLinesP(
                edges,
                1,
                np.pi/180,
                threshold=50,
                minLineLength=50,
                maxLineGap=10
            )

            if lines is not None:

                longest_line = None
                max_len = 0

                for line in lines:

                    x1, y1, x2, y2 = line[0]

                    length = math.hypot(x2 - x1, y2 - y1)

                    if length > max_len:
                        max_len = length
                        longest_line = (x1, y1, x2, y2)

                if longest_line:

                    x1, y1, x2, y2 = longest_line

                    # Draw needle
                    cv2.line(
                        roi,
                        (x1, y1),
                        (x2, y2),
                        (0, 0, 255),
                        3
                    )

                    # Calculate angle
                    dx = x2 - x1
                    dy = y1 - y2

                    angle = math.degrees(math.atan2(dy, dx))

                    # Normalize angle
                    angle = angle - 180

                    # Clamp
                    angle = max(MIN_ANGLE, min(MAX_ANGLE, angle))

                    # Map angle to value
                    value = np.interp(
                        angle,
                        [MIN_ANGLE, MAX_ANGLE],
                        [MIN_VALUE, MAX_VALUE]
                    )

                    # Display
                    cv2.putText(
                        output,
                        f"Value: {value:.2f}",
                        (50, 50),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1,
                        (0, 255, 0),
                        2
                    )

    cv2.imshow("Gauge Reader", output)

    key = cv2.waitKey(1)

    if key == 27:
        break

cap.release()
cv2.destroyAllWindows()