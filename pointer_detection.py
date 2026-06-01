import cv2

# Open Video
cap = cv2.VideoCapture(
    r
)

# Float Detection Function
def detect_float(cap):

    # Read frame from video
    ret, frame = cap.read()

    # Check if frame loaded
    if not ret:
        return None, None

    # Resize frame
    frame = cv2.resize(frame, (960, 540))

    # Convert to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Threshold
    _, thresh = cv2.threshold(
        gray,
        120,
        255,
        cv2.THRESH_BINARY_INV
    )

    # Find contours
    contours, _ = cv2.findContours(
        thresh,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    largest = None
    max_area = 0

    # Find largest contour
    for contour in contours:

        area = cv2.contourArea(contour)

        if area > max_area:

            max_area = area
            largest = contour

    # Draw bounding box
    if largest is not None:

        x, y, w, h = cv2.boundingRect(largest)

        center_y = y + h // 2

        cv2.rectangle(
            frame,
            (x, y),
            (x + w, y + h),
            (0,255,0),
            2
        )

        # Display detected position
        cv2.putText(
            frame,
            f"Y Position: {center_y}",
            (20,40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0,0,255),
            2
        )

        return frame, center_y

    return frame, None


# VIDEO LOOP
# VIDEO LOOP
while True:

    # Detect float
    result, center_y = detect_float(cap)

    # If video ended
    if result is None:

        print("Video Finished")

        while True:

            key = cv2.waitKey(0)

            if key == ord('q'):
                break

        break

    # Print position
    print("Detected Y Position:", center_y)

    # Show output
    cv2.imshow("Float Detection", result)

    # Press Q to quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


# Release resources
cap.release()
cv2.destroyAllWindows()