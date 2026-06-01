import cv2
from value_mapping import best_needle_line
# Load image
image = cv2.imread(r"C:\Users\hp\Desktop\image_processing\images\sample2.jpeg")

# Check image loaded
if image is None:
    print("Image not found")
    exit()

# Resize for display
image = cv2.resize(image, (1024, 576))

# Show image
cv2.imshow("Rotameter", image)

cv2.waitKey(0)
cv2.destroyAllWindows()
