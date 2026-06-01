import cv2
image = cv2.imread(r"C:\Users\hp\Desktop\image_processing\images\sample2.jpeg")
def preprocess_image(image):
    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Reduce noise
    blur = cv2.GaussianBlur(gray, (5, 5), 0)

    # Improve contrast
    equalized = cv2.equalizeHist(blur)

    # Edge detection
    edges = cv2.Canny(equalized, 50, 150)
  
    return edges
cv2.imshow("Rotameter", image)

cv2.waitKey(0)
cv2.destroyAllWindows()