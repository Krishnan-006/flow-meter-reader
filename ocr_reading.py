import cv2
import pytesseract
from value_mapping import angle_to_value
pytesseract.pytesseract.tesseract_cmd = (
    r"C:\Program Files\Tesseract-OCR\tesseract.exe"
)
image = cv2.imread(r"C:\Users\hp\Desktop\image_processing\images\sample2.jpeg")

gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

text = pytesseract.image_to_string(gray)

print(text)