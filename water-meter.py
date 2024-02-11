import cv2
import pytesseract

# image path
image_path = "./meter.png"

# Load image
image = cv2.imread(image_path)

# Pre-process image (adjust based on image quality)
# Convert to grayscale
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Apply noise reduction (adjust parameters as needed)
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
closed = cv2.morphologyEx(gray, cv2.MORPH_CLOSE, kernel)

# Apply adaptive thresholding (adjust parameters as needed)
thresh = cv2.adaptiveThreshold(closed, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 11, 2)

# Find contours and locate digit region
contours, hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
roi = None
for cnt in contours:
    x, y, w, h = cv2.boundingRect(cnt)
    # Adjust aspect ratio and size criteria based on your meter format
    if w / h > 0.5 and w / h < 1.5 and 20 < w < 100:
        roi = thresh[y:y+h, x:x+w]
        break

# Check if digit region found
if roi is None:
    print("Error: Digit region not found!")
else:
    # Apply additional pre-processing if needed (e.g., deskewing)

    # Perform OCR with Tesseract
    text = pytesseract.image_to_string(roi, config='--psm 10')

    # Post-process the extracted text (e.g., remove non-numeric characters)
    digits = ''.join(c for c in text if c.isdigit())

    # Print the extracted reading
    print("Extracted reading:", digits)
