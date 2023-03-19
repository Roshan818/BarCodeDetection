import cv2
import numpy as np
import pyzbar.pyzbar as pyzbar

# Define the colors for bounding boxes
BLUE = (255, 0, 0)
YELLOW = (0, 255, 255)
RED = (0, 0, 255)
BLACK = (0, 0, 0)

# Load the image
image = cv2.imread("BarCodeDetection/sample/allbarcode/IMG_20220303_173846.jpg")

# Convert the image to grayscale
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Detect barcodes using pyzbar
barcodes = pyzbar.decode(gray)

# Store barcode values and their count in a dictionary
barcode_counts = {}
for barcode in barcodes:
    barcode_value = barcode.data.decode("utf-8")
    if barcode_value in barcode_counts:
        barcode_counts[barcode_value] += 1
    else:
        barcode_counts[barcode_value] = 1

# Loop over each detected barcode and draw a bounding box around it
for barcode in barcodes:
    # Extract barcode value and bounding box coordinates
    barcode_value = barcode.data.decode("utf-8")
    x, y, w, h = barcode.rect

    # Draw a blue bounding box around barcodes that were correctly read
    if barcode_counts[barcode_value] == 1:
        cv2.rectangle(image, (x, y), (x + w, y + h), BLUE, 2)
    # Draw a yellow bounding box around partially visible barcodes
    else:
        cv2.rectangle(image, (x, y), (x + w, y + h), YELLOW, 2)

# Loop over the image and identify items without barcodes
gray_blurred = cv2.GaussianBlur(gray, (5, 5), 0)
edges = cv2.Canny(gray_blurred, 100, 200)
contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
for contour in contours:
    x, y, w, h = cv2.boundingRect(contour)
    if w < 100 and h < 100:
        cv2.rectangle(image, (x, y), (x + w, y + h), RED, 2)

# Draw black bounding boxes around each item in the image
gray_blurred = cv2.GaussianBlur(gray, (5, 5), 0)
edges = cv2.Canny(gray_blurred, 100, 200)
contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
for contour in contours:
    x, y, w, h = cv2.boundingRect(contour)
    if w > 100 and h > 100:
        cv2.rectangle(image, (x, y), (x + w, y + h), BLACK, 2)

# Display the input image with bounding boxes
cv2.imshow("Grocery Items", image)
cv2.waitKey(0)

# Print barcode values and their count
for barcode_value, count in barcode_counts.items():
    print(f"{barcode_value}: {count}")
