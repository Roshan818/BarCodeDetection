import cv2
import numpy as np
from pyzbar.pyzbar import decode
import argparse
import os

# Import YOLOv5 from the installed package
from yolov5 import detect

# Define the colors for bounding boxes
BLUE = (255, 0, 0)
YELLOW = (0, 255, 255)
RED = (0, 0, 255)
BLACK = (0, 0, 0)

# Parse the command line arguments
parser = argparse.ArgumentParser()
parser.add_argument("BarCodeDetection/sample/allbarcode/IMG_20220303_175324.jpg", type=str, help="path to input image")
args = parser.parse_args()

# Define the paths for the YOLOv5 model and class names file
model_path = 'yolov5s.pt'
class_names_path = 'coco.names'

# Define the object detector and barcode detector
object_detector = detect.Detector(model_path, class_names_path)
barcode_detector = decode

# Load the input image
img = cv2.imread(args.image_path)

# Detect the objects in the image using YOLOv5
class_ids, confidences, boxes = object_detector.detect(img)

# Get the cropped images and barcode values from the grocery scanner
cropped_images = []
barcode_values = []
for i, box in enumerate(boxes):
    x1, y1, x2, y2 = box
    cropped_img = img[y1:y2, x1:x2]
    cropped_images.append(cropped_img)
    barcode_value = barcode_detector(cropped_img)
    if len(barcode_value) > 0:
        barcode_values.append(barcode_value[0].data.decode())

# Count the unique barcodes
barcode_counts = {}
for barcode in barcode_values:
    if barcode in barcode_counts:
        barcode_counts[barcode] += 1
    else:
        barcode_counts[barcode] = 1

# Print the barcode counts
print("Barcode Counts:")
for barcode, count in barcode_counts.items():
    print(f"{barcode}: {count}")

# Draw the bounding boxes on the input image
for i, box in enumerate(boxes):
    x1, y1, x2, y2 = box
    bbox_color = BLACK
    bbox_label = None
    if len(barcode_values) > i and barcode_values[i] in barcode_counts:
        if barcode_counts[barcode_values[i]] > 1:
            bbox_label = f"{barcode_values[i]} ({barcode_counts[barcode_values[i]]})"
        else:
            bbox_label = barcode_values[i]
        bbox_color = BLUE
    else:
        bbox_label = "No Barcode"
        bbox_color = RED
    cv2.rectangle(img, (x1, y1), (x2, y2), bbox_color, thickness=2)
    cv2.putText(img, bbox_label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, bbox_color, 2)

# Display the input image
cv2.imshow("Input Image", img)
cv2.waitKey(0)
cv2.destroyAllWindows()
