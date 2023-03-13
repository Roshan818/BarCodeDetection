# import necessary libraries
import cv2
import numpy as np
from pyzbar.pyzbar import decode
import torch

# load YOLOv5 object detection model
model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
model.conf = 0.5  # set confidence threshold

# read the input image
img = cv2.imread('BarCodeDetection/sample/allbarcode/IMG_20220303_173846.jpg')

# perform YOLOv5 object detection
results = model(img)

# initialize lists to store cropped images and barcode values
cropped_images = []
barcode_values = []

# loop through the detected objects and crop the images
for detection in results.xyxy[0]:
    class_id = int(detection[5])
    confidence = float(detection[4])
    if confidence > 0.5:
        x1, y1, x2, y2 = map(int, detection[:4])
        w = x2 - x1
        h = y2 - y1

        # crop the image and append it to the list
        cropped_img = img[y1:y2, x1:x2]
        cropped_images.append(cropped_img)

        # detect barcode in the cropped image
        gray = cv2.cvtColor(cropped_img, cv2.COLOR_BGR2GRAY)
        _, binary = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY)
        barcode = decode(binary)
        if len(barcode) > 0:
            barcode_values.append(barcode[0].data.decode())

# create a list of unique barcode values
unique_barcodes = list(set(barcode_values))

# loop through the unique barcode values and count their occurrences
for barcode in unique_barcodes:
    count = barcode_values.count(barcode)
    print(f'{barcode}: {count}')

# display the cropped images and detected barcodes
for i, cropped_img in enumerate(cropped_images):
    cv2.imshow(f'cropped image {i}', cropped_img)
    cv2.waitKey(0)

# close all windows
cv2.destroyAllWindows()
