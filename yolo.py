# import necessary libraries
import cv2
import numpy as np
from pyzbar.pyzbar import decode

# load YOLO object detection model
net = cv2.dnn.readNet('yolo.weights', 'yolo.cfg')
classes = []
with open('yolo.names', 'r') as f:
    classes = [line.strip() for line in f.readlines()]
layer_names = net.getLayerNames()
output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]

# read the input image
img = cv2.imread('BarCodeDetection/sample/allbarcode/IMG_20220303_173611.jpg')

# get image dimensions
height, width, channels = img.shape

# perform YOLO object detection
blob = cv2.dnn.blobFromImage(img, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
net.setInput(blob)
outs = net.forward(output_layers)

# initialize lists to store cropped images and barcode values
cropped_images = []
barcode_values = []

# loop through the detected objects and crop the images
for out in outs:
    for detection in out:
        scores = detection[5:]
        class_id = np.argmax(scores)
        confidence = scores[class_id]
        if confidence > 0.5:
            center_x = int(detection[0] * width)
            center_y = int(detection[1] * height)
            w = int(detection[2] * width)
            h = int(detection[3] * height)
            x = int(center_x - w / 2)
            y = int(center_y - h / 2)

            # crop the image and append it to the list
            cropped_img = img[y:y+h, x:x+w]
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
