# import necessary libraries
import cv2
import numpy as np
from pyzbar.pyzbar import decode

# read the input image
img = cv2.imread('sample/allbarcode/IMG_20220303_173611.jpg')

# convert the image to grayscale
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# apply a threshold to the grayscale image
_, binary = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY)

# find contours in the binary image
contours, _ = cv2.findContours(binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

# initialize lists to store barcode values and bounding box colors
barcode_values = []
bounding_box_colors = []

# loop through the contours and check if it contains a barcode
for contour in contours:
    # get the bounding box of the contour
    x, y, w, h = cv2.boundingRect(contour)

    # decode the barcode
    barcode = decode(binary[y:y+h, x:x+w])
    
    # check if the barcode is readable
    if len(barcode) > 0:
        # store the barcode value
        barcode_values.append(barcode[0].data.decode())
        # draw a blue bounding box around the barcode
        bounding_box_colors.append((255, 0, 0))
        cv2.rectangle(img, (x, y), (x+w, y+h), (255, 0, 0), 2)
    else:
        # if the barcode is not readable, draw a yellow bounding box around it
        bounding_box_colors.append((0, 255, 255))
        cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 255), 2)

# create a list of unique barcode values
unique_barcodes = list(set(barcode_values))

# loop through the unique barcode values and count their occurrences
for barcode in unique_barcodes:
    count = barcode_values.count(barcode)
    print(f'{barcode}: {count}')

# loop through the contours again and draw bounding boxes around the items
for i, contour in enumerate(contours):
    if bounding_box_colors[i] != (255, 0, 0) and bounding_box_colors[i] != (0, 255, 255):
        cv2.rectangle(img, (x, y), (x+w, y+h), (0, 0, 255), 2)
    cv2.rectangle(img, (x, y), (x+w, y+h), (0, 0, 0), 2)

# display the image
cv2.imshow('output', img)
cv2.waitKey(0)
cv2.destroyAllWindows()
