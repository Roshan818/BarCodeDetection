import cv2
import numpy as np
from pyzbar.pyzbar import decode

class ObjectDetector:
    def __init__(self, model_path, config_path, classes_path):
        self.net = cv2.dnn.readNet(model_path, config_path)
        with open(classes_path, 'r') as f:
            self.classes = [line.strip() for line in f.readlines()]
        self.layer_names = self.net.getLayerNames()
        self.output_layers = [self.layer_names[i[0] - 1] for i in self.net.getUnconnectedOutLayers()]

    def detect_objects(self, image):
        # get image dimensions
        height, width, channels = image.shape

        # perform YOLO object detection
        blob = cv2.dnn.blobFromImage(image, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
        self.net.setInput(blob)
        outs = self.net.forward(self.output_layers)

        # initialize list to store cropped images
        cropped_images = []

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
                    cropped_img = image[y:y+h, x:x+w]
                    cropped_images.append(cropped_img)

        return cropped_images


class BarcodeScanner:
    def __init__(self):
        pass

    def detect_barcodes(self, images):
        # initialize list to store barcode values
        barcode_values = []

        # loop through the images and detect barcodes
        for image in images:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            _, binary = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY)
            barcode = decode(binary)
            if len(barcode) > 0:
                barcode_values.append(barcode[0].data.decode())

        # create a dictionary of barcode values and their occurrences
        barcode_dict = {}
        for barcode in barcode_values:
            if barcode in barcode_dict:
                barcode_dict[barcode] += 1
            else:
                barcode_dict[barcode] = 1

        return barcode_dict


class GroceryScanner:
    def __init__(self, model_path, config_path, classes_path):
        self.object_detector = ObjectDetector(model_path, config_path, classes_path)
        self.barcode_scanner = BarcodeScanner()

    def scan_groceries(self, image):
        # detect objects in the image
        cropped_images = self.object_detector.detect_objects(image)

        # scan barcodes in the cropped images
        barcode_dict = self.barcode_scanner.scan_barcodes(cropped_images)

        return barcode_dict
