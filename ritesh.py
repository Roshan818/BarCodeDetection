import cv2
import numpy as np
from pyzbar.pyzbar import decode


class ObjectDetector:
    def __init__(self, weights_path, config_path, class_names_path):
        self.net = cv2.dnn.readNet(weights_path, config_path)
        with open(class_names_path, "r") as f:
            self.classes = [line.strip() for line in f.readlines()]
        self.layer_names = self.net.getLayerNames()
        self.output_layers = [
            self.layer_names[i[0] - 1] for i in self.net.getUnconnectedOutLayers()
        ]

    def detect_objects(self, img):
        height, width, channels = img.shape
        blob = cv2.dnn.blobFromImage(
            img, 0.00392, (416, 416), (0, 0, 0), True, crop=False
        )
        self.net.setInput(blob)
        outs = self.net.forward(self.output_layers)
        class_ids = []
        confidences = []
        boxes = []
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
                    boxes.append([x, y, w, h])
                    confidences.append(float(confidence))
                    class_ids.append(class_id)
        return boxes, confidences, class_ids


class BarcodeDetector:
    def __init__(self):
        pass

    def detect_barcodes(self, img):
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        _, binary = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY)
        barcode = decode(binary)
        if len(barcode) > 0:
            return barcode[0].data.decode()
        else:
            return None


class GroceryScanner:
    def __init__(self, object_detector, barcode_detector):
        self.object_detector = object_detector
        self.barcode_detector = barcode_detector

    def scan_groceries(self, img):
        boxes, confidences, class_ids = self.object_detector.detect_objects(img)
        cropped_images = []
        barcode_values = []
        for i, box in enumerate(boxes):
            x, y, w, h = box
            cropped_img = img[y : y + h, x : x + w]
            cropped_images.append(cropped_img)
            if class_ids[i] == 0:  # check if the object is a barcode
                barcode_value = self.barcode_detector.detect_barcodes(cropped_img)
                if barcode_value is not None:
                    barcode_values.append(barcode_value)
        return cropped_images, barcode_values

    @staticmethod
    def count_unique_barcodes(barcode_values):
        unique_barcodes = list(set(barcode_values))
        barcode_counts = {}
        for barcode in unique_barcodes:
            count = barcode_values.count(barcode)
            barcode_counts[barcode] = count
        return barcode_counts

    @staticmethod
    def draw_bbox(img, bbox, color, thickness):
        x, y, w, h = bbox
        cv2.rectangle(img, (x, y), (x + w, y + h), color, thickness)

    # def visualize_groceries(self, img, cropped_images, barcode_values):
