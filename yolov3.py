import cv2
import numpy as np
from pyzbar.pyzbar import decode


class YoloDetector:
    """
    Yolo Detector class
    """
    def __init__(self, config_path, weights_path, names_path):
        """
        Initialize the Yolo Detector
        :param config_path: Path to the yolo config file
        :param weights_path: Path to the yolo pre-trained weights
        :param names_path: Path to the text file containing the names of the classes
        """
        self.net = cv2.dnn.readNet(weights_path, config_path)
        with open(names_path, 'r') as f:
            self.classes = [line.strip() for line in f.readlines()]
        self.layer_names = self.net.getLayerNames()
        self.output_layers = [self.layer_names[i[0] - 1] for i in self.net.getUnconnectedOutLayers()]

        self.output_layers = []
        out_layers = self.net.getUnconnectedOutLayers()
        if len(out_layers) > 0:
            self.output_layers = [self.layer_names[i[0] - 1] for i in out_layers]


    def detect_objects(self, img):
        """
        Detect objects in the input image
        :param img: Input image
        :return: height, width, outs
        """
        height, width, channels = img.shape
        blob = cv2.dnn.blobFromImage(img, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
        self.net.setInput(blob)
        outs = self.net.forward(self.output_layers)
        return height, width, outs


class BarcodeDetector:
    """
    Barcode Detector class
    """
    def __init__(self, threshold=150):
        """
        Initialize the Barcode Detector
        :param threshold: Threshold value for the image
        """
        self.threshold = threshold

    def detect_barcodes(self, cropped_images):
        """
        Detect barcodes in the input image
        :param cropped_images: Input image
        :return: barcode_values
        """
        barcode_values = []
        for cropped_img in cropped_images:
            gray = cv2.cvtColor(cropped_img, cv2.COLOR_BGR2GRAY)
            _, binary = cv2.threshold(gray, self.threshold, 255, cv2.THRESH_BINARY)
            barcode = decode(binary)
            if len(barcode) > 0:
                barcode_values.append(barcode[0].data.decode())
            else:
                barcode_values.append(None)
        return barcode_values


class GroceryDetector:
    """
    Grocery Detector class
    """
    def __init__(self, yolo_detector, barcode_detector):
        """
        Initialize the Grocery Detector
        :param yolo_detector: Yolo Detector object
        :param barcode_detector: Barcode Detector object
        """
        self.yolo_detector = yolo_detector
        self.barcode_detector = barcode_detector

    def detect_grocery_items(self, img):
        """
        Detect grocery items in the input image
        :param img: Input image
        :return: boxes, confidences, class_ids, barcode_values
        """
        height, width, outs = self.yolo_detector.detect_objects(img)
        cropped_images = []
        boxes = []
        confidences = []
        class_ids = []
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

                    cropped_img = img[y:y+h, x:x+w]
                    cropped_images.append(cropped_img)
                    boxes.append([x, y, w, h])
                    confidences.append(float(confidence))
                    class_ids.append(class_id)

        barcode_values = self.barcode_detector.detect_barcodes(cropped_images)
        return boxes, confidences, class_ids, barcode_values


class GroceryAnalyzer:
    """
    Grocery Analyzer class
    """
    def __init__(self, grocery_detector):
        """
        Initialize the Grocery Analyzer
        :param grocery_detector: Grocery Detector object
        """
        self.grocery_detector = grocery_detector

    def analyze_grocery_items(self, img):
        """
        Analyze grocery items in the input image
        :param img: Input image
        :return: boxes, confidences, class_ids, barcode_values
        """
        boxes, confidences, class_ids, barcode_values = self.grocery_detector.detect_grocery_items(img)
        unique_barcodes = list(set(barcode_values))
        print('Barcode Values and Counts')
        for barcode in unique_barcodes:
            if barcode is not None:
                count = barcode_values.count(barcode)
                print(f'{barcode}: {count}')
            else:
                print('None: 1')

        return boxes, confidences, class_ids, barcode_values


class GroceryVisualizer:
    """
    Grocery Visualizer class
    """
    def __init__(self, grocery_analyzer):
        """
        Initialize the Grocery Visualizer
        :param grocery_analyzer: Grocery Analyzer object
        """
        self.grocery_analyzer = grocery_analyzer

    def visualize_grocery_items(self, img):
        """
        Visualize grocery items in the input image
        :param img: Input image
        :return: img
        """
        boxes, confidences, class_ids, barcode_values = self.grocery_analyzer.analyze_grocery_items(img)
        indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)
        font = cv2.FONT_HERSHEY_PLAIN
        colors = np.random.uniform(0, 255, size=(len(boxes), 3))
        for i in range(len(boxes)):
            if i in indexes:
                x, y, w, h = boxes[i]
                label = str(barcode_values[i])
                color = colors[i]
                cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)
                cv2.putText(img, label, (x, y + 30), font, 3, color, 3)
        return img


def main():
    """
    Main function
    :return: None
    """
    yolo_detector = YoloDetector('D:/Mowito/darknet/cfg/yolov3.cfg', 'D:/Mowito/yolov3.weights', 'D:/Mowito/darknet/data/coco.names')
    barcode_detector = BarcodeDetector()
    grocery_detector = GroceryDetector(yolo_detector, barcode_detector)
    grocery_analyzer = GroceryAnalyzer(grocery_detector)
    grocery_visualizer = GroceryVisualizer(grocery_analyzer)

    img = cv2.imread('images/grocery.jpg')
    img = grocery_visualizer.visualize_grocery_items(img)
    cv2.imshow('Image', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()

