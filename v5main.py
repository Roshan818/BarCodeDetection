import cv2
from v5modular import YoloDetector, BarcodeDetector, GroceryDetector, GroceryAnalyzer

# Initialize the detectors
yolo_detector = YoloDetector(config_path='yolo_config.cfg', weights_path='yolo_weights.weights', names_path='classes.txt')
barcode_detector = BarcodeDetector(threshold=150)
grocery_detector = GroceryDetector(yolo_detector=yolo_detector, barcode_detector=barcode_detector)
grocery_analyzer = GroceryAnalyzer(grocery_detector=grocery_detector)

# Load the input image
img = cv2.imread('groceries.jpg')

# Analyze the groceries in the image
boxes, confidences, class_ids, barcode_values = grocery_analyzer.analyze_grocery_items(img)

# Visualize the results
for i in range(len(boxes)):
    x, y, w, h = boxes[i]
    cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
    cv2.putText(img, str(barcode_values[i]), (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

cv2.imshow('Grocery Analyzer', img)
cv2.waitKey(0)
cv2.destroyAllWindows()
