import cv2
import numpy as np
from ritesh import ObjectDetector, BarcodeDetector, GroceryScanner


def main():
    # initialize object detector
    weights_path = "yolov3.weights"
    config_path = "yolov3.cfg"
    class_names_path = "coco.names"
    object_detector = ObjectDetector(weights_path, config_path, class_names_path)

    # initialize barcode detector
    barcode_detector = BarcodeDetector()

    # initialize grocery scanner
    grocery_scanner = GroceryScanner(object_detector, barcode_detector)

    # read image
    img = cv2.imread("groceries.jpg")

    # scan groceries
    cropped_images, barcode_values = grocery_scanner.scan_groceries(img)

    # count unique barcodes
    barcode_counts = grocery_scanner.count_unique_barcodes(barcode_values)

    # visualize groceries
    for i, cropped_img in enumerate(cropped_images):
        bbox = object_detector.detect_objects(cropped_img)[0][0]
        grocery_scanner.draw_bbox(img, bbox, (0, 255, 0), 2)
        barcode_value = barcode_values[i]
        cv2.putText(
            img,
            f"{barcode_value} ({barcode_counts[barcode_value]})",
            (bbox[0], bbox[1] - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 255, 0),
            2,
        )

    # show image
    cv2.imshow("Groceries", img)
    cv2.waitKey(0)


if __name__ == "__main__":
    main()
