import cv2
from barcode_scanner import GroceryScanner

def main():
    # initialize the grocery scanner
    model_path = 'yolov5s.pt'
    config_path = 'path/to/config/file'
    classes_path = 'path/to/classes/file'
    grocery_scanner = GroceryScanner(model_path, config_path, classes_path)

    # load the image
    image_path = 'path/to/image/file'
    image = cv2.imread(image_path)

    # scan the groceries in the image
    barcode_dict = grocery_scanner.scan_groceries(image)

    # print the barcode values and their occurrences
    for barcode, count in barcode_dict.items():
        print(f'{barcode}: {count}')

if __name__ == '__main__':
    main()
