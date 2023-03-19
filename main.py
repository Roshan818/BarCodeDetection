import cv2
from barcode_scanner import BarcodeScanner

# define the image file path
image_path = "BarCodeDetection/sample/allbarcode/IMG_20220303_173846.jpg"

# read the input image
img = cv2.imread(image_path)

# create an instance of the BarcodeScanner class
scanner = BarcodeScanner()

# perform barcode detection and scanning
scanner.detect_barcodes(img)
scanner.scan_barcodes()

# display the output image with bounding boxes
output_img = scanner.draw_bounding_boxes(img)
cv2.imshow("Output Image", output_img)
cv2.waitKey(0)
cv2.destroyAllWindows()
