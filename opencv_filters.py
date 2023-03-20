import cv2
from cv2 import dnn_superres
from pyzbar.pyzbar import decode
import numpy as np
import argparse
import imutils

# Make one method to localize the barcode location
def localized(image):
    # load the image and convert it to grayscale
    image = cv2.imread(image)
    scale = 0.5
    width = int(image.shape[1] * scale)
    height = int(image.shape[0] * scale)
    image = cv2.resize(image, (width, height))
    cv2.imshow("orignal", image)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # compute the Scharr gradient magnitude representation of the images
    # in both the x and y direction using OpenCV 2.4
    ddepth = cv2.cv.CV_32F if imutils.is_cv2() else cv2.CV_32F
    gradX = cv2.Sobel(gray, ddepth=ddepth, dx=1, dy=0, ksize=-1)
    gradY = cv2.Sobel(gray, ddepth=ddepth, dx=0, dy=1, ksize=-1)
    # subtract the y-gradient from the x-gradient
    gradient = cv2.subtract(gradX, gradY)
    gradient = cv2.convertScaleAbs(gradient)
    blurred = cv2.blur(gradient, (9, 9))
    (_, thresh) = cv2.threshold(blurred, 225, 255, cv2.THRESH_BINARY)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (21, 7))
    closed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
    closed = cv2.erode(closed, None, iterations=4)
    closed = cv2.dilate(closed, None, iterations=4)
    # find the contours in the thresholded image, then sort the contours
    # by their area, keeping only the largest one
    cnts = cv2.findContours(closed.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    c = sorted(cnts, key=cv2.contourArea, reverse=True)[0]
    # compute the rotated bounding box of the largest contour
    rect = cv2.minAreaRect(c)
    box = cv2.cv.BoxPoints(rect) if imutils.is_cv2() else cv2.boxPoints(rect)
    box = np.int0(box)
    # draw a bounding box arounded the detected barcode and display the
    # image
    print(box)
    img = image.copy()
    mask = np.zeros((img.shape[0], img.shape[1]))
    cv2.fillConvexPoly(mask, box, 1)
    mask = mask.astype(bool)
    out = np.zeros_like(img)
    out[mask] = img[mask]
    rect = cv2.boundingRect(box)  # returns (x,y,w,h) of the rect
    cropped = out[rect[1] : rect[1] + rect[3], rect[0] : rect[0] + rect[2]]
    cv2.imshow("res", cropped)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    # methood to increase bar code resolution
    cv2.imwrite(r"BarCodeDetection/sample/allbarcode/IMG_20220303_173611.jpg", cropped)
    crop_res = cv2.imread(r"BarCodeDetection/sample/allbarcode/IMG_20220303_173611.jpg")
    sr = cv2.dnn_superres.DnnSuperResImpl_create()
    path = r"BarCodeDetection/sample/allbarcode/IMG_20220303_173611.jpg"
    sr.readModel(path)
    sr.setModel("espcn", 3)
    result_res = sr.upsample(crop_res)
    cv2.imshow("res", result_res)
    cv2.imwrite(
        r"BarCodeDetection/sample/allbarcode/IMG_20220303_173611.jpg", result_res
    )
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    result_res2 = cv2.imread(
        r"BarCodeDetection/sample/allbarcode/IMG_20220303_173611.jpg"
    )
    BarcodeReader(result_res2)


# methood to decode bar code
def BarcodeReader():
    # read the image in numpy array using cv2
    img = cv2.imread(
        r"BarCodeDetection/sample/allbarcode/IMG_20220303_173611.jpg", cropped
    )
    # img=cv2.imread(r"C:\Users\Aniket Verma\Desktop\BRCODE\zyro-image.png")
    # cv2.imshow("C", img)
    # Decode the barcode image
    detectedBarcodes = decode(img)
    print(detectedBarcodes)
    # If not detected then print the message
    if not detectedBarcodes:
        print("Barcode Not Detected or your barcode is blank/corrupted!")
    else:
        # Traverse through all the detected barcodes in image
        for barcode in detectedBarcodes:

            # Locate the barcode position in image
            (x, y, w, h) = barcode.rect

            # Put the rectangle in image using
            # cv2 to heighlight the barcode
            cv2.rectangle(
                img, (x - 10, y - 10), (x + w + 10, y + h + 10), (255, 0, 0), 5
            )

            if barcode.data != "":
                # Print the barcode data
                print(barcode.data)
                print(barcode.type)

    # Display the image
    # cv2.namedWindow("Image", cv2.WINDOW_NORMAL)
    # cv2.imshow("Image", img)

    cv2.waitKey(0)
    cv2.destroyAllWindows()


# methood to detect boxes based on contour
def getContours(image):
    img = cv2.imread(image)
    img = cv2.resize(img, (960, 540))
    imgContour = img.copy()
    imgBlur = cv2.GaussianBlur(img, (5, 5), 1)
    imgGray = cv2.cvtColor(imgBlur, cv2.COLOR_BGR2GRAY)
    imgCanny = cv2.Canny(imgGray, 167, 180)
    kernel = np.ones((5, 5))
    kernel2 = np.ones((3, 3))
    imgDil = cv2.dilate(imgCanny, kernel, iterations=1)
    imgEro = cv2.erode(imgDil, kernel2, iterations=1)
    contours, hierarchy = cv2.findContours(
        imgEro, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    for cnt in contours:
        area = cv2.contourArea(cnt)
        print(area)
        areaMin = 4000
        areaMax = 20000
        if area > areaMin and area < areaMax:
            cv2.drawContours(imgContour, cnt, -1, (255, 0, 255), 3)
            peri = cv2.arcLength(cnt, True)
            approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)
            x, y, w, h = cv2.boundingRect(approx)
            cv2.rectangle(imgContour, (x, y), (x + w, y + h), (0, 255, 0), 3)
    cv2.imshow("final", imgContour)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    # Take the image from user
    image = r"BarCodeDetection/sample/allbarcode/IMG_20220303_175324.jpg"
    getContours(image)
    localized(image)
    BarcodeReader()
