# BarCode Detection

<img src = "https://socialify.git.ci/Roshan818/BarCodeDetection/image?description=1&descriptionEditable=Mowito%27s%20Barcode%20Finding%20and%20Reading%20Software&font=KoHo&language=1&name=1&owner=1&pattern=Solid&stargazers=1&theme=Dark">

---

## Introduction

This is a solution for Mowito's <strong>Barcode Finding and Reading Software</strong> aiming to automate the number checkpout of items at the store.

Given a picture, I developed a system which identifies the number of objects present in the picture along with the barcode on each object. Further to expand it also identifies the missing and incomplete barcodes. The system is developed using Python and OpenCV and we have integrated yoloV5 for object detection and pyzbar for barcode detection.

---

## Input Images

- An image containing multiple grocery items, with a background of single color.
- For some items, the barcode will be visible, and for others, it might not be visible.
- Some of the items might be repeated.

### Sample Input Image

<img src ="sample\allbarcode\IMG_20220303_173846.jpg" height = 350>

---

## Output

The output contains the following information:

- The value of every barcode, and number of times every barcode appears in the image
- Bounding box of blue color on the input image, around all the barcodes which are correctly read.
- Bounding box of yellow color on the input image around the barcodes which are partially visible and couldn’t be completely read.
- Bounding box of red color on the input image around items that did not have any barcodes.
- Bounding boxes of black color around each ITEM in the image.

### Sample Output Image

<img src ="sample\allbarcode\IMG_20220303_173846.jpg" height = 350>

---

## How to run the code

### Prerequisites

- python 3.9
- pipenv

### Setup

#### Pyzbar Setup

```bash
# Pyzbar Setup
pip install python-barcode[images]
pip install qrcode[pil]
sudo apt-get install libzbar0
pip install pyzbar

# Opencv Setup
pip install opencv-python
pip install opencv-contrib-python

pip install imutils
```

#### Yolov5 Setup

```bash
git clone https://github.com/ultralytics/yolov5  # clone
cd yolov5
pip install -r requirements.txt  # install
```
