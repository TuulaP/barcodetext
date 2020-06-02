# USAGE
# python dete2.py --image images/barcode_01.jpg

# import the necessary packages
from pyzbar import pyzbar
import numpy as np
import argparse
import imutils
import cv2


def detectBarcode(barcodes):
    if len(barcodes) > 0:
        barcode = barcodes[0]
    else:
        return None
    # the barcode data is a bytes object so if we want to draw it on
    # our output image we need to convert it to a string first
    barcodeData = barcode.data.decode("utf-8")
    barcodeType = barcode.type
    (x, y, w, h) = barcode.rect
    #cv2.rectangle(crop_img, (x, y), (x + w, y + h), (0, 0, 255), 2)

    # draw the barcode data and barcode type on the image
    text = "{} ({})".format(barcodeData, barcodeType)
    cv2.putText(crop_img, text, (x, y - 50), cv2.FONT_HERSHEY_SIMPLEX,
                1, (0, 0, 255), 2)

    cv2.imwrite('barb.png', crop_img)
    print("text of barcode:", text)
    return text


# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True,
                help="path to the image file")
args = vars(ap.parse_args())

# load the image and convert it to grayscale
image = cv2.imread(args["image"])
orig = image.copy()
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# compute the Scharr gradient magnitude representation of the images
# in both the x and y direction using OpenCV 2.4
ddepth = cv2.cv.CV_32F if imutils.is_cv2() else cv2.CV_32F
gradX = cv2.Sobel(gray, ddepth=ddepth, dx=1, dy=0, ksize=-1)
gradY = cv2.Sobel(gray, ddepth=ddepth, dx=0, dy=1, ksize=-1)

# subtract the y-gradient from the x-gradient
gradient = cv2.subtract(gradX, gradY)
gradient = cv2.convertScaleAbs(gradient)

# blur and threshold the image
blurred = cv2.blur(gradient, (9, 9))
(_, thresh) = cv2.threshold(blurred, 225, 255, cv2.THRESH_BINARY)

# construct a closing kernel and apply it to the thresholded image
kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (21, 7))
closed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)

# perform a series of erosions and dilations
closed = cv2.erode(closed, None, iterations=4)
closed = cv2.dilate(closed, None, iterations=4)

# find the contours in the thresholded image, then sort the contours
# by their area, keeping only the largest one
cnts = cv2.findContours(closed.copy(), cv2.RETR_EXTERNAL,
                        cv2.CHAIN_APPROX_SIMPLE)

# print(cnts)

cnts = imutils.grab_contours(cnts)
c = sorted(cnts, key=cv2.contourArea, reverse=True)[0]

# compute the rotated bounding box of the largest contour
rect = cv2.minAreaRect(c)
box = cv2.cv.BoxPoints(rect) if imutils.is_cv2() else cv2.boxPoints(rect)
box = np.int0(box)

basefix = 100

# some manual adjustments for box sizing...
box[0][0] -= basefix*3
box[1][0] -= basefix*3
box[2][0] += basefix*4
box[3][0] += basefix*4


cv2.drawContours(image, [box], -1, (0, 255, 0), 3)
# cv2.imshow("Image", image)
# cv2.waitKey(0)

cv2.imwrite('result.png', image)

#
crop_img = orig[box[2][1]-basefix*3:box[2][1]+600,
                box[1][0]-basefix*2:box[1][0]+basefix*3+500]

# crop_img = image[box[1][1]:box[1][1]+500, box[1][0]:box[1][0]+500]
# crop2 = image[box]
# box[0][1]:box[1][1]+200, box[0][0]: box[1][0]+200]

cv2.imwrite('bar.png', crop_img)
# cv2.imwrite('b.png', crop2)

# magick convert bar.png -deskew 40% bar2.png

# import the necessary packages
# find the barcodes in the image and decode each of the barcodes
# load the input image
crop_img = cv2.imread('bar.png')

barcodes = pyzbar.decode(crop_img)

res = detectBarcode(barcodes)

if res is None:
    print("Sorry, no barcode detected...")
    crop_img = cv2.imread('bar2.png')
    barcodes = pyzbar.decode(crop_img)

    # deskew
    barcode = detectBarcode(barcodes)
    print("Barcode: ", barcode)
