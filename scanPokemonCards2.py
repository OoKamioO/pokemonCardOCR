# import the necessary packages
from PIL import Image
import pytesseract
import argparse
import cv2
import os
 
import numpy as np

from matplotlib import pyplot as plt
from matplotlib import image as image

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True,
	help="path to input image to be OCR'd")
ap.add_argument("-p", "--preprocess", type=str, default="thresh",
	help="type of preprocessing to be done")
args = vars(ap.parse_args())

card = cv2.imread(args["image"])
height, width = np.shape(card)[:2]
height = height*3
width = width*3

card = cv2.resize(card, dsize = (width, height))

image = card[0: height/8, 0:width]
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
 
# check to see if we should apply thresholding to preprocess the
# image
if args["preprocess"] == "thresh":
	gray = cv2.threshold(gray, 0, 255,
		cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
 
# make a check to see if median blurring should be done to remove
# noise
elif args["preprocess"] == "blur":
	gray = cv2.medianBlur(gray, 3)
 
# write the grayscale image to disk as a temporary file so we can
# apply OCR to it
filename = "{}.png".format(os.getpid())
cv2.imwrite(filename, gray)

# load the image as a PIL/Pillow image, apply OCR, and then delete
# the temporary file
text = pytesseract.image_to_string(Image.open(filename))
os.remove(filename)
print(text)
 
# show the output images
cv2.imshow("Image", gray)
cv2.waitKey(0)
cv2.destroyAllWindows()