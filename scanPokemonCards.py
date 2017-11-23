# import the necessary packages
from PIL import Image
import pytesseract
import argparse
import cv2
import os
 
import numpy as np

from matplotlib import pyplot as plt
from matplotlib import image as image

def applyThreshold(imageG):
    maxval = 255
    adaptiveMethod = cv2.ADAPTIVE_THRESH_MEAN_C #Use mean instead of sum of neighbourhood area
    thresholdType = cv2.THRESH_BINARY
    blockSize = 9 #5x5 region which is the size of the neighbourhood area
    C = 18 #Constant offset from the mean or weight calculated

    #Use adaptive threshold to get the signature
    cardNameMask = cv2.adaptiveThreshold(imageG,
            maxval,
            adaptiveMethod, 
            thresholdType,
            blockSize,
            C)

    return cardNameMask

def applyThreshold2(imageG):
    thresh = np.mean(image) + np.std(image)
    maxval = 255

    T, B = cv2.threshold(imageG,
        thresh,
        maxval, 
        cv2.THRESH_BINARY)

    return B

def lineApply(image):
    height, width = np.shape(image)[:2]

    #lowY = height
    #lowY2 = height
    #lowY3 = 0
    
    yAxisArray = []

    gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray,50,150,apertureSize = 3)

    lines = cv2.HoughLines(edges,1,np.pi/180,200)

    print lines
    print len(lines)


    for i in range(0, len(lines)):
        for rho,theta in lines[i]:
            a = np.cos(theta)
            b = np.sin(theta)
            x0 = a*rho
            y0 = b*rho
            x1 = int(x0 + 1000*(-b))
            y1 = int(y0 + 1000*(a))
            x2 = int(x0 - 1000*(-b))
            y2 = int(y0 - 1000*(a))

            if(x0 <= 0):

            #    if(y0 < lowY):
            #        lowY = y0

            #    if(lowY2 > lowY):
            #        highY = y0

                yAxisArray.append(y1)

            #print x0, y0, x1, y1, x2, y2
            #cv2.line(image,(x1,y1),(x2,y2),(0,0,255),2)

    yAxisArray.sort()

    #print lowY
    #print highY

    print yAxisArray

    cv2.imwrite('houghlines3.png',image)

    #return lowY, highY
    return yAxisArray[0], yAxisArray[2]

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True,
    help="path to input image to be OCR'd")
args = vars(ap.parse_args())

#Load the example image and convert it to grayscale
card = cv2.imread(args["image"])

height, width = np.shape(card)[:2]
height = height*2
width = width*2

card = cv2.resize(card, dsize = (width, height))

#Pastes the top part of the image
image2 = card[0: height/7, 0:width]

h1, h2 = lineApply(image2)

image = card[int(h1) + 5:int(h2), 0:width]

imageG = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

#Converts top part to gray scale
#imageG = cv2.cvtColor(card, cv2.COLOR_BGR2GRAY)

cardNameMask = applyThreshold(imageG)

#Apply morphological transformations to the card
shape = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))

#cleanImage = cv2.morphologyEx(cardNameMask, cv2.MORPH_OPEN, shape)

cleanImage = cv2.erode(cardNameMask, shape, iterations = 1)

#cleanImage2 = cv2.bitwise_and(image, image, mask = cleanImageN)
#cleanImage2 = cv2.cvtColor(cleanImage2, cv2.COLOR_BGR2GRAY)

#shape = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
#cleanImage = cv2.dilate(cardNameMask, shape, iterations = 1)

#shape = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (1, 1))
#cleanImage = cv2.erode(cardNameMask, shape, iterations = 1)

filename = "{}.png".format(os.getpid())
#cleanImage = cv2.resize(cleanImage, dsize = (width*2, height*2))
cv2.imwrite(filename, cleanImage)

#i, contours, hierarchy = cv2.findContours(cleanImage, cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)

#for contour in contours:
    # get rectangle bounding contour
    #[x, y, w, h] = cv2.boundingRect(contour)

    # Don't plot small false positives that aren't text
    #if w < 35 and h < 35:
        #continue

    # draw rectangle around contour on original image
    #cv2.rectangle(cleanImage, (x, y), (x + w, y + h), (255, 0, 0), 2)

#image = Image.open(filename)

text = pytesseract.image_to_string(Image.open(filename), lang = 'eng', config = 'pokemon')
os.remove(filename)
print(text)
#print(text2)

cv2.imshow('Top Border', cleanImage)
#cv2.imshow('Clean Mask', image)

cv2.waitKey(0)
cv2.destroyAllWindows()
