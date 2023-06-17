import cv2 as cv
from cvzone.HandTrackingModule import HandDetector
import numpy as np
import math
import time

cap = cv.VideoCapture(1)
detector = HandDetector(maxHands=1)

offset = 20
imgsize = 300
folder = 'C:/Users/scs/Desktop/fathallh/Data/z'
counter = 0

while True:
    success, img = cap.read()
    hands, img = detector.findHands(img)

    if hands:
        hand = hands[0]
        x, y, w, h = hand['bbox']
        imgcrop = img[y-offset:y + h + offset, x-offset: x + w + offset]

        imgwhite = np.ones((imgsize, imgsize, 3), np.uint8)*255
        aspectratio = h/w
        if aspectratio > 1:
            k = imgsize / h
            wcal = math.ceil(k*w)
            imgresize = cv.resize(imgcrop,(wcal, imgsize))
            wgap = math.ceil((imgsize-wcal)/2)
            imgresizeshape = imgresize.shape
            imgwhite[:, wgap:wcal + wgap] = imgresize
        else:
            # Detect the Position of Hand in White Box
            k = imgsize / w
            hcal = math.ceil(k*h)
            imgresize = cv.resize(imgcrop,(imgsize ,hcal))
            hgap = math.ceil((imgsize-hcal)/2)
            imgresizeshape = imgresize.shape
            imgwhite[hgap:hcal + hgap, :] = imgresize

        cv.imshow('Imagecrop', imgcrop)
        cv.imshow('Imagewhite', imgwhite)

    cv.imshow('Image', img)
    
    key = cv.waitKey(1)
    if key == ord('s'):
        counter += 1
        cv.imwrite(f'{folder}/Image_{time.time()}.jpg', imgwhite)
        print(counter)

    elif key & 0xFF == 27:
        break

cv.destroyAllWindows()
