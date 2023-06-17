import cv2 as cv
from cvzone.HandTrackingModule import HandDetector
import numpy as np
from cvzone.ClassificationModule import Classifier
import math

cap = cv.VideoCapture(1)
detector = HandDetector(maxHands=1)
Classifier = Classifier('C:/Users/scs/Desktop/Mohamed_Raslan/data_scince_track/projects/fathallh/Model/keras_model.h5',
                        'C:/Users/scs/Desktop/Mohamed_Raslan/data_scince_track/projects/fathallh/Model/labels.txt')
offset = 20
imgsize = 300
folder = 'C:/Users/scs/Desktop/fathallh/Data/Z'
counter = 0
labels = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L',
          'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W',  'X', 'Y', 'Z']

while True:
    success, img = cap.read()
    imgoutput = img.copy()
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
            imgresize = cv.resize(imgcrop, (wcal, imgsize))
            wgap = math.ceil((imgsize-wcal)/2)
            imgresizeshape = imgresize.shape
            imgwhite[:, wgap:wcal + wgap] = imgresize
            prediction, index = Classifier.getPrediction(imgwhite)
            print(prediction, index)

        else:
            k = imgsize / w
            hcal = math.ceil(k*h)
            imgresize = cv.resize(imgcrop, (imgsize, hcal))
            hgap = math.ceil((imgsize-hcal)/2)
            imgresizeshape = imgresize.shape
            imgwhite[hgap:hcal + hgap, :] = imgresize
            prediction, index = Classifier.getPrediction(imgwhite)

        cv.rectangle(imgoutput, (x - offset, y - offset - 50),
                     (x - offset + 90, y - offset - 50 + 50), (255, 0, 255), cv.FILLED)
        cv.putText(imgoutput, labels[index], (x, y-26), cv.FONT_HERSHEY_COMPLEX, 1.7, (255, 255, 255), 2)
        cv.rectangle(imgoutput, (x - offset,y - offset), (x + w + offset, y + h + offset), (255, 0, 255), 4)

        cv.imshow('Imagecrop', imgcrop)
        cv.imshow('Imagewhite', imgwhite)

    cv.imshow('Image', imgoutput)
    key = cv.waitKey(1)
    if key & 0xFF == 27:
        break
cv.destroyAllWindows()





