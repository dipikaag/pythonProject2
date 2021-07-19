# Import libraries

import numpy
import cv2
import time

import numpy as np
# waking camera to run
cap = cv2.VideoCapture(0)
# buffer time for camera to start
time.sleep(3)
# bg variable will store initial frame to be displayed
background=0
# capturing background here
while (True):
    cv2.waitKey(1000)
    ret, background = cap.read()
    # check if the frame is returned then brake
    if (ret):
        break
 # capturing image to be cloaked
while(cap.isOpened()):

    ret,img = cap.read()

    if not ret:
        break
    hsv = cv2.cvtColor(img,cv2.COLOR_BGR2HSV) #hue saturation vale, hue is responsible for color
    #HSV_values
    lower_hsv=np.array([0,120,70])
    upper_hsv=np.array([10,255,255])
    mask1 =cv2.inRange(hsv,lower_hsv,upper_hsv) #separating the cloak part

    lower_hsv = np.array([170, 120, 70])
    upper_hsv = np.array([180, 255, 255])
    mask2 = cv2.inRange(hsv, lower_hsv, upper_hsv)

    mask1 = mask1 + mask2 #bitwise OR to add both the ranges to mask1
    #defining a matrix(kernel) for image dialation
    matrix = np.ones((5,5),np.uint8)
    #smoothening the image and noise removal
    mask1= cv2.morphologyEx(mask1,cv2.MORPH_OPEN,matrix,iterations=5)
    #mask1 = cv2.morphologyEx(mask1, cv2.MORPH_DILATE,
                            # matrix, iterations=5)
    mask2 = cv2.bitwise_not(mask1)

    res1 = cv2.bitwise_and(background,background, mask=mask1) # used for segmentation of the color
    res2= cv2.bitwise_and(img,img,mask= mask2) # used to substitute the cloak part
    final_output = cv2.addWeighted(res1,1,res2,1,0)

    cv2.imshow('HARRY_POTTER',final_output)
    cv2.imshow("Original",img)
    k= cv2.waitKey(10)
    if k==27:
        break
cap.release()
cv2.destroyAllWindows()