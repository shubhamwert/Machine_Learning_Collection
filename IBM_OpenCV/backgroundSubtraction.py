from __future__ import print_function

import cv2 as cv

import argparse
import time

backSub = cv.createBackgroundSubtractorMOG2()

cap=cv.VideoCapture(0)
while(True):
    ret,frame=cap.read()
   
    


    fgMask=backSub.apply(frame)
    cv.rectangle(frame, (10, 2), (100,20), (255,255,255), -1)
    cv.putText(frame, str(cap.get(cv.CAP_PROP_POS_FRAMES)), (15, 15),
               cv.FONT_HERSHEY_SIMPLEX, 0.5 , (0,0,0))
    
    
    cv.imshow('Frame', frame)
    cv.imshow('FG Mask', fgMask)
    
    keyboard = cv.waitKey(30)
    if keyboard == 'q' or keyboard == 27:
        break
    
cap.release()
cv.destroyAllWindows()

