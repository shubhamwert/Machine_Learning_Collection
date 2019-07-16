import cv2
import sys
import numpy as np

def faceCapture():
    im=cv2.VideoCapture(0)


    ret,frame=im.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)



    cv2.imwrite("faceimage.jpg",gray)

    im.release()
    cv2.destroyAllWindows()
    return gray
    


faceCapture()