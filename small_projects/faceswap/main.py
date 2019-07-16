import cv2
import sys
import numpy as np

img=cv2.imread("faceimage.jpg",0)

faceCascade=cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
faces = faceCascade.detectMultiScale(img, 1.3, 5)
print(len(faces))
for (x, y, w, h) in faces:
    cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)
cv2.imshow("before swapping", img)
cv2.waitKey(0)

face1=faces[1]
face2=faces[2]






cv2.waitKey(0)
cv2.destroyAllWindows()