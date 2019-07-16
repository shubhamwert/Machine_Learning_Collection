import cv2
import numpy as np

img=cv2.imread('bunny.png')

blurredImg=cv2.blur(img,(10,10))

cv2.imshow('Averaging',blurredImg) 
cv2.waitKey(0)

gausBlur = cv2.GaussianBlur(img, (5,5),0)  
cv2.imshow('Gaussian Blurring', gausBlur) 
cv2.waitKey(0) 


mask = np.zeros((512, 512, 3), dtype=np.uint8)
mask = cv2.circle(mask, (258, 258), 100, np.array([255, 255, 255]), -1)

out = np.where(mask==np.array([255, 255, 255]), img, gausBlur)

cv2.imshow('Gaussian Blurring', out) 
cv2.waitKey(0) 

cv2.destroyAllWindows()
