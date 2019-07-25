import cv2
import numpy as np
try:
    
    
    img=cv2.imread('bunny.png')

    blurredImg=cv2.blur(img,(10,10))

    cv2.imshow('Averaging',blurredImg) 
    cv2.waitKey(0)

    gausBlur = cv2.GaussianBlur(img, (5,5),0)  
    cv2.imshow('Gaussian Blurring', gausBlur) 
    cv2.waitKey(0) 


    cv2.waitKey(0) 

except TypeError as identifier:
    print("ERROR")
finally:
    cv2.destroyAllWindows()
cv2.destroyAllWindows()
