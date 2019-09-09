import cv2 as cv
import numpy as np
import time
print(cv.__version__)
#detecting background

VideoF=cv.VideoCapture(0)


time.sleep(1)  
count = 0 
background = 0 
  
for i in range(60): 
    return_val, background = VideoF.read() 
    if return_val == False : 
        continue                               #bg saved
  
background = np.flip(background, axis = 1)
while(True):
    return_val, img = VideoF.read() 
    if not return_val : 
        break 
    count = count + 1
    img = np.flip(img, axis = 1)
    sen=15
    hsv = cv.cvtColor(img, cv.COLOR_BGR2HSV)   #conversion 
    # setting the lower and upper range for mask1 
    lower_red = np.array([60-sen, 40, 40])        
    upper_red = np.array([60+sen, 255, 255]) 
    mask1 = cv.inRange(hsv, lower_red, upper_red) 
    # setting the lower and upper range for mask2  
    lower_red = np.array([150, 40, 40]) 
    upper_red = np.array([180, 255, 255]) 
    mask2 = cv.inRange(hsv, lower_red, upper_red) 
    mask2 = cv.inRange(hsv, lower_red, upper_red)  


    mask1 = mask1 + mask2
    mask1 = cv.morphologyEx(mask1, cv.MORPH_OPEN, np.ones((3, 3), 
                                         np.uint8), iterations = 2) 
    mask1 = cv.dilate(mask1, np.ones((3, 3), np.uint8), iterations = 1) 
    mask2 = cv.bitwise_not(mask1) 
    res1 = cv.bitwise_and(background, background, mask = mask1) 
    res2 = cv.bitwise_and(img, img, mask = mask2) 
    final_output = cv.addWeighted(res1, 1, res2, 1, 0) 
  
    cv.imshow("INVISIBLE Cloack", final_output) 
    k = cv.waitKey(10) 
    if k == 27: 
        break

VideoF.release()
cv.destroyAllWindows()

