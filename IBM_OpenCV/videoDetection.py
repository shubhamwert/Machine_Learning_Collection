import cv2 
import numpy as np
from matplotlib import pyplot as plt
def edgeDetection(mImage):
       
    edges=cv2.Canny(mImage,50,40)
    plt.imshow(edges,cmap = 'gray')
    plt.title('Edge Image'), plt.xticks([]), plt.yticks([])
    return edges

cap=cv2.VideoCapture(0)

while(True):
    ret,frame=cap.read()
    cv2.imshow('frame',frame)
    
    gray=cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    cv2.imshow('frame',edgeDetection(gray))
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()

background = np.flip(background, axis = 1)