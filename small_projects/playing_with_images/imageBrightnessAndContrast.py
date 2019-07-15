import cv2 as cv
from matplotlib import pyplot as plt
import numpy as np
import sys
img=cv.imread('bunny.png')
img=cv.cvtColor(img,cv.COLOR_BGR2RGB)
img2=np.asarray(img)
img2=(img2-np.mean(img2))/np.std(img2,axis=0)
plt.imshow(img2)

plt.hist(img.ravel(),256,[0,256])

for a in range(1,10):
    plt.figure(a)
    plt.subplot(211)
    new_image=cv.convertScaleAbs(img,alpha=float(a/10),beta=0.5)
    plt.imshow(new_image)
plt.show() 

    
