import cv2
from matplotlib import pyplot as plt
from pylab import rcParams

def edgeDetection(mImage):
       
    edges=cv2.Canny(mImage,50,40)
    plt.imshow(edges,cmap = 'gray')
    plt.title('Edge Image'), plt.xticks([]), plt.yticks([])
    return edges


print(cv2.__version__)

mImage=cv2.imread('mimage.jfif')

plt.imshow(mImage)


mImage_correctedColor=cv2.cvtColor(mImage,cv2.COLOR_BGR2RGB)

rcParams['figure.figsize'] = 10, 12


plt.imshow(mImage_correctedColor)
mImage_gray = cv2.cvtColor(mImage_correctedColor, cv2.COLOR_BGR2GRAY)

rcParams['figure.figsize'] = 10, 12
edgeDetection(mImage_gray)

#making it more insteresting

video_Capture=cv2.VideoCapture(0)


ret,frame=video_Capture.read()
video_Capture.release()

frameRGB = frame[:,:,::-1] # BGR => RGB
plt.subplot()
plt.imshow(frameRGB)
gray_frame=cv2.cvtColor(frameRGB, cv2.COLOR_RGB2GRAY)
edgeDetection(gray_frame)

plt.hist(frameRGB.ravel(),256,[0,256])

rcParams['figure.figsize'] = 8, 4

color = ('b','g','r')
for i,col in enumerate(color):
    histr = cv2.calcHist([frameRGB],[i],None,[256],[0,256])
    plt.plot(histr,color = col)
    plt.xlim([0,256])
plt.show()

newImage=cv2.equalizeHist(gray_frame)

plt.imshow(gray_frame)
clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
cl1 = clahe.apply(gray_frame)

edgedImage=edgeDetection(gray_frame)

cv2.imwrite('edgedImage.jpg',edgedImage)
cv2.imwrite('edgedImageBEtter.jpg',edgeDetection(cl1))

