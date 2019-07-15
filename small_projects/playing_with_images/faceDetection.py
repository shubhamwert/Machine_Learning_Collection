import cv2

cap=cv2.VideoCapture(0)
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

while True:
    ret,img=cap.read()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(img, 1.3, 5)
    n_image=cv2.convertScaleAbs(img,alpha=0.7,beta=2)

    for (x,y,w,h) in faces:
        cv2.rectangle(n_image,(x,y),(x+w,y+h),(255,0,0),2)
        
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = n_image[y:y+h, x:x+w]
    cv2.imshow('img',n_image)

    k = cv2.waitKey(30) & 0xff
        
    if k == 'q':
        break
    #zooming on faces

cap.release()
cv2.destroyAllWindows()        
