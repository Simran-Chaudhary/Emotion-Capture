import cv2
import matplotlib.pyplot as plt 
import time
import numpy as np

imageNew=cv2.imread('no1.jpg')
img=cv2.imread("no1.jpg")
grayImg=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
imgCopy = img.copy()

cv2.imshow("Original Image",imageNew)
cv2.imshow("Gray Scale Image",grayImg)
cv2.waitKey(0)
cv2.destroyAllWindows()

haarFaceCascade=cv2.CascadeClassifier('haar.xml')
faces = haarFaceCascade.detectMultiScale(grayImg, scaleFactor=1.1, minNeighbors=5)

print("Number of Faces Found :",len(faces))

for (x, y, w, h) in faces:
	print(x,y,w,h)
	cropped = imageNew[y:y+w, x:x+h]
	print(x,x+w,y,y+w)
	#cv2.imshow("Cropped", cropped)
	#cv2.waitKey(0)
	#cv2.destroyAllWindows()	
	cv2.rectangle(imgCopy, (x, y), (x+w, y+h), (0, 255, 0), 2)	
	cv2.imwrite("face.jpg",imgCopy)
cv2.imshow("Faces",imgCopy)
cv2.waitKey(0)
cv2.destroyAllWindows()