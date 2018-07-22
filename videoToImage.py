import cv2
import matplotlib.pyplot as plt 
import time
import numpy as np
import os

vid=cv2.VideoCapture('videoTest.mp4')
sucess,images=vid.read()
success=True
count=1
print("Video read.")
while(count!=26):
	vid.set(cv2.CAP_PROP_POS_MSEC,count*1000)
	success,image=vid.read()
	cv2.imwrite("imagesSecondWise/no%d.jpg" %count,image)
	count+=1
	print("Frame number ",count," done.")
print(count,"number of frames extracted")
'''
img=cv2.imread("imagesSecondWise//no1.jpg")
height , width , layers = img.shape
video = cv2.VideoWriter('video.avi',-1,1,(width,height))

i=0
for file in os.listdir("imagesSecondWise/"):
	file="imagesSecondWise/"+file
	imageNew=cv2.imread(file)
	img=cv2.imread(file)
	grayImg=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
	imgCopy = img.copy()
	haarFaceCascade=cv2.CascadeClassifier('haar.xml')
	faces = haarFaceCascade.detectMultiScale(grayImg, scaleFactor=1.1, minNeighbors=5)
	print("Number of Faces Found :",len(faces))
	if(len(faces)!=0):
		for (x, y, w, h) in faces:
			cropped = imageNew[y:y+w, x:x+h]
			cv2.rectangle(imgCopy, (x, y), (x+w, y+h), (0, 255, 0), 2)	
			filename="faceNumber"+str(i)+".jpg"
			cv2.imwrite(file,imgCopy)
			i+=1
			video.write(imgCopy)'''