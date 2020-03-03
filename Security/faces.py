import numpy as np
import cv2
import time
face_cascade = cv2.CascadeClassifier('cascades/data/haarcascade_frontalface_default.xml')
facex = ["brad" , "atul" , "leo", "Kohli"] 
facey = cv2.CascadeClassifier("cascade(1).xml")
cap = cv2.VideoCapture(2)
recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read("trainner.yml")
while(True):
    # Capture frame-by-frame
    ret, frame = cap.read()

    # Our operations on the frame come here
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor = 1.5,minNeighbors=6)
    faces_atul = facey.detectMultiScale(gray, scaleFactor = 1.2,minNeighbors=4)
    print(faces)
    for (x, y, w, h) in faces:
     	#print(x,y,w,h)
     	roi_gray = gray[y:y+h, x:x+h]
     	roi_color  = frame[y:y+h, x:x+h]
     	cv2.rectangle(frame, (x,y),(x+w,y+h),(0,0,255),2)
     	cv2.imwrite('img2.jpg',roi_gray)
     	img2 = cv2.imread('img2.jpg',0)
     	id_,conf = recognizer.predict(img2)
     	if conf>=80 and id_==1:
     		font = cv2.FONT_HERSHEY_SIMPLEX
     		strl = facex[1]+'   '+str(conf)
     		cv2.putText(frame,strl,(140,250), font, .5,(255,255,255),2,cv2.LINE_AA)
     		print(id_)

    # 	# if conf>=65  and id_==1:
    # 	# 	font = cv2.FONT_HERSHEY_SIMPLEX
    # 	# 	cv2.putText(frame,"Brad",(140,250), font, .5,(255,255,255),2,cv2.LINE_AA)
    # 	# 	print(id_)


    # # Display the resulting frame
    cv2.imshow('frame',frame)
    # # #cv2.imshow('gray',gray)
    if cv2.waitKey(20) & 0xFF == ord('q'):
    	break
    	id_,conf = recognizer.predict(img2)
    	print(id_,conf)
    	

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()