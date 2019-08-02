import cv2
import matplotlib.pyplot as plt
import numpy as np
first_frame=None
cap=cv2.VideoCapture(0)
if not cap.isOpened():
    raise IOError
while True:
    ret,frame=cap.read()
    rows,cols = frame.shape[:2]
    face_cascade1 = cv2.CascadeClassifier('C:\\opencv\\data\\haarcascades\\haarcascade_frontalface_default.xml')
    gray1=cv2.cvtColor(frame,cv2.COLOR_RGB2GRAY)
    faces = face_cascade1.detectMultiScale(gray1, scaleFactor=1.03, minNeighbors=5)
    for (x, y, w, h) in faces:cv2.rectangle(frame, (x, y), (x+w, y+h), (50, 50, 200), 2)
    c=cv2.waitKey(1)
    if c==27:
        break
    cv2.imshow("result",frame)
cap.release()
cv2.destroyAllWindows()
