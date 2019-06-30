import cv2
import numpy as np
import os
import matplotlib.pyplot as plt
from video_capture import videoCamera
from face_detection import faceDetector
from face_normalization import normalization

cap=videoCamera()
face_detect=faceDetector("./DATA/haarcascades/haarcascade_frontalface_alt.xml")
norm=normalization()

folder="student/" + input("name : ").lower()
cv2.namedWindow('create_data',cv2.WINDOW_AUTOSIZE)
if not os.path.exists(folder):
    os.mkdir(folder)
    counter=0
    timer=0
    while counter<20:
        frame=cap.get_frame()
        faces_coord=face_detect.detect(frame)
        if len(faces_coord) and timer % 700==50:
            faces=norm.face_normalization(frame,faces_coord)
            cv2.imwrite(folder+"/"+str(counter)+".jpg",faces[0])
            #plt_show(faces[0],"image saved:"+str(counter))
            counter+=1
        for (x,y,w,h) in faces_coord:
            cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),8)
        cv2.imshow("create_data",frame)
        cv2.waitKey(50)
        timer+=50
    del cap
    cv2.destroyAllWindows()
else:
    print("student already present")