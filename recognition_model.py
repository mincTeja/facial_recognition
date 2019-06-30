import cv2
import numpy as np
import os
import matplotlib.pyplot as plt
from video_capture import videoCamera
from face_detection import faceDetector
from face_normalization import normalization

rec_test=cv2.face.EigenFaceRecognizer_create()
rec_test.read("recognition_model.yml")



cap=videoCamera()
face_detect=faceDetector("./DATA/haarcascades/haarcascade_frontalface_alt.xml")
norm=normalization()

def collect_dataset():
    images=[]
    labels=[]
    labels_dic={}
    person= [p for p in os.listdir('student')]
    for i,s in enumerate(person):
        labels_dic[i]=s
        for img in os.listdir('student/'+s):
            images.append(cv2.imread("student/"+s+"/"+img,0))
            labels.append(i)
    return (images,np.array(labels),labels_dic)

images,labels,labels_dic=collect_dataset()

cv2.namedWindow('f')
while True:
    frame=cap.get_frame()
    face_coord=face_detect.detect(frame)
    if len(face_coord):
        faces=norm.face_normalization(frame,face_coord)
        for i,face in enumerate(faces):
            label,confidence=rec_test.predict(face)
            thres=140
            #print(labels_dic[label],confidence)
                
            cv2.putText(frame,labels_dic[label].capitalize(),(face_coord[i][0],face_coord[i][1]-10),cv2.FONT_HERSHEY_PLAIN,3,(255,255,0),2)
                
    for (x,y,w,h) in face_coord:
        cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),8)
    cv2.imshow('f',frame)
    if cv2.waitKey(20) & 0xFF==27:
        break


del cap
cv2.destroyAllWindows()