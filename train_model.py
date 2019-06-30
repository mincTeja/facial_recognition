import cv2
import numpy as np
import os
import matplotlib.pyplot as plt


def collect_dataset():
    images=[]
    labels=[]
    labels_dic={}
    person= [p for p in os.listdir('student')]
    for i,s in enumerate(person):
        labels_dic[i]=s
        for img in os.listdir('student/'+s):

            i_to_c=cv2.imread("student/"+s+"/"+img,0)
            i_to_c=cv2.resize(i_to_c,(50,50),interpolation=cv2.INTER_CUBIC)
            images.append(i_to_c)
            labels.append(i)
    return (images,np.array(labels),labels_dic)

images,labels,labels_dic=collect_dataset()

rec_eig = cv2.face.EigenFaceRecognizer_create()
rec_eig.train(images,labels)
rec_eig.save('recognition_model.yml')