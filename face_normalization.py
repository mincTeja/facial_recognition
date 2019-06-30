import cv2
import numpy as np
import os
import matplotlib.pyplot as plt
class normalization():
    def face_cut(self,image,coord):
        faces=[]
        for (x,y,w,h) in coord:
            #w_rm=int(0.2*w/2)
            faces.append(image[y:y+h , x:x+w])
        return faces
    
    def normalize(self,image):
        n_img=[]
        for i in image:
            is_color= len(i.shape)==3
            if is_color:
                i=cv2.cvtColor(i,cv2.COLOR_BGR2GRAY)
            n_img.append(cv2.equalizeHist(i))
        return n_img
    
    def resize(self,image,size=(50,50)):
        img_norm=[]
        for i in image:
            if i.shape<size:
                i_norm=cv2.resize(i,size,interpolation=cv2.INTER_AREA)
            else:
                i_norm=cv2.resize(i,size,interpolation=cv2.INTER_CUBIC)
            img_norm.append(i_norm)
        return img_norm
    
    def face_normalization(self,img,coord):
        faces=self.face_cut(img,coord)
        faces=self.normalize(faces)
        faces=self.resize(faces)
        return faces