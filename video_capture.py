import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
class videoCamera():
    def __init__(self,index=0):
        self.video= cv2.VideoCapture(index)
        self.index=index
    def __del__(self):
        self.video.release()
    def get_frame(self,in_gray=False):
        ret,frame=self.video.read()
        if in_gray:
            frame=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
        return frame 