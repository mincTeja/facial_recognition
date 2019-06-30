import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
class faceDetector():
    def __init__(self,xml_file):
        self.classifier = cv2.CascadeClassifier(xml_file)
    def detect(self,image):
        f_coord=self.classifier.detectMultiScale(image)
        return f_coord