import cv2
import numpy as np
import matplotlib.pyplot as plt
import json
from IPython import embed

def findInterestPoints(img, detector, stride = 10):
    keypoints = []

    if detector == "Dense":
        for i in np.arange(0, img.shape[0], stride):
            for j in np.arange(0,img.shape[1], stride):
                k = cv2.KeyPoint(i,j,20)
                keypoints.append(k)                    
    if detector == "SIFT":
        sift = cv2.xfeatures2d.SIFT_create()
        keypoints = sift.detect(img,None)
    return keypoints

class Dataset:
    butterflyData = {}
    butterflyIndexes = {}
    testOrTrainJson = "train"
    def __init__(self, jsonFile, testOrTrain = "train"):
        with open(jsonFile) as f:
            self.butterflyData = json.load(f)
        if testOrTrain == "train":
            self.testOrTrainJson = "training_images"
        elif testOrTrain == "test":
            self.testOrTrainJson = "train_images"
        else:
            print "ERROR: testOrTrain parameter must be test or train"
            exit()
            
        self.butterflyIndexes[self.testOrTrainJson] = {}
        for cat in self.butterflyData[self.testOrTrainJson]:
            self.butterflyIndexes[self.testOrTrainJson][str(cat)] = {}
            self.butterflyIndexes[self.testOrTrainJson][str(cat)]["len"] = len(self.butterflyData[self.testOrTrainJson][str(cat)])
            self.butterflyIndexes[self.testOrTrainJson][str(cat)]["current"] = 0
    
    def getNextImage(self, cat = "1"):
        length = self.butterflyIndexes[self.testOrTrainJson][str(cat)]["len"]
        currentIndex = self.butterflyIndexes[self.testOrTrainJson][str(cat)]["current"]
        filename = self.butterflyData[self.testOrTrainJson][str(cat)][currentIndex % length]  
        self.butterflyIndexes[self.testOrTrainJson][str(cat)]["len"] += 1
        return cv2.imread("../" + filename, 1)
