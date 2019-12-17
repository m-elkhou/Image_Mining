# -*- coding: utf-8 -*-
"""
Created on Tue Oct 15 10:32:48 2019

@author: Mohammed EL KHOU
"""
import cv2, imutils, os,csv
from glob import glob
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

from sklearn import svm
from sklearn.neighbors import KNeighborsClassifier
# from sklearn.model_selection import train_test_split
import random
import joblib
from scipy import stats
from feature import Feature as ft
Ft = ft()
home = "D:/WISD/S3/Image_Mining/Atelier_3/"

def segmentation(img):
#    # define the upper and lower boundaries of the HSV pixel
#    # intensities to be considered 'skin'
#    lower = np.array([0, 48, 80], dtype = "uint8")
#    upper = np.array([20, 255, 255], dtype = "uint8")
#    #  convert the image to the HSV color space,
#	# and determine the HSV pixel intensities that fall into
#	# the speicifed upper and lower boundaries
#    converted = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
#    skinMask = cv2.inRange(converted, lower, upper)
# 
#	# apply a series of erosions and dilations to the mask
#	# using an elliptical kernel
#    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
#    skinMask = cv2.erode(skinMask, kernel, iterations = 2)
#    skinMask = cv2.dilate(skinMask, kernel, iterations = 2)
# 
#	# blur the mask to help remove noise, then apply the
#	# mask to the frame
#    skinMask = cv2.GaussianBlur(skinMask, (3, 3), 0)
#    skin = cv2.bitwise_and(img, img, mask = skinMask)
    
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    handHist = cv2.calcHist([hsv], [0, 1], None, [180, 256], [0, 180, 0, 256])
    handHist = cv2.normalize(handHist, handHist, 0, 255, cv2.NORM_MINMAX)
    dst = cv2.calcBackProject([hsv], [0, 1], handHist, [0, 180, 0, 256], 1)
    disc = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (21, 21))
    cv2.filter2D(dst, -1, disc, dst)
    # dst is now a probability map
    # Use binary thresholding to create a map of 0s and 1s
    # 1 means the pixel is part of the hand and 0 means not
    _, thresh = cv2.threshold(dst, 150, 255, cv2.THRESH_BINARY)
    kernel = np.ones((5, 5), np.uint8)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=7)
    thresh = cv2.merge((thresh, thresh, thresh))
    skin   = cv2.bitwise_and(img, thresh)

    plt.xticks([]),plt.yticks([])
    plt.title('ImageRequete')
    plt.imshow(skin)
    plt.show()
        
    return skin

#Helper function to load images from given directories    
def indexing_images(directorys):
    print("\nLoad images and Indexing : ")
    all_files = glob(str(directorys)+os.path.sep+"*"+os.path.sep)
    features = []
    labels = []

    for dir in all_files:
        if os.path.isdir(dir):
            label = dir.split(os.path.sep)[-2]
            print("\nClasse : " +str(label))
            files = glob.glob(str(dir)+os.path.sep+"*.*")

            tmp = len(files)/40
            cpp = 0 
            for imagePath in files:                
                feature = Ft.getFeatures(imagePath)
                features.append(feature)
                labels.append(label)
                cpp += 1
                if(cpp % int(tmp) == 0):
                    print("*", end="", flush=True)
    features = np.array(features)
    labels = np.array(labels)
    np.save('features.npy',features,allow_pickle=True)
    np.save('labels.npy',labels,allow_pickle=True)

def Apprentissage(modelName):
    print("\n\n** Read data : ")
    features =np.load('features.npy',allow_pickle=True)
    labels = np.load('labels.npy',allow_pickle=True)

    print("\n\n** Normalization : ")
    # print(features.shape)
    # features = [Ft.normalization_zscore(f) for f in features]
    features = Ft.normalization_train(features, labels, 100)
    
    print("\n\n** Apprentissage : ")
    # (trainFeat, testFeat, trainLabels, testLabels) = train_test_split(
    #         features, labels, test_size=0.01, random_state=random.seed())
    
    # train and evaluate a k-NN classifer on the raw pixel intensities
    # model = KNeighborsClassifier(n_neighbors=1)

    # model = svm.SVC(gamma=0.01, C=100)
    model = svm.SVC(gamma='scale')
#    cl= svm.LinearSVC()
    print("[INFO] evaluating raw pixel accuracy...")
    # model.fit(trainFeat,trainLabels)
    model.fit(features,labels)
    # save the model to disk
    joblib.dump(model, modelName)
    # np.save('model.npy',model,allow_pickle=True)
    print("[DONE]")
    # print("\n\nTaux de précision :", model.score(testFeat,testLabels)*100 , "%")
    # print("\n\nTaux de précision  pour train :", model.score(trainFeat[:100],trainLabels[:100])*100 , "%")
        
def Classification(filename , ImageRequetePath):
    print("[INFO] Classification...")
    loaded_model = joblib.load(filename)
    # loaded_model = np.load('model.npy',allow_pickle=True)
    if os.path.isdir(ImageRequetePath):
        files = glob(str(ImageRequetePath)+os.path.sep+"*.*")
        for imagePath in files:
            feature = Ft.getFeatures(imagePath)

            plt.xticks([]),plt.yticks([])
            imgName = imagePath.split(os.path.sep)[-1]
            plt.title(imgName)
            
            # segmentation(cv2.imread(imagePath))
            plt.imshow(mpimg.imread(imagePath))
            plt.show()
            print("\nObject  Type : ==> ", loaded_model.predict([feature])[0])
    else:
        feature = Ft.getFeatures(ImageRequetePath)
    
        plt.xticks([]),plt.yticks([])
        plt.title('ImageRequete')
        plt.imshow(mpimg.imread(ImageRequetePath))
        plt.show()
        print("\nObject  Type : ==> ", loaded_model.predict([feature])[0])

def Classification2(filename , ImageRequetePath):
    loaded_model = joblib.load(filename)
    if os.path.isdir(ImageRequetePath):
        files = glob(str(ImageRequetePath)+os.path.sep+"*.*")
        features = []
        labels = []
        for imagePath in files:
            features.append(Ft.getFeatures(imagePath))
            labels.append(imagePath.split(os.path.sep)[-1].split('_')[0])
        features = Ft.normalization_test(features, 100)
        print("\n\nTaux de précision :", loaded_model.score(features, labels)*100 , "%")
    
# La fonction main
if __name__ == '__main__':
    data_train = home + "asl-alphabet/asl_alphabet_train"
    data_test  = home + "asl-alphabet/asl_alphabet_test"
    model      = home + 'finalized_model.sav'
    csv_file   = home + "index.csv"
    
    # indexing_images(data_train)
    Apprentissage(model)
    # Classification(model, data_test)
    Classification2(model, data_test)
    
'''
    pour instaler mahotas :
        install Microsoft Visual C++ 14.0 build tools : https://go.microsoft.com/fwlink/?LinkId=691126
        pip install --upgrade setuptools
        pip install mahotas
        
        or the Binary install it the simple way!:
            pip install --only-binary :all: mahotas
            
        or :
            conda config --add channels conda-forge
            conda install mahotas
        or :
            conda install -c https://conda.anaconda.org/conda-forge mahotas

surce :
    https://gogul.dev/software/image-classification-python
    https://scikit-image.org/docs/dev/api/skimage.feature.html?highlight=energy
    https://github.com/scikit-image/scikit-image
    https://stackoverflow.com/questions/50834170/image-texture-with-skimage
    https://github.com/Gogul09/image-classification-python/blob/master/global.py
    https://www.pyimagesearch.com/2014/08/18/skin-detection-step-step-example-using-python-opencv/
        
'''