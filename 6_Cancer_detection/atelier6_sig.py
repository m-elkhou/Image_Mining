# -*- coding: utf-8 -*-
"""
@author: Mohammed EL KHOU
"""
import cv2, imutils, os, csv, mahotas
from glob import glob
import numpy as np
import pandas as pd
from sklearn import svm

from sklearn.neighbors import KNeighborsClassifier
import random
import joblib
from scipy import stats
import skimage.feature.texture as sft
from sklearn.preprocessing import StandardScaler , MinMaxScaler
from sklearn.decomposition import PCA
from ReliefF import ReliefF
from sklearn.feature_selection import SelectKBest, f_classif

home = "D:/WISD/S3/Image_Mining/Atelier_6/"


class Feature:
    def __init__(self):
        self.scaler = StandardScaler()
        self.mscaler = MinMaxScaler(feature_range=(0, 1))
        self.pca = PCA(n_components=40)
        self.relief = ReliefF(n_features_to_keep=30)
        self.bestK = SelectKBest(f_classif, k=30)

    def extract_global_features(self, img):
        global_features = []
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

        R, G, B = img[:,:,0], img[:,:,1], img[:,:,2]
        feature = [np.mean(R), np.std(R), np.mean(G), np.std(G), np.mean(B), np.std(B)]
        global_features.extend(feature/np.mean(feature))

        hist = cv2.calcHist([img_hsv], [0, 1, 2], None, (8, 2, 2), [0, 256, 0, 256, 0, 256])
        cv2.normalize(hist, hist)
        global_features.extend(hist.flatten())      

        glcm = sft.greycomatrix(img_gray, distances = [1], angles = [0], symmetric=True, normed=True)
        # glcm = sft.greycomatrix(img_gray, distances = [1], angles = [0, np.pi/4, np.pi/2, 3*np.pi/4], symmetric=True, normed=True)
        props = ['contrast', 'correlation', 'energy', 'homogeneity']
        feature = [sft.greycoprops(glcm, prop).ravel()[0] for prop in props]
        feature = feature / np.sum(feature)
        global_features.extend(feature)

        # feature = mahotas.features.haralick(img_gray).ravel()
        # global_features.extend(feature)
        
        feature = cv2.HuMoments(cv2.moments(img_gray)).flatten()
        global_features.extend(feature)

        # global_features = stats.zscore(np.array(global_features))

        return global_features

    def normalization_train(self, data_train, target):
        
        # data_train = self.pca.fit_transform(data_train)
        # data_train = self.mscaler.fit_transform(data_train)
        data_train = self.scaler.fit_transform(data_train)

        dic = { val : nb for nb, val in enumerate(set(target))}
        target2nb = [dic[l] for l in target]
        data_train = np.array(data_train)
        target2nb = np.array(target2nb)
        data_train = self.bestK.fit_transform(data_train, target2nb)
        # data_train = self.relief.fit_transform(data_train, target2nb)
        return data_train

    def normalization_test(self, data_test):
        
        # data_test = self.pca.transform(data_test)
        # data_test = self.mscaler.transform(data_test)
        data_test = self.scaler.transform(data_test)
        data_test = np.array(data_test)
        data_test = self.bestK.transform(data_test)
        # data_test = self.relief.transform(data_test)
        return data_test

Ft = Feature()

def segmentation(img, mask):
    # _, mask = cv2.threshold(mask, 150, 255, cv2.THRESH_BINARY)
    r = img[:,:,0]
    g = img[:,:,1]
    b = img[:,:,2]
    rc = np.multiply(r, mask)
    gc = np.multiply(g, mask)
    bc = np.multiply(b, mask)
    return np.dstack((rc,gc,bc)).astype(np.uint8)

#Helper function to load images from given directories    
def indexing_images(dir1, dir2, csv_path):
    df = pd.read_csv(csv_path)
    print("\nLoad images and Indexing : ")
    features = []
    labels = []

    for img_path, mask_path in zip(glob(dir1+os.path.sep+"*.*"), glob(dir2+os.path.sep+"*.*")): 
        img = cv2.imread(img_path)
        mask = cv2.imread(mask_path,0)
        # img = cv2.resize(cv2.imread(img_path), (32, 32))
        # mask = cv2.resize(cv2.imread(mask_path,0), (32, 32))
        img = segmentation(img, mask)
        feature = Ft.extract_global_features(img)
        features.append(feature)
        imgName = img_path.split(os.path.sep)[-1].split(".")[0]
        print(imgName)
        label = np.array(df[df['image_id']==imgName]['melanoma'])[0]
        labels.append(label)

    return np.array(features), np.array(labels)

def Apprentissage(modelName, data_train, data_train_mask, target_train):
    print("\n\n** Read data : ")
    features, labels = indexing_images(data_train, data_train_mask, target_train)

    print("\n\n** Normalization : ")
    # features = [Ft.normalization_zscore(f) for f in features]
    features = Ft.normalization_train(features, labels)
    
    print("\n\n** Apprentissage : ")    
    # train and evaluate a k-NN classifer on the raw pixel intensities
    model = KNeighborsClassifier(n_neighbors=100)

    # model = svm.SVC(gamma=0.01, C=100)
    print("[INFO] evaluating raw pixel accuracy...")

    model.fit(features,labels)
    # save the model to disk
    joblib.dump(model, modelName)
    print("[DONE]")

def Classification(model_path , dir1, dir2, csv_path):
    features, labels = indexing_images(dir1, dir2, csv_path)
    features = Ft.normalization_test(features)
    loaded_model = joblib.load(model_path)
    print("\n\nTaux de précision :", loaded_model.score(features, labels)*100 , "%")

def Classification_2_csv(model_path , dir1, csv_path):
    loaded_model = joblib.load(model_path)

    features = []
    imgIds = []
    for img_path in glob(dir1+os.path.sep+"*.*"):
        features.append(Ft.extract_global_features(cv2.imread(img_path)))
        imgIds.append(img_path.split(os.path.sep)[-1].split(".")[0])
    
    features = Ft.normalization_test(features)
    melanoma = [loaded_model.predict([feature])[0] for feature in features]
    df = pd.DataFrame({'image_id': imgIds, 'melanoma': melanoma})
    df.to_csv(csv_path, index=False)

# La fonction main
if __name__ == '__main__':
    data_train      = home + "Classification DataSet/ISIC-2017_Training_Data"
    data_train_mask = home + "Classification DataSet/ISIC-2017_Training_Part1_GroundTruth"
    data_valid      = home + "Classification DataSet/ISIC-2017_Validation_Data"
    data_valid_mask = home + "Classification DataSet/ISIC-2017_Validation_Data"
    data_test       = home + "Classification DataSet/ISIC-2017_Test_v2_Data"

    target_train    = home + "ISIC-2017_Training_Part3_GroundTruth.csv"
    target_valid    = home + "ISIC-2017_Validation_Part3_GroundTruth.csv"

    model_path      = home + 'finalized_model.sav'
    csv_file   = home + "index.csv"
    
    Apprentissage(model_path, data_train, data_train_mask, target_train)
    Classification(model_path, data_valid, data_valid_mask, target_valid)
    # Classification_2_csv(model_path , data_test, csv_file)
    