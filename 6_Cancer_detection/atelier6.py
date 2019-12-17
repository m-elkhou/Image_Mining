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
        
        data_train = self.pca.fit_transform(data_train)
        # data_train = self.mscaler.fit_transform(data_train)
        data_train = self.scaler.fit_transform(data_train)

        dic = { val : nb for nb, val in enumerate(set(target))}
        target2nb = [dic[l] for l in target]
        data_train = np.array(data_train)
        target2nb = np.array(target2nb)
        # data_train = self.bestK.fit_transform(data_train, target2nb)
        data_train = self.relief.fit_transform(data_train, target2nb)
        return data_train

    def normalization_test(self, data_test):
        
        data_test = self.pca.transform(data_test)
        # data_test = self.mscaler.transform(data_test)
        data_test = self.scaler.transform(data_test)
        data_test = np.array(data_test)
        # data_test = self.bestK.transform(data_test)
        data_test = self.relief.transform(data_test)
        return data_test

Ft = Feature()

#Helper function to load images from given directories    
def indexing_images(directorys, csv_path):
    df = pd.read_csv(csv_path)
    print("\nLoad images and Indexing : ")
    all_files = glob(str(directorys)+os.path.sep+"*.*")
    features = []
    labels = []

    for imagePath in all_files:       
        feature = Ft.extract_global_features(cv2.imread(imagePath))
        features.append(feature)
        imgName = imagePath.split(os.path.sep)[-1].split(".")[0]
        print(imgName)
        label = np.array(df[df['image_id']==imgName]['melanoma'])[0]
        labels.append(label)

    return np.array(features), np.array(labels)

def Apprentissage(modelName, data_train, target_train):
    print("\n\n** Read data : ")
    features, labels = indexing_images(data_train, target_train)

    print("\n\n** Normalization : ")
    features = Ft.normalization_train(features, labels)
    
    print("\n\n** Apprentissage : ")    
    # train and evaluate a k-NN classifer on the raw pixel intensities
    # model = KNeighborsClassifier(n_neighbors=16)

    # model = svm.SVC(gamma=0.01, C=100, kernel='linear')
    # model = svm.SVC(kernel='rbf')
    model = svm.SVC(kernel='linear')
    print("[INFO] evaluating raw pixel accuracy...")

    model.fit(features,labels)
    # save the model to disk
    joblib.dump(model, modelName)
    print("[DONE]")

def best_SVM(X_train, y_train, X_test, y_test):
    # from sklearn import metrics
    C = 200
    mean_acc = np.zeros((C-1))
    for n in range(1, C):
        # model = svm.SVC(gamma=0.01, C=100, kernel='linear').fit(X_train,y_train)
        model = svm.SVC(gamma=0.01, C=C, kernel='rbf').fit(X_train,y_train)
        mean_acc[n-1] = model.score(X_test, y_test)
    import matplotlib.pyplot as plt
    plt.plot(range(1,C),mean_acc,'g')
    plt.legend(('Accuracy ', '+/- 3xstd'))
    plt.ylabel('Accuracy ')
    plt.xlabel('Number of Nabors (K)')
    plt.tight_layout()
    plt.show()
    print( "The best accuracy was with", mean_acc.max(), "with k=", mean_acc.argmax()+1) 
    return mean_acc.argmax()+1 , mean_acc.max()

def bestK_KNN(X_train, y_train, X_test, y_test):
    Ks = 200
    mean_acc = np.zeros((Ks-1))
    for n in range(1,Ks):
        neigh = KNeighborsClassifier(n_neighbors = n).fit(X_train,y_train)
        mean_acc[n-1] = neigh.score(X_test, y_test)

    import matplotlib.pyplot as plt
    plt.plot(range(1,Ks),mean_acc,'g')
    plt.legend(('Accuracy ', '+/- 3xstd'))
    plt.ylabel('Accuracy ')
    plt.xlabel('Number of Nabors (K)')
    plt.tight_layout()
    plt.show()
    print( "The best accuracy was with", mean_acc.max(), "with k=", mean_acc.argmax()+1) 
    return mean_acc.argmax()+1 , mean_acc.max()

def Classification(model_path , directorys, csv_path):
    features, labels = indexing_images(directorys, csv_path)
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
    
    home = "D:/WISD/S3/Image_Mining/Atelier_6/"

    data_train      = home + "Classification DataSet/ISIC-2017_Training_Data"
    data_train_mask = home + "Classification DataSet/ISIC-2017_Training_Part1_GroundTruth"
    data_valid      = home + "Classification DataSet/ISIC-2017_Validation_Data"
    data_valid_mask = home + "Classification DataSet/ISIC-2017_Validation_Data"
    data_test       = home + "Classification DataSet/ISIC-2017_Test_v2_Data"

    target_train    = home + "ISIC-2017_Training_Part3_GroundTruth.csv"
    target_valid    = home + "ISIC-2017_Validation_Part3_GroundTruth.csv"

    model      = home + 'finalized_model.sav'
    csv_file   = home + "index.csv"
    
    # Apprentissage(model, data_train, target_train)
    # Classification(model, data_valid, target_valid)
    # Classification_2_csv(model , data_test, csv_file)


    f_train, l_test = indexing_images(data_train, target_train)
    # f_train = Ft.normalization_train(f_train)

    f_valid, l_valid = indexing_images(data_valid, target_valid)
    # f_valid = Ft.normalization_test(f_valid)

    # accuracy, k = bestK_KNN(f_train, l_test, f_valid, l_valid )
    accuracy, C = best_SVM(f_train, l_test, f_valid, l_valid )