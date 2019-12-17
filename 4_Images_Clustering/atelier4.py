"""
@author: Mohammed EL KHOU
"""
import cv2, glob, imutils, sys, os, csv, mahotas
import numpy as np

import skimage.feature.texture as sft
from sklearn.neighbors import KNeighborsClassifier
from sklearn.mixture import GaussianMixture
from sklearn.cluster import KMeans
from sklearn import cluster

import pandas as pd
from sklearn.preprocessing import StandardScaler
# Importing the library that supports centering and scaling vectors
from sklearn import metrics

from feature import Feature as ft

#Helper function to load images from given directories    
def indexing_images(directorys):
    print("\nLoad images and Indexing : ")
    all_files = glob.glob(str(directorys)+os.path.sep+"*")
    features = []
    imgNames = []
    Ft = ft()
    for imagePath in all_files:
        img = cv2.imread(imagePath)

        # feature = Ft.extract_global_features(img)
        feature = Ft.getFeatures(img)
        imgName = imagePath.split(os.path.sep)[-1]
        features.append(feature)
        imgNames.append(imgName)

    return  (imgNames,features)
    
def clustering(features, number_of_clusters):
    # model = KMeans(n_clusters = number_of_clusters)
    model = KMeans(n_clusters = number_of_clusters, init = 'k-means++')
    # model = GaussianMixture(n_components = number_of_clusters)

    # stdSlr = StandardScaler().fit(features)
    # img_features = stdSlr.transform(features)

    scaler = StandardScaler()
    img_features = scaler.fit_transform(features)

    model.fit(img_features)#features)
    predictions = model.predict(img_features)
    print(predictions)
    return predictions

def image_clustering(data_path, nb_clusters, csv_fil):
    imgNames, features = indexing_images(data_path)
    tab = clustering(features, nb_clusters)

    with open(csv_fil, mode='w', newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, ['image_id','classe'])
        writer.writeheader()
        for t, n in zip(tab, imgNames):
            writer.writerow({'image_id': n, 'classe': t+1})


# La fonction main
if __name__ == '__main__':
    data_path = "D:/WISD/S3/Image_Mining/Atelier_4/2Classes"
    data_path2 = "D:/WISD/S3/Image_Mining/Atelier_4/4Classes"
    csv_path  = "D:/WISD/S3/Image_Mining/Atelier_4/index.csv"

    image_clustering(data_path, 2, csv_path)
    # image_clustering(data_path2, 4, csv_path)