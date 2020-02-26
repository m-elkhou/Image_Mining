import cv2, sys, os, joblib, mahotas
from glob import glob
# from tqdm import tqdm
from tqdm import trange, tqdm
import numpy as np
import math
from sklearn.model_selection import KFold
from sklearn.model_selection import ShuffleSplit

from sklearn.metrics import confusion_matrix
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

home = 'C:/Users/mhmh2/Desktop/PlantDiseaseDataSet/'

scaler = StandardScaler()
pca = PCA(n_components=40)
relief = ReliefF(n_features_to_keep=30)

def normalization_train(data_train, target):
    global pca
    global relief
    global scaler
    data_train = pca.fit_transform(data_train)
    data_train = scaler.fit_transform(data_train)

    dic = { val : nb for nb, val in enumerate(set(target))}
    target2nb = [dic[l] for l in target]
    data_train = np.array(data_train)
    target2nb = np.array(target2nb)
    data_train = relief.fit_transform(data_train, target2nb)
    return data_train

def normalization_test(data_test):
    global pca
    global relief
    global scaler
    data_test = pca.transform(data_test)

    data_test = scaler.transform(data_test)
    data_test = np.array(data_test)

    data_test = relief.transform(data_test)
    return data_test

def get_feature(img):
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

    feature = mahotas.features.haralick(img_gray).ravel()
    global_features.extend(feature)
    
    feature = cv2.HuMoments(cv2.moments(img_gray)).flatten()
    global_features.extend(feature)

    # global_features = stats.zscore(np.array(global_features))
    return global_features


def indexing(dir_path):
    features =[]
    labels=[]
    folders = glob(str(dir_path)+os.path.sep+"*"+os.path.sep)
    for dir in tqdm(folders, desc='DB '):
        if os.path.isdir(dir):
            label = dir.split(os.path.sep)[-2]
            files = glob(str(dir)+os.path.sep+"*.*")
            for img_path in tqdm(files, desc=label+"\t", leave=False):
                img = cv2.imread(img_path)
                feature = get_feature(img)
                if feature is None:
                    continue
                features.append(feature)
                labels.append(label)
    return np.array(features), np.array(labels)
    
# train
features, labels = indexing(home+'train')
# features = normalization_train(features, labels)

# validation
features_v, labels_v = indexing(home+'validation')
# features_v = normalization_test(features_v)

# score
# model = svm.SVC(kernel='rbf')
# model.fit(features,labels)

# print(model.score(features_v, labels_v))

# final
features = np.concatenate([ features, features_v])
labels = np.concatenate([ labels, labels_v])

features = normalization_train(features, labels)

model = svm.SVC(kernel='rbf')
model.fit(features,labels)

joblib.dump(model, home+'model')

# test 

features_test = []
imgIds =[]
files = glob(home+'test'+os.path.sep+"*.*")
for img_path in tqdm(files):
    img = cv2.imread(img_path)
    feature = get_feature(img)
    if feature is None:
        continue
    features_test.append(feature)
    imgIds.append(img_path.split(os.path.sep)[-1])

features_test = np.array(features_test)
features_test = normalization_test(features_test)
labels_test = [model.predict([feature])[0] for feature in features_test]

labels_test = ['Plante_'+str(l) for l in  labels_test]
df = pd.DataFrame({'nom_image': imgIds, 'classe_predite': labels_test})
df.to_csv(home+'csv_file.csv', index=False)

print("[DONE]")
