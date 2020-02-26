#---------------------------------------------------------------------------------
# Install necessary library and packages
#---------------------------------------------------------------------------------
# !pip install -U -r requirements.txt
#---------------------------------------------------------------------------------
# Import necessary library
#---------------------------------------------------------------------------------
import cv2, sys, os, joblib
from glob import glob
from tqdm import trange, tqdm
import numpy as np
import pandas as pd
# import math
import random
from skimage import feature as ft
import skimage.feature.texture as sft
import mahotas
from scipy import stats

from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier

from sklearn.model_selection import train_test_split
# from sklearn.model_selection import KFold, ShuffleSplit
from sklearn.metrics import confusion_matrix

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.feature_selection import SelectKBest, f_classif
from ReliefF import ReliefF

import matplotlib.pyplot as plt
# import matplotlib.image as mpimg
import seaborn as sns 

home = 'C:/Users/mhmh2/Desktop/PlantDiseaseDataSet/'


#---------------------------------------------------------------------------------
# Problem Definition
#---------------------------------------------------------------------------------
'''
Problem definition is all about understanding the problem being solved in order to find 
the most suitable ML techniques for use. This starts by deciding the scope of the problem. 
That is whether it is a supervised problem (classification or regression) or an 
unsupervised problem (clustering). After defining the problem scope, next is to decide 
which ML algorithms to be used. For example, if it is a supervised problem then which 
ML algorithm to be used. It is linear or non-linear, parametric or non-parametric, and so on.

Defining the problem serves the next step in the ML pipeline which is data preparation. 
The machine learns by examples. Each example has inputs and outputs. If the problem is a 
classification problem for example in which each sample is to be classified to one of 
the pre-defined categories, then the outputs are labels. If it is a regression problem 
in which the outputs are represented as a continuous line, then the output is no longer 
a label but a number. So, by defining the problem we are able to prepare the data in a 
suitable form.
'''

#---------------------------------------------------------------------------------
# Data Preparation
#---------------------------------------------------------------------------------

#---------------------------------------------------------------------------------
# Helper function to load images from given directories 
#---------------------------------------------------------------------------------
# Pre-prepared dataset
def indexing(dir_path, step):
    features, labels = [], []
    folders = glob(str(dir_path)+os.path.sep+"*"+os.path.sep)
    for dir in folders:#tqdm(folders, desc='DB ', file=sys.stdout):
    # for d in trange(len(folders), desc='DB '+step, file=sys.stdout):
        # tqdm._instances.clear()
        # dir = folders[d]
        if not os.path.isdir(dir):
            continue
        label = dir.split(os.path.sep)[-2]
        files = glob(str(dir)+os.path.sep+"*.*")
        for img_path in tqdm(files, desc=label+"\t", leave=False):
            # tqdm._instances.clear()
            img = cv2.imread(img_path)
            feature = extract_features(img)
            if feature is None:
                continue
            features.append(feature)
            labels.append(label)
    
    features, labels = np.array(features), np.array(labels)
    print(labels.shape, features.shape)
    if not os.path.exists(home+'tools'):
        os.makedirs(home+'tools')
    np.save(home+'tools/features_'+str(step)+'.npy', features, allow_pickle=True)
    np.save(home+'tools/labels_'+str(step)+'.npy', labels, allow_pickle=True)
    return np.array(features), np.array(labels)

# dataset "mrawna" (1 folder) (labels in the name of image)
def indexing2(dir_path, step,features=[], labels=[]):
    dictionary = {'NE': 'neutral', 'HA': 'happy', 'SA': 'sadness',
                    'SU': 'surprise', 'AN': 'anger', 'DI': 'disgust', 'AF': 'fear'}
    imgs_path = glob(str(dir_path)+os.path.sep+"*"+os.path.sep+"*.JPG")
    for img_path in tqdm(imgs_path, desc='DB ', file=sys.stdout):
        try:
            img_name = img_path.split(os.path.sep)[-1]
        except:
            continue
        if img_name[6:8] in ['FR', 'FL']:
            continue
        img = cv2.imread(img_path)
        feature = extract_features(img)
        if feature is None:
            continue
        features.append(feature)
        labels.append(dictionary[img_name[4:6]])
    
    features, labels = np.array(features), np.array(labels)
    shape = features.shape
    print(labels.shape, features.shape)
    if not os.path.exists(home+'tools'):
        os.makedirs(home+'tools')
    np.save(home+'tools/features_'+str(step)+'.npy', features, allow_pickle=True)
    np.save(home+'tools/labels_'+str(step)+'.npy', labels, allow_pickle=True)
    return np.array(features), np.array(labels), shape

# Half-prepared dataset   
def indexing3(directorys, csv_path, step, features=[], labels=[]):
    df = pd.read_csv(csv_path)
    all_files = glob(str(directorys)+os.path.sep+"*.*")

    for img_path in tqdm(all_files, desc='DB ', file=sys.stdout):       
        feature = extract_features(cv2.imread(img_path))
        if feature is None:
            continue
        img_name = img_path.split(os.path.sep)[-1].split(".")[0]
        label = np.array(df[df['image_id']==img_name]['melanoma'])[0]
        if label is None:
            continue
        labels.append(label)
        features.append(feature)

    features, labels = np.array(features), np.array(labels)
    shape = features.shape
    print(labels.shape, features.shape)
    if not os.path.exists(home+'tools'):
        os.makedirs(home+'tools')
    np.save(home+'tools/features_'+str(step)+'.npy', features, allow_pickle=True)
    np.save(home+'tools/labels_'+str(step)+'.npy', labels, allow_pickle=True)
    return np.array(features), np.array(labels), shape

#---------------------------------------------------------------------------------
# Extract Feature Descriptors
#---------------------------------------------------------------------------------
def extract_features(img):
    global_features = []
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # Extract color moments
    R, G, B = img[:,:,0], img[:,:,1], img[:,:,2]
    feature = [np.mean(R), np.std(R), np.mean(G), np.std(G), np.mean(B), np.std(B)]
    global_features.extend(feature/np.mean(feature))

    # extract a 3D color histogram from the HSV color space using
	# the supplied number of 'bins' per channel // bins=(8, 2, 2)
    hist = cv2.calcHist([img_hsv], [0, 1, 2], None, (8, 2, 2), [0, 256, 0, 256, 0, 256])
    cv2.normalize(hist, hist)
    global_features.extend(hist.flatten())      

    # Basée sur l'analayse de textures par la GLCM (Gray-Level Co-Occurrence Matrix)
    glcm = sft.greycomatrix(img_gray, distances = [1], angles = [0], symmetric=True, normed=True)
    # glcm = sft.greycomatrix(img_gray, distances = [1], angles = [0, np.pi/4, np.pi/2, 3*np.pi/4], symmetric=True, normed=True)
    props = ['contrast', 'correlation', 'energy', 'homogeneity']
    feature = [sft.greycoprops(glcm, prop).ravel()[0] for prop in props]
    feature = feature / np.sum(feature)
    global_features.extend(feature)

    # Compute the haralick texture feature vector
    feature = mahotas.features.haralick(img_gray).ravel()
    global_features.extend(feature)
    
    # Hu Moments that quantifies shape of the Object in img.
    feature = cv2.HuMoments(cv2.moments(img_gray)).flatten()
    global_features.extend(feature)

    # ** Extraction des Features en se basant sur la texture: **
    """
    La méthode LBP contient des informations sur la distribution des micro-motifs locaux.

    Les expressions faciales peuvent être considérées comme une composition de micro-motifs 
    qui peuvent être efficacement décrits par les fonctionnalités LBP. 

    Un histogramme LBP présente uniquement les occurrences des micro-motifs sans aucune indication sur leur emplacement.
    """
    # numPoints, radius, eps = 25, 8, 1e-7 
    # lbp = ft.local_binary_pattern(img_gray, numPoints, radius, method="uniform")
    # (hist, _) = np.histogram(lbp.ravel(), bins=np.arange(0, numPoints + 3), range=(0, numPoints + 2))
    # # normalize the histogram
    # hist = hist.astype("float")
    # hist /= (hist.sum() + eps)
    # global_features.extend(hist)

    # # Extraction de features en utilisant Histogram of oriented gradient
    # feature = ft.hog(img_gray, orientations=6, pixels_per_cell=(9, 9), cells_per_block=(1, 1))
    # global_features.extend(feature)


    # global_features = stats.zscore(np.array(global_features))
    return global_features


#---------------------------------------------------------------------------------
# Feature Engineering
#---------------------------------------------------------------------------------
class FeatureEngineering:

    def __init__(self, n_PCA , n_Relief ):
        self.pca = PCA(n_components=n_PCA)
        self.scaler = StandardScaler()
        self.relief = ReliefF(n_features_to_keep = n_Relief)

    def normalization_train(self, data_train, target):
        data_train = self.pca.fit_transform(data_train)
        data_train = self.scaler.fit_transform(data_train)

        dic = {val: nb for nb, val in enumerate(set(target))}
        target2nb = [dic[l] for l in target]
        data_train = np.array(data_train)
        target2nb = np.array(target2nb)
        data_train = self.relief.fit_transform(data_train, target2nb)
        return data_train

    def normalization_test(self, data_test):
        data_test = self.pca.transform(data_test)
        data_test = self.scaler.transform(data_test)
        data_test = np.array(data_test)
        data_test = self.relief.transform(data_test)
        return data_test

    def export_modules(self):
        joblib.dump(self.pca, home+"model_PCA.sav")
        joblib.dump(self.scaler, home+"model_Scaler.sav")
        joblib.dump(self.relief, home+"model_reliefF.sav")
    
    def import_modules(self):
        self.pca = joblib.load(home+"model_PCA.sav")
        self.scaler = joblib.load(home+"model_Scaler.sav")
        self.relief = joblib.load(home+"model_reliefF.sav")

FE = FeatureEngineering(50,30)

#---------------------------------------------------------------------------------
# Training ML Algorithm
#---------------------------------------------------------------------------------
def training(X_train, y_train):
    # train and evaluate a k-NN classifer on the raw pixel intensities
    # model = KNeighborsClassifier(n_neighbors=1)

    # model = SVC(gamma=0.01, C=100)
    # model = SVC(gamma='scale')
                                # , verbose = True)
    model = SVC(kernel='rbf')
    # Set the classifier as a support vector machines with polynomial kernel
    # model = SVC(kernel='linear', probability=True, tol=1e-3)
    
    model.fit(X_train, y_train)
    joblib.dump(model, home+'tools/model.sav')


# def training(shape):
#     print("\n[Read DATA]")
#     features = np.load(home+'tools/f'+str(shape)+'.npy', allow_pickle=True)
#     labels = np.load(home+'tools/l'+str(shape)+'.npy', allow_pickle=True)
#     print(features.shape, labels.shape)

#     print("\n[NORMALIZATION]")
# #     features = [Ft.normalization_zscore(f) for f in features]
#     features = FE.normalization_train(features, labels)

#     print("\n[TRAINING]")
#     (trainFeat, testFeat, trainLabels, testLabels) = train_test_split(
#         features, labels, test_size=0.1, random_state=random.seed())

#     # train and evaluate a k-NN classifer on the raw pixel intensities
#     # model = KNeighborsClassifier(n_neighbors=1)

#     # model = SVC(gamma=0.01, C=100)
# #     model = SVC(gamma='scale')
#     # , verbose = True) #Set the classifier as a support vector machines with polynomial kernel
#     model = SVC(kernel='linear', probability=True, tol=1e-3)
#     model.fit(trainFeat,trainLabels)
# #     model.fit(features, labels)
#     # save the model to disk
#     joblib.dump(model, home+'tools/model'+str(shape)+'.sav')
#     # np.save('model.npy',model,allow_pickle=True)
#     score = model.score(testFeat, testLabels)*100
#     print("[DONE]\n[INFO]\tAccuracy = ", score, "%")

def best_SVM(X_train, y_train, X_test, y_test):
    # from sklearn import metrics
    C = 200
    mean_acc = np.zeros((C-1))
    for n in tqdm(range(1, C)):
        # model = svm.SVC(gamma=0.01, C=100, kernel='linear').fit(X_train,y_train)
        model = SVC(gamma=0.01, C=C, kernel='rbf').fit(X_train,y_train)
        mean_acc[n-1] = model.score(X_test, y_test)

    plt.plot(range(1,C),mean_acc,'g')
    plt.legend(('Accuracy ', '+/- 3xstd'))
    plt.ylabel('Accuracy ')
    plt.xlabel('Number of Nabors (K)')
    plt.tight_layout()
    plt.show()
    print( "The best accuracy was with", mean_acc.max(), "with C=", mean_acc.argmax()+1) 
    return mean_acc.argmax()+1 , mean_acc.max()

def bestK_KNN(X_train, y_train, X_test, y_test):
    Ks = 200
    mean_acc = np.zeros((Ks-1))
    for n in tqdm(range(1,Ks)):
        neigh = KNeighborsClassifier(n_neighbors = n).fit(X_train,y_train)
        mean_acc[n-1] = neigh.score(X_test, y_test)

    plt.plot(range(1,Ks),mean_acc,'g')
    plt.legend(('Accuracy ', '+/- 3xstd'))
    plt.ylabel('Accuracy ')
    plt.xlabel('Number of Nabors (K)')
    plt.tight_layout()
    plt.show()
    print( "The best accuracy was with", mean_acc.max(), "with k=", mean_acc.argmax()+1) 
    return mean_acc.argmax()+1 , mean_acc.max()
#---------------------------------------------------------------------------------
# Testing Trained Model ( Evalute Model Performance )
#---------------------------------------------------------------------------------
# Classification : Accuracy, Sensitivity, Specificity, MCC

def evalute_model(X_test, y_test):
    loaded_model = joblib.load(home+'tools/model.sav')
    score = loaded_model.score(X_test, y_test)*100
    print("[INFO]\tAccuracy = ", score, "%")

    y_pred = loaded_model.predict(X_test)
    cm = confusion_matrix(y_test, y_pred)
    print('Confusion Matrix :\n', cm)
    plot_confusion_matrix(y_test, y_pred, loaded_model.classes_)

def plot_confusion_matrix(test_y, predict_y, labels):
    C = confusion_matrix(test_y, predict_y)

    #divid each element of the confusion matrix with the sum of elements in that column
    # C  =(((C.T)/(C.sum(axis=1))).T)
    
    #divid each element of the confusion matrix with the sum of elements in that row
    # C =(C/C.sum(axis=0))
        
    plt.figure(figsize=(20,8))
    sns.heatmap(C, annot=True, cmap="YlGnBu", fmt=".3f", xticklabels=labels, yticklabels=labels)
    plt.xlabel('Predicted Class')
    plt.ylabel('Original Class')
    plt.show()

def best_nb_features(features_train, labels_train, features_valid, labels_valid):
    scores, index = [], []
    for i in trange(50, desc='PCA', file= sys.stdout):
        for j in tqdm(range(10,20+i), desc='ReliefF (n_PCA = '+str(i+20)+')', leave=False, file= sys.stdout):
            FE = FeatureEngineering(i+20, j)
            features_t = FE.normalization_train(features_train, labels_train)    
            features_v = FE.normalization_test(features_valid)

            training(features_t, labels_train)
            model = SVC(kernel='rbf').fit(features_t, labels_train)
            score = model.score(features_v, labels_valid)*100
            scores.append(score)
            index.append((i+20,j))

    (i,j) = index[np.argmax(scores)]
    print("[INFO]\tAccuracy = ", np.max(scores), "%\n\tn_PCA = "+str(i)+",\tn_ReliefF = "+str(j) )
    return i, j

    
#---------------------------------------------------------------------------------
# Model Deployment
#---------------------------------------------------------------------------------
def deployment():
    features_train = np.load(home+'tools/features_train.npy', allow_pickle=True)
    labels_train = np.load(home+'tools/labels_train.npy', allow_pickle=True)
    features_valid = np.load(home+'tools/features_valid.npy', allow_pickle=True)
    labels_valid = np.load(home+'tools/labels_valid.npy', allow_pickle=True)

    features = np.concatenate([features_train, features_valid])
    labels = np.concatenate([labels_train, labels_valid])

    FE = FeatureEngineering(40, 30)
    features = FE.normalization_train(features, labels)
    FE.export_modules()

    model = SVC(kernel='rbf')
    model.fit(features,labels)

    joblib.dump(model, home+'model.sav')

def prediction():
    loaded_model = joblib.load(home+'model.sav')

    FE = FeatureEngineering(40, 30)
    FE.import_modules()

    features_test = []
    imgIds =[]
    files = glob(home+'test'+os.path.sep+"*.*")
    for img_path in tqdm(files, desc="Test", file=sys.stdout):
        img = cv2.imread(img_path)
        feature = extract_features(img)
        if feature is None:
            continue
        features_test.append(feature)
        imgIds.append(img_path.split(os.path.sep)[-1])

    features_test = np.array(features_test)
    features_test = FE.normalization_test(features_test)
    labels_test = [loaded_model.predict([feature])[0] for feature in features_test]

    labels_test = ['Plante_'+str(l) for l in  labels_test]
    df = pd.DataFrame({'nom_image': imgIds, 'classe_predite': labels_test})
    df.to_csv(home+'csv_file.csv', index=False, sep=",")


#---------------------------------------------------------------------------------
# La fonction main
#---------------------------------------------------------------------------------
if __name__ == '__main__':

    features_train, labels_train = indexing(home+"train",'train')
    features_valid, labels_valid = indexing(home+"validation",'valid')

    # if you have all ready extracting the images features    
    # features_train = np.load(home+'tools/features_train.npy', allow_pickle=True)
    # labels_train = np.load(home+'tools/labels_train.npy', allow_pickle=True)
    # features_valid = np.load(home+'tools/features_valid.npy', allow_pickle=True)
    # labels_valid = np.load(home+'tools/labels_valid.npy', allow_pickle=True)
    print(features_train.shape, features_valid.shape)
    
    i, j = best_nb_features(features_train, labels_train, features_valid, labels_valid)
    # i, j = 24, 21
    # FE = FeatureEngineering(i, j)
    # features_train = FE.normalization_train(features_train, labels_train)    
    # features_valid = FE.normalization_test(features_valid)

    # training(features_train, labels_train)
    # evalute_model(features_valid, labels_valid)

    # best_SVM(features_train, labels_train, features_valid, labels_valid)

    # deployment(i, j)

    # prediction()

    print("[DONE]")




'''
    Pour instaler mahotas :
        Install Microsoft Visual C++ 14.0 build tools : https://go.microsoft.com/fwlink/?LinkId=691126

        or
        pip install --upgrade setuptools
        pip install mahotas
        
        or the Binary install it the simple way!:
            pip install --only-binary :all: mahotas
            
        or :
            conda config --add channels conda-forge
            conda install mahotas
        or :
            conda install -c https://conda.anaconda.org/conda-forge mahotas
'''