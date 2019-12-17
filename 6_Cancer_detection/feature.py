import cv2
import imutils
import mahotas
import numpy as np
import pandas as pd
import skimage.feature.texture as sft
from scipy import stats
from sklearn.preprocessing import StandardScaler , MinMaxScaler
from sklearn.decomposition import PCA
from ReliefF import ReliefF
from sklearn.feature_selection import SelectKBest, f_classif


class Feature:

    def __init__(self):
        self.scaler = StandardScaler()
        self.mscaler = MinMaxScaler(feature_range=(0, 1))
        self.pca = PCA(n_components=30)
        self.relief = ReliefF(n_features_to_keep=20)

    # Recherche par Couleur
    def extract_color_moments(self, img):
            R, G, B = img[:,:,0], img[:,:,1], img[:,:,2]
            feature = [np.mean(R), np.std(R), np.mean(G), np.std(G), np.mean(B), np.std(B)]
            return feature/np.mean(feature)
        #    (means, stds) = cv2.meanStdDev(img)
        #    stats = np.concatenate([means, stds]).flatten()

    def extract_color_histogram(self, image, bins=(8, 2, 2)):
    	# extract a 3D color histogram from the HSV color space using
	    # the supplied number of `bins` per channel
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        hist = cv2.calcHist([hsv], [0, 1, 2], None, bins, [0, 256, 0, 256, 0, 256])
    	# handle normalizing the histogram if we are using OpenCV 2.4.X
        if imutils.is_cv2():
            hist = cv2.normalize(hist)
    	# otherwise, perform "in place" normalization in OpenCV 3 (I
        # personally hate the way this is done
        else:
            cv2.normalize(hist, hist) 
	    # return the flattened histogram as the feature vector
        return hist.flatten()

    def extract_texture_features(self, img):
        # Basée sur l'analayse de textures par la GLCM (Gray-Level Co-Occurrence Matrix)
        # Le vecteur de taille 1x4 contiendra [Contrast, Correlation, Energy, Homogeneity]
        # convert to grayscale
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
        glcm = sft.greycomatrix(img_gray, distances = [1], angles = [0], symmetric=True, normed=True)
        # glcm = sft.greycomatrix(img_gray, distances = [1], angles = [0, np.pi/4, np.pi/2, 3*np.pi/4], symmetric=True, normed=True)
        props = ['contrast', 'correlation', 'energy', 'homogeneity']
        texture_features = [sft.greycoprops(glcm, prop).ravel()[0] for prop in props]
        texture_features = texture_features / np.sum(texture_features)
        return texture_features


    def textureFeature(self, img):
        from skimage import feature
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
        glcm = feature.greycomatrix(img_gray, distances = [1], angles = [0], symmetric=True, normed=True)
        props = ['contrast', 'correlation', 'energy', 'homogeneity']
        texture_features = [feature.greycoprops(glcm, prop).ravel()[0] for prop in props]
        texture_features = texture_features / np.sum(texture_features)
        return texture_features


    def extract_haralick_texture(self, image):
        # convert the image to grayscale
        img_gray = cv2.cvtColor(np.uint8(image), cv2.COLOR_BGR2GRAY)
        # compute the haralick texture feature vector
        features = mahotas.features.haralick(img_gray).ravel()
        # # take the mean of it and return it
        # ht_mean = textures.mean(axis=0)
        # return the resultconda
        return features

    # Recherche par Forme
    def extract_hu_moments(self, img): 
        # Hu Moments that quantifies shape of the Object in img.
        image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        feature = cv2.HuMoments(cv2.moments(image)).flatten()
        return feature
 
    def extract_sift_features(self, img):
        # Create SIFT Feature Detector object
        sift = cv2.xfeatures2d.SIFT_create()
        img = cv2.cvtColor(np.uint8(img), cv2.COLOR_BGR2GRAY)
        _, desc = sift.detectAndCompute(img, None)
        # Stack all the descriptors vertically in a numpy array
        descriptors = desc[1]
        descriptors = []
        for descriptor in desc[:]:
            # descriptors = np.hstack((descriptors, descriptor))  # Stacking the descriptors
            descriptors.extend(descriptor)
        # print(len(descriptors))


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

        # print(len(global_features),type(global_features),global_features)
        # global_features = stats.zscore(np.array(global_features))
        # print(len(global_features),type(global_features),global_features)

        return global_features

    def getFeatures(self, img):
        # img = cv2.resize(cv2.imread(imagePath), (32, 32))

        color_moments = self.extract_color_moments(img)
        color_histogram = self.extract_color_histogram(img)
        texture = self.extract_texture_features(img)
        haralick_texture  = self.extract_haralick_texture(img)
        hu_moments = self.extract_hu_moments(img)
        # sift_f = extract_sift_features(img)

        # Concatenate global features
        global_features = np.concatenate([hu_moments, haralick_texture, color_histogram]).flatten()
        # global_features = np.concatenate([color_moments, color_histogram, texture, hu_moments]).flatten() # score = 93.5
        # global_feature = np.concatenate([color_histogram]).flatten() # scor = 94.5
        # global_features = np.concatenate([color_histogram, haralick_texture]).flatten() # 11 err
        # global_features = np.hstack((color_histogram, haralick_texture)) # 96% 2 _
        # global_features = np.hstack((haralick_texture, color_histogram)) # 98% 2 _
        # global_features = np.hstack((color_moments, color_histogram, haralick_texture, texture, hu_moments)) 
        # global_features = np.hstack((color_moments, color_histogram, haralick_texture, hu_moments))

        return global_features

    def Scaler_train(self, feature):
        return self.scaler.fit_transform(feature)

    def Scaler_test(self, feature):
        return self.scaler.transform(feature)

    def zscore(self, feature):
        return stats.zscore(np.array(feature))

    def MinMaxScaler_train(self, features):
        mscaler = MinMaxScaler(feature_range=(0, 1))
        rescaled_features = mscaler.fit_transform(features)
        return rescaled_features

    def standarization_basic(self, feature):
        # feature = feature/max(feature)
        # feature = (feature - min(feature)) / (max(feature) - min(feature))
        feature = (feature - np.mean(feature)) / np.std(feature)
        return feature

    def Relief_train(self, data, target):
        dic = { val : nb for nb, val in enumerate(set(target))}
        target2nb = np.array([dic[l] for l in target])
        data = np.array(data)
        X_train = self.relief.fit_transform(data, target2nb)
        return X_train

    def Relief2_test(self, data):
        return self.relief.transform(data)
        
    def SelectKBest(self, data_train, target): #Selection of kbest features
        bestK = SelectKBest(f_classif, k=20)
        X_new = bestK.fit_transform(data_train, target)
        return X_new

    def normalization_train(self, data_train, target):
        
        data_train = self.pca.fit_transform(data_train)
        data_train = self.mscaler.fit_transform(data_train)
        # data_train = self.scaler.fit_transform(data_train)

        dic = { val : nb for nb, val in enumerate(set(target))}
        target2nb = [dic[l] for l in target]
        data_train = np.array(data_train)
        target2nb = np.array(target2nb)
        data_train = self.relief.fit_transform(data_train, target2nb)
        return data_train

    def normalization_test(self, data_test):
        
        data_test = self.pca.transform(data_test)
        data_test = self.mscaler.transform(data_test)
        # data_test = self.scaler.transform(data_test)
        data_test = np.array(data_test)
        data_test = self.relief.transform(data_test)
        return data_test

    def list_2_df(self, data_lst, target_lst):
        clm = [i for i in range(len(target_lst))]
        df = pd.DataFrame(data_lst, columns = clm)
        return df  

'''
    pour instaler mahotas :
        install Microsoft Visual C++ 14.0 build tools : https://go.microsoft.com/fwlink/?LinkId=691126

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
