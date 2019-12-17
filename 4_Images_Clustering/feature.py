import cv2
import imutils
import mahotas
import numpy as np
import skimage.feature.texture as sft
from sklearn.preprocessing import StandardScaler
from scipy import stats

class Feature:

    # Recherche par Couleur
    def extract_color_moments(self, img):
            R, G, B = img[:,:,0], img[:,:,1], img[:,:,2]
            feature = [np.mean(R), np.std(R), np.mean(G), np.std(G), np.mean(B), np.std(B)]
            return feature/np.mean(feature)
        #    (means, stds) = cv2.meanStdDev(img)
        #    stats = np.concatenate([means, stds]).flatten()

    def extract_color_histogram(self, image, bins=(8, 8, 8)):
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
        # return the result
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
        color_moments = self.extract_color_moments(img)
        color_histogram = self.extract_color_histogram(img)
        texture  = self.extract_texture_features(img)
        haralick_texture  = self.extract_haralick_texture(img)
        hu_moments = self.extract_hu_moments(img)
        # sift_f = extract_sift_features(img)

        # Concatenate global features
        # global_features = np.concatenate([color_moments, color_histogram, texture, hu_moments]).flatten() # score = 93.5
        # global_feature = np.concatenate([color_histogram]).flatten() # scor = 94.5
        # global_features = np.concatenate([color_histogram, haralick_texture]).flatten() # 11 err
        # global_features = np.hstack((color_histogram, haralick_texture)) # 96% 2 _
        global_features = np.hstack((haralick_texture, color_histogram)) # 98% 2 _
        # global_features = np.hstack((color_moments, color_histogram, haralick_texture, texture, hu_moments)) 
        # global_features = np.hstack((color_moments, color_histogram, haralick_texture, hu_moments))
        return global_features
