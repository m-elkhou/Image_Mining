import cv2
from glob import glob
import numpy as np
# import pandas as pd
# from sklearn import svm
from sklearn.neighbors import KNeighborsClassifier
# import random
import joblib
from tqdm import tqdm
from sklearn.model_selection import ShuffleSplit
import matplotlib.pyplot as plt
from datetime import datetime, timedelta

home = 'D:/WISD/S3/Image_Mining/Atelier_7_HandRecongnition/'

# ----------------------------------------------------------------------------------------------------
# Loading images and mask
# ----------------------------------------------------------------------------------------------------
def loding(data_train_path, data_GTtrain_path):
    print('# Loading : [ START ]')
    features = []
    labels = []

    all_img_path = np.array(glob(data_train_path+"*.*"))
    all_mask_paths = np.array(glob(data_GTtrain_path+"*.*"))

    ss = ShuffleSplit(n_splits=1, test_size=0.5, random_state=42)
    for _, index in ss.split(all_img_path):
        paths = zip( all_img_path[index], all_mask_paths[index])

    pbar = tqdm(total=len(all_img_path[index]))
    for img_path, mask_path in paths : 

        img = cv2.imread(img_path)
        img = img[3:198,3:198,:]

        imgYCC = cv2.cvtColor(img, cv2.COLOR_BGR2YCR_CB)
        imgYCC = imgYCC.reshape((-1, 3))

        mask = cv2.imread(mask_path,0)
        _, mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
        mask = mask.flatten()

        for pixel in imgYCC:
            features.append(np.array(pixel))
        labels.extend(mask)

        pbar.update(1)
    pbar.close()

    features = np.array(features)
    labels = np.array(labels)

    np.save(home + 'features_seg.npy', features, allow_pickle=True)
    np.save(home + 'labels_seg.npy', labels, allow_pickle=True)
    print('# Loading : [ DONE ]')
    
# ----------------------------------------------------------------------------------------------------
# Predicting
# ----------------------------------------------------------------------------------------------------
def predicting():
    print('# Predicting : [ START ]')
    date = datetime.today()
    features =np.load(home + 'features_seg.npy', allow_pickle=True)
    labels = np.load(home + 'labels_seg.npy', allow_pickle=True)

    # model = svm.SVC(gamma=0.01, C=1, kernel='rbf').fit(features, labeles)

    model = KNeighborsClassifier(n_neighbors = 10).fit(features, labels)

    # from sklearn.ensemble import RandomForestRegressor
    # model=RandomForestRegressor(verbose=2,n_estimators=100)
    # model.fit(features, labels)
    joblib.dump(model, model_path)
    print('# Predicting : [ DONE ]')
    print('# Time left ', date - datetime.today())

# ----------------------------------------------------------------------------------------------------
# Segmentation
# ----------------------------------------------------------------------------------------------------
def segmentation(img):
    img = img[3:198,3:198,:]
    imgYCC = cv2.cvtColor(img, cv2.COLOR_BGR2YCR_CB)
    imgYCC = imgYCC.reshape((-1, 3))

    model = joblib.load(model_path)
    mask = model.predict(imgYCC)
    mask = np.array(mask).reshape((195,195))

    # Y = np.multiply(imgYCC[:,:,0], mask)
    # Cb = np.multiply(imgYCC[:,:,1], mask)
    # Cr = np.multiply(imgYCC[:,:,2], mask)
    # skin = np.dstack((Y,Cb,Cr)).astype(np.uint8)

    # apply a series of erosions and dilations to the mask using an elliptical kernel
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    mask = cv2.erode(mask, kernel, iterations = 2)
    mask = cv2.dilate(mask, kernel, iterations = 2)
    # blur the mask to help remove noise, then apply the mask to the frame
    mask = cv2.GaussianBlur(mask, (3, 3), 0)

    # _, mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    cnt = max(contours, key=cv2.contourArea)
    # Finding the convex hull of largest contour 
    # hull = cv2.convexHull(cnt,returnPoints=True)    
    # cv2.drawContours(frameD,[hull],0,(0, 0, 255),2) # 5dar :  (0,255,0)
    x, y, width, height = cv2.boundingRect(cnt)
    # Straight Bounding Rectangle
    # from : https://docs.opencv.org/3.1.0/dd/d49/tutorial_py_contour_features.html
    # Let (x,y) be the top-left coordinate of the rectangle and (w,h) be its width and height.
    

    mask = cv2.merge((mask, mask, mask))

    skin = cv2.bitwise_and(img, mask)

    # skin = cv2.rectangle(skin, (x, y), (x + width, y + height), (0,255,0), 2) 
    new_img = skin[y:(y + height),x:(x + width)]

    plt.subplot(2, 2, 1)
    plt.xticks([]),plt.yticks([])
    plt.title('Image Originale')
    plt.imshow(img)

    plt.subplot(2, 2, 2)
    plt.xticks([]),plt.yticks([])
    plt.title('mask')
    plt.imshow(mask)

    plt.subplot(2, 2, 3)
    plt.xticks([]),plt.yticks([])
    plt.title('skin')
    plt.imshow(skin)

    plt.subplot(2, 2, 4)
    plt.xticks([]),plt.yticks([])
    plt.title('new skin')
    plt.imshow(new_img)

    plt.rcParams['figure.figsize'] = [15, 7]
    plt.show()

# La fonction main
if __name__ == '__main__':
    home = 'D:/WISD/S3/Image_Mining/Atelier_7_HandRecongnition/'
    data_train_path = home + 'training/training/'
    data_GTtrain_path = home + 'training/GTtraining/'
    model_path = home + "model_seg0.sav"

    # loding(data_train_path, data_GTtrain_path)
    # predicting()
    img = cv2.imread(home+'test/A2.jpg')
    segmentation(img)

    print('# Main : [ DONE ] ')