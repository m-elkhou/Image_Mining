# -*- coding: utf-8 -*-
"""
Created on Fri Oct  4 14:11:21 2019

@author: EL KHOU Mohammed

Ci-joint l'Atelier 1 Image mining:
1- La base de donnees
2- L'image requete
l'objectif est de developper un systeme CBIR utilisant Python ou Matlab 
permettant d'afficher 5 images, de la base de donn¨¦es nomm¨¦e DataSet, 
les plus similaires  a l'image requete.
"""
import cv2, glob, imutils, io, sys, csv, pandas
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

'''     Scale Invariant Feature Transform (SIFT)
        keypoints : sont les points d’intérêt dans une image
'''    
# Create SIFT Feature Detector object
sift = cv2.xfeatures2d.SIFT_create()

FLANN_INDEX_KDITREE=0
flannParam = dict(algorithm=FLANN_INDEX_KDITREE,tree=5)
flann = cv2.FlannBasedMatcher(flannParam,{})
MIN_MATCH_COUNT = 2


                
# 1- Phase hors ligne appelé souvent indexation où pour chaque image un vecteur descripteur 
# sera extrait et sauvegarder sous forme de base d’indexes 
def Indexing(data_path, data_set):
    # Extraire les paths des images de type jpg qui se trouve dans le dossier : data_path
    all_files = glob.glob(str(data_path)+"/*.jpg")
    # ovrir un fichier de type csv on mode ecriture pour stocker les entites extrait pour ces images 
    csv_file = open(data_set, mode='w')
    
    # Créez un objet qui fonctionne comme un écrivain mais mappe les dictionnaires sur les lignes de sortie. 
    fieldnames = ['imgName', 'trainDesc']
    writer = csv.DictWriter(csv_file, fieldnames=fieldnames)

    writer.writeheader()
    print("Indexing : ", end="", flush=True)
    # Parcourire l'ensemble des images de base de donnees
    for imagePath in all_files:
        img = cv2.imread(imagePath,0)
        _, trainDesc = sift.detectAndCompute(img,None)
        
#        Array 2 String [[1,2,3,..],[1,2,3,..],..] --> "#|1|2|3|..#|1|2|..#|.."
        Desk = ""
        for i in trainDesc:
            Desk += '#'
            for j in i:
                Desk +='|'+ str(j)
        
        imageID = imagePath[imagePath.rfind("\\") + 1:]
        writer.writerow({'imgName': imageID, 'trainDesc': Desk})
        print(".", end="", flush=True)
    print("[DONE]")
    csv_file.close()

# 2- Phase en ligne : c’est la recherche d’images similaires pour une image requête. 
#Le même type de descripteur sera extrait de l’image requête et comparé avec la base d’indexes.

def Searching(data_path, data_set, imgName):
    top5 = []
    img = cv2.imread(imgName,0)
    queryKP, queryDesc = sift.detectAndCompute(img,None)
    
    # ovrir un fichier de type csv on mode lecture 
    csv_file = open(data_set, mode='r')
    csv_reader = csv.DictReader(csv_file)
    
    for row in csv_reader:
        # parcourir chauque ligne du fichier et on  prend le vecteur descriptive qui est de type String 
        # et en le transfere en vecteur du vecteur du point de type float32
        trainDesc=[]
        desc = row["trainDesc"][1:].split('#')
        for i in desc:
            elem = i.split('|')
            trainDesc.append([float(x) for x in elem[1:]]) # Ajouter vecteur dans notre vecteur descriptive
            
        trainDesc = np.asarray(trainDesc,np.float32)
        
        matches = flann.knnMatch(queryDesc, trainDesc, k=2)
                    
        cppGoodMatch=0
        if matches:
            for m,n in matches:
                if m.distance < 0.75*n.distance:
                    cppGoodMatch+=1
            
            if(cppGoodMatch > MIN_MATCH_COUNT):
                if len(top5) < 5 :
                    top5.append(['',0])
                    
                # trie par insertion
                i = 0
                while i < len(top5) and  top5[i][1] >= cppGoodMatch:
                    i += 1
                if i < 5 :
                    for j in reversed(range(i+1,len(top5))):
                        top5[j] = top5[j-1]
                    top5[i] = [row["imgName"], cppGoodMatch]  
                        
    csv_file.close()
#    fig=plt.figure(figsize=(15, 12), dpi= 80, facecolor='w', edgecolor='k')
    for i, top in enumerate(top5) :
        print(top[0])
        plt.subplot(2, 3, i+2)
        plt.title(top[0])
        plt.xticks([]),plt.yticks([])
        image = mpimg.imread(data_path+top[0])
        plt.imshow(image)
    
    plt.subplot(2, 3, 1)
    plt.xticks([]),plt.yticks([])
    plt.title('ImageRequete')
    plt.imshow(mpimg.imread(imgName))
    plt.rcParams['figure.figsize'] = [15, 7]
    plt.show()
#    fig.show()

# La fonction main
if __name__ == '__main__':
    data_path = 'obj_decoys/'
    data_set = 'dataSet.csv'

    # Indexing(data_path , data_set)
    # Searching(data_path, data_set, "ImageRequete.jpg")
#    Searching(data_path, data_set, "382020.jpg")
    Searching(data_path, data_set, data_path+"655070.jpg")
    sys.exit