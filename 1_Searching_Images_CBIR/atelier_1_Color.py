# -*- coding: utf-8 -*-
"""
@author: EL KHOU Mohammed

Ci-joint l'Atelier 1 Image mining:
1- La base de donnees
2- L'image requete
l'objectif est de developper un systeme CBIR utilisant Python ou Matlab 
permettant d'afficher 5 images, de la base de donn¨¦es nomm¨¦e DataSet, 
les plus similaires  a l'image requete.
"""
import cv2, glob, csv
import numpy as np
from math import sqrt
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import scipy
from scipy.spatial import distance

def getFeatures(img):
    # img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    R, G, B = img[:,:,0], img[:,:,1], img[:,:,2]
#    print(R)
    feature = [np.mean(R), np.std(R), np.mean(G), np.std(G), np.mean(B), np.std(B)]
#    print(type(feature/np.mean(feature)))
#    print(type(np.mean(R)))
    return feature/np.mean(feature)

# 1- Phase hors ligne appelé souvent indexation où pour chaque image un vecteur descripteur 
# sera extrait et sauvegarder sous forme de base d’indexes 
def Indexing(data_path, data_set):
    # Extraire les paths des images de type jpg qui se trouve dans le dossier : data_path
    all_files = glob.glob(str(data_path)+"/*.jpg")
    # ovrir un fichier de type csv on mode ecriture pour stocker les entites extrait pour ces images 
    csv_file = open(data_set, mode='w')
    
    # Créez un objet qui fonctionne comme un écrivain mais mappe les dictionnaires sur les lignes de sortie. 
    fieldnames = ['imgName', 'vect']
    writer = csv.DictWriter(csv_file, fieldnames=fieldnames)

    writer.writeheader()
    print("Indexing : ", end="", flush=True)
    # Parcourire l'ensemble des images de base de donnees
    for imagePath in all_files:
        imageID = imagePath[imagePath.rfind("\\") + 1:]

        img = cv2.imread(imagePath)
        feature = getFeatures(img)
        

        features = np.array(feature).astype('|S20')

#        features = [str(f) for f in feature]
        writer.writerow({'imgName': imageID, 'vect': b",".join(features[:])})
        print(".", end="", flush=True)
#        break
    print("[DONE]")
    csv_file.close()

# 2- Phase en ligne : c’est la recherche d’images similaires pour une image requête. 
#Le même type de descripteur sera extrait de l’image requête et comparé avec la base d’indexes.

def Searching(data_path, data_set, imgName):
    img = cv2.imread(imgName)
    req = getFeatures(img)
    # ovrir un fichier de type csv on mode lecture 
    csv_file = open(data_set, mode='r')
    csv_reader = csv.DictReader(csv_file)
   
    top5 = [['',float(1000)] for _ in range(5)]
    for row in csv_reader:
       # parcourir chauque ligne du fichier et on  prend le vecteur descriptive qui est de type String 
       # et en le transfere en vecteur du vecteur du point de type float32
        vect = row["vect"].split('\'')[1].split(',')
        feature = [np.float64(x) for x in vect] # Ajouter vecteur dans notre vecteur descriptive
        
        feature = np.array(vect).astype(np.float64)

#        s = 0
#        for v_i, u_i in zip(feature, req):
#            s += (v_i - u_i)**2
#        dist = s ** 0.5

        dist = sqrt(sum([(xi-yi)**2 for xi,yi in zip(feature, req)]))
        
        # trie par insertion
        i = 0
        while i < len(top5) and  dist > top5[i][1]:
            i += 1
        if i < len(top5) :
            for j in reversed(range(i+1,len(top5))):
                top5[j] = top5[j-1]
            top5[i] = [row["imgName"], dist]                     
    csv_file.close()
    
    print(top5)
    for i, top in enumerate(top5) :
        if top[0]:
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

# La fonction main
if __name__ == '__main__':
    data_path = 'obj_decoys/'
    data_set = 'dataSetColor.csv'

    Indexing(data_path , data_set)
    Searching(data_path, data_set, "ImageRequete.jpg")
#    Searching(data_path, data_set, data_path+"382020.jpg")
    # Searching(data_path, data_set, data_path+"655070.jpg")
