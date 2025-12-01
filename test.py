#cellule pour les imports

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

import matplotlib.colors as mcolors
from skimage import io, transform
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score,confusion_matrix, ConfusionMatrixDisplay
from sklearn.metrics import RocCurveDisplay, auc, roc_curve
from sklearn.preprocessing import LabelBinarizer
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from scipy.stats import mode
from sklearn.decomposition import PCA
import plotly.express as px
from plotly.offline import init_notebook_mode, iplot

DATA_PATH = "datasets/cats_and_dogs"
LABEL_TO_IDX = {"cats": 0, "dogs": 1}

dataimagecats = io.imread_collection("datasets/cats_and_dogs/cats/*.jpg")
dataimagedogs = io.imread_collection("datasets/cats_and_dogs/dogs/*.jpg")

print(dataimagecats[0])


# 2)  trouver la plus petite dimension d'image pour redimensionner toutes les images à cette taille




"""
dataimages = list(dataimagecats) + list(dataimagedogs)

minx = 2000
miny = 2000
for img in dataimages:
    if(img.shape[0]<minx):
        minx=img.shape[0]
    if(img.shape[1]<miny):
        miny=img.shape[1]
print(minx,miny)
"""
minx=133
miny=133



df = pd.DataFrame();

# redimensionner les images
img_resizedcats=[]
for img in dataimagecats:
    temp =np.ravel(transform.resize(img, (minx, miny)))
    img_resizedcats.append(np.append(temp,0))

img_resizeddogs=[]
for img in dataimagedogs:
    temp= np.ravel(transform.resize(img,(minx,miny)))
    img_resizeddogs.append(np.append(temp,1))

# for i in img_resizedcats:
#     print(i.shape)
# for i in img_resizeddogs:
#     print(i.shape)


def rgb(i):
    if i%3==0:
        return "R"
    elif i%3==1:
        return "G"
    else:
        return "B"

pixels = ["pixel"+str((i//3)+1)+rgb(i) for i in range(0,minx*miny*3)]
col = pixels[0:255]

# print(img_resizedcats)
df = pd.DataFrame()

