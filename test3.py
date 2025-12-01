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

"""
DATA_PATH = "datasets/cats_and_dogs"
LABEL_TO_IDX = {"cats": 0, "dogs": 1}

dataimagecats = io.imread_collection("datasets/cats_and_dogs/cats/*.jpg")
dataimagedogs = io.imread_collection("datasets/cats_and_dogs/dogs/*.jpg")

minx=133
miny=133

def rgb(i):
    if i%3==0:
        return "R"
    elif i%3==1:
        return "G"
    else:
        return "B"

pixels = ["pixel"+str((i//3)+1)+rgb(i) for i in range(0,minx*miny*3)]
pixels.insert(0,"cat/dog")


img_resizedcats=[]
for img in dataimagecats:
    img_resizedcats.append(np.insert(np.ravel(transform.resize(img,(minx,miny))), 0, 0))
# si on met ,preserve_range=True on gare le meme range mais quand on le met pas les données sont normalisèes

img_resizeddogs=[]
for img in dataimagedogs:
    img_resizeddogs.append(np.insert(np.ravel(transform.resize(img,(minx,miny))), 0, 1))

df = pd.DataFrame(columns=pixels,data=img_resizedcats+img_resizeddogs)
df.to_csv("data_part2.csv", index=False) 
"""


df = pd.read_csv("data_part2.csv")

df = df.dropna()

pixel_cols = [col for col in df.columns if col != "cat/dog"]
X = df[pixel_cols].values
y = df["cat/dog"].values

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train, y_train)

y_pred = knn.predict(X_test)

# Accuracy
acc = accuracy_score(y_test, y_pred)
print("Accuracy:", acc)


cm = confusion_matrix(y_true=y_test, y_pred=y_pred)

# Affichage
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["cat", "dog"])



""" pour générer l'image de la matrice de confusion
disp.plot()
plt.savefig("confusion_matrix.png", dpi=300, bbox_inches="tight")
plt.close()
"""

# 1 Encoder les labels en binaire
lb = LabelBinarizer()
y_test_bin = lb.fit_transform(y_test)  # chat=0, dog=1

# 2 Obtenir les probabilités de la classe positive
y_score = knn.predict_proba(X_test)[:,1]  # probabilité de "dog"

# 3 Calculer FPR, TPR et AUC
fpr, tpr, thresholds = roc_curve(y_test_bin, y_score)
roc_auc = auc(fpr, tpr)

# 4 Plot
plt.figure(figsize=(6,6))
plt.plot(fpr, tpr, color="darkorange", lw=2, label="ROC curve (AUC = %0.2f)" % roc_auc)
plt.plot([0, 1], [0, 1], color="navy", lw=2, linestyle="--")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve - k-NN Classifier")
plt.legend(loc="lower right")

# 5 Sauvegarder le plot en fichier image
plt.savefig("roc_curve.png", dpi=300, bbox_inches="tight")
plt.close()  # fermer la figure pour libérer la mémoire