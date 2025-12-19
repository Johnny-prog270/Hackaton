import numpy as np
import pandas as pd
from skimage import io, transform
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# ==============================
# Chargement des images satellite
# ==============================

desert_images = io.imread_collection("datasets/satellite/desert/*.jpg")
cloudy_images = io.imread_collection("datasets/satellite/cloudy/*.jpg")
green_area_images = io.imread_collection("datasets/satellite/green_area/*.jpg")
water_images = io.imread_collection("datasets/satellite/water/*.jpg")


minx, miny = 32, 32

def rgb(i):
    return ["R", "G", "B"][i % 3]

pixels = ["pixel" + str(i+1) + c for i in range(minx*miny) for c in ['R','G','B','A']]
pixels.append("label")

def process_images_rgba(images, label):
    data = []
    for img in images:

        img_resized = transform.resize(img, (minx, miny, 3), preserve_range=True)

        img_resized = img_resized.astype(np.float32) / 255.0
        
        alpha_channel = np.ones((minx, miny, 1), dtype=np.float32)
        img_rgba = np.concatenate([img_resized, alpha_channel], axis=2)
        
        img_flat = img_rgba.flatten()
        img_flat_with_label = np.append(img_flat, label)  
        data.append(img_flat_with_label)
        
    return data

desert_data = process_images_rgba(desert_images, 0)
cloudy_data = process_images_rgba(cloudy_images, 1)
green_data = process_images_rgba(green_area_images, 2)
water_data = process_images_rgba(water_images, 3)

df = pd.DataFrame(
    data=desert_data + cloudy_data + green_data + water_data,
    columns=pixels
)

df.dropna(inplace=True)
df.to_csv("satellite_dataset.csv", index=False)
print(df.describe())

# ==============================
# Split train / test
# ==============================

X = df.drop(columns="label").values
y = df["label"].values

seed = 42
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=seed, stratify=y
)

# ==============================
# Standardisation
# ==============================

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

print("len(X_train) :", len(X_train))
print("len(X_test)  :", len(X_test))

import plotly.express as px
import pandas as pd
from sklearn.decomposition import PCA


# PCA 3D
pca_3d = PCA(n_components=3)
X_pca_3d = pca_3d.fit_transform(X_train)

df_pca_3d = pd.DataFrame({
    'PC1': X_pca_3d[:,0],
    'PC2': X_pca_3d[:,1],
    'PC3': X_pca_3d[:,2],
    'label': y_train
})

fig_3d = px.scatter_3d(
    df_pca_3d, x='PC1', y='PC2', z='PC3',
    color=df_pca_3d['label'].astype(str),
    title="PCA 3D des images satellite",
    labels={'color':'Classe'}
)
fig_3d.show()  # s'ouvre dans le navigateur

