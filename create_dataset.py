import os
import cv2
import pywt
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA

# CONSTANTS AND VARIABLES
FOLDER_PATH = 'datasets/Mariposas/images'
OUTPUT_FILEPATH = 'final.csv'
NEW_SIZE_IMAGES = (224, 224) # 224x224 pixels
NUM_COMPONENTS_PCA = 240 # at least 80% cumulative variance ratio
characteristic_vectors = []
target_list = []
file_list = os.listdir(FOLDER_PATH)

# PCA
pca = PCA(n_components=NUM_COMPONENTS_PCA)

for file_name in file_list:
    file_path = os.path.join(FOLDER_PATH, file_name)
    image = cv2.imread(file_path)
    resized_image = cv2.resize(image, NEW_SIZE_IMAGES)
    gray_image = cv2.cvtColor(resized_image, cv2.COLOR_BGR2GRAY)
    coeffs = pywt.dwt2(gray_image, 'bior1.3')
    approx, (horizontal_detail, vertical_detail, diagonal_detail) = coeffs
    flattened_coeffs = np.concatenate([approx.flatten(), horizontal_detail.flatten(),
                            vertical_detail.flatten(), diagonal_detail.flatten()])
    normalized_coeffs = (flattened_coeffs - flattened_coeffs.mean()) / flattened_coeffs.std()
    characteristic_vectors.append(normalized_coeffs)
    target_list.append(int(file_name[:3]))

data_matrix = np.vstack(characteristic_vectors)

# Applying PCA
principal_components = pca.fit_transform(data_matrix)

explained_variance = pca.explained_variance_
explained_variance_ratio = pca.explained_variance_ratio_
cumulative_variance_ratio = np.cumsum(explained_variance_ratio)

print("Varianza explicada por cada componente principal:")
print(explained_variance)
print("\nVarianza explicada relativa (proporción) por cada componente principal:")
print(explained_variance_ratio)
print("\nVarianza acumulada explicada:")
print(cumulative_variance_ratio)

# Convert to pandas, then to csv file
df = pd.DataFrame(principal_components)
df['target'] = target_list
df.to_csv(OUTPUT_FILEPATH, index=False)