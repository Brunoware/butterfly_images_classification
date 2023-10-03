import os
import cv2
import pywt
import numpy as np
import pandas as pd

# CONSTANTS AND VARIABLES
FOLDER_PATH = 'datasets/Mariposas'
OUTPUT_FILEPATH = 'datasets/wavelet_tranformed_images_2.csv'
NEW_SIZE_IMAGES = (256, 256) # 256 x 256 pixels
LEVEL = 6
characteristic_vectors = []
target_list = []
file_list = os.listdir(FOLDER_PATH)

def read_resize_gray(file_name):
    file_path = os.path.join(FOLDER_PATH, file_name)
    image = cv2.imread(file_path)
    resized_image = cv2.resize(image, NEW_SIZE_IMAGES)
    gray_image = cv2.cvtColor(resized_image, cv2.COLOR_BGR2GRAY)
    return gray_image

for file_name in file_list:
    # reading and converting image into a 2D-gray one
    gray_image = read_resize_gray(file_name)
    
    # extracting features acording to the set levels
    coeffs = pywt.wavedecn(gray_image, 'db3', mode = 'symmetric', level = LEVEL)
    
    # selecting the extracted feature which is the first value of the array
    # then it is flattened
    cA = coeffs[0].flatten()
    
    # adding to the corresponding lists
    characteristic_vectors.append(cA)
    target_list.append(int(file_name[:3]))
    
# putting all together
data_matrix = pd.DataFrame(np.vstack(characteristic_vectors))
data_matrix['target'] = target_list

# saving to csv file
data_matrix.to_csv(OUTPUT_FILEPATH, index = False) 