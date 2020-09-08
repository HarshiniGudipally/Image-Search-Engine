import os
import cv2
import csv

import numpy as np


IMAGES_FOLDER = '/home/harshini/workspace/keras-autoencoder-cbir-20200908T054132Z-001/keras-autoencoder-cbir/dataset/test'
OUTPUT = '/home/harshini/workspace/keras-autoencoder-cbir-20200908T054132Z-001/keras-autoencoder-cbir/dataset/output'

### Initialise empty numpy arrays

data = np.empty((0,512,512,3), dtype=np.int8)

### Read annotation file, fetch image, normalise image and array, compose data and target arrays

for root, folders, files in os.walk(IMAGES_FOLDER):
    for file in files:
        image_path = os.path.join(IMAGES_FOLDER, file)
        image = cv2.imread(image_path)
        image = np.expand_dims(image, axis=0)

        if image is not None:
            data = np.vstack((data, image))


### Shuffle data and target synchronously

num_samples = data.shape[0]
arr = np.arange(num_samples)
np.random.shuffle(arr)
print("num_samples", num_samples)
data = data[arr]


print(data.shape)

np.save(os.path.join(OUTPUT,'test.npy'), data)
