# Generates run original

import csv
import numpy as np
import cv2
import tensorflow as tf
from PIL import Image
import random
import matplotlib.pyplot as plt
#%matplotlib inline


lines = []
with open ('new_data1/driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    for line in reader:
        lines.append(line)
        
images = []
measurements = []
correction = 0.5


for line in lines:
    
    #center images load
    source_path_c =  line[0]
    filename_c = source_path_c.split('/')[-1]
    current_path_c = 'new_data1/IMG/' + filename_c
    image_c = cv2.imread(current_path_c)
    images.append(image_c)
    measurement_c = float(line[3])
    measurements.append(measurement_c)
    
    #left images load
    source_path_l =  line[1]
    filename_l = source_path_l.split('/')[-1]
    current_path_l = 'new_data1/IMG/' + filename_l
    image_l = cv2.imread(current_path_l)
    images.append(image_l)
    measurement_l = measurement_c + correction
    measurements.append(measurement_l)
    
    #right images load
    source_path_r =  line[2]
    filename_r = source_path_r.split('/')[-1]
    current_path_r = 'new_data1/IMG/' + filename_r
    image_r = cv2.imread(current_path_r)
    images.append(image_r)
    measurement_r = measurement_c - correction
    measurements.append(measurement_r)

print('Total steering angle positions original' , len(measurements))
print('Total number of original images ', len(images))

# display the images
r = 9
plt.figure(figsize=(15, 30))

for i in range(10):
    image = images[i]
    plt.subplot(r,5,i+1)
    plt.axis('off')
    plt.imshow(image)
    plt.text(10, 20,'Angle {0:.2f}'.format(measurements[i]),backgroundcolor = [1,1,1])
    plt.savefig("Display_Images_original.png")
    
augmented_images, augmented_measurements = [], []

for image,measurement in zip(images,measurements):
    augmented_images.append(image)
    augmented_measurements.append(measurement)
    augmented_images.append(cv2.flip(image,1))
    augmented_measurements.append(measurement*-1)
    
X_train = np.array(augmented_images)
y_train = np.array(augmented_measurements)

print('Data Augmentation done')
print('Total train images and validation images ', X_train.shape)
print('Total train angles and validation angles ', y_train.shape)

# Display few augmented images
#no. of rows and figure size
r = 9
plt.figure(figsize=(15, 30))

for i in range(10):
    image = X_train[i]
    plt.subplot(r,5,i+1)
    plt.axis('off')
    plt.imshow(image)
    plt.text(10, 20,'Angle {0:.2f}'.format(y_train[i]),backgroundcolor = [1,1,1])
    plt.savefig("Display_Images_augmented.png")