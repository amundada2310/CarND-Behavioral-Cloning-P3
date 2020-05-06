
# Generates run original

import csv
import numpy as np
import cv2
import tensorflow as tf
from PIL import Image


lines = []
with open ('newdata_try/driving_log.csv') as csvfile:
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
    current_path_c = 'newdata_try/IMG/' + filename_c
    image_c = cv2.imread(current_path_c)
    images.append(image_c)
    measurement_c = float(line[3])
    measurements.append(measurement_c)
    
    #left images load
    source_path_l =  line[1]
    filename_l = source_path_l.split('/')[-1]
    current_path_l = 'newdata_try/IMG/' + filename_l
    image_l = cv2.imread(current_path_l)
    images.append(image_l)
    measurement_l = measurement_c + correction
    measurements.append(measurement_l)
    
    #right images load
    source_path_r =  line[2]
    filename_r = source_path_r.split('/')[-1]
    current_path_r = 'newdata_try/IMG/' + filename_r
    image_r = cv2.imread(current_path_r)
    images.append(image_r)
    measurement_r = measurement_c - correction
    measurements.append(measurement_r)

print('Total steering angle positions original' , len(measurements))
print('Total number of original images ', len(images))

    
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

#import random
#import matplotlib.pyplot as plt
#%matplotlib inline

#no. of rows and figure size
#r = 9
#plt.figure(figsize=(15, 30))

#for i in range(10):
    # randomly select the samples from X_train data set
    #index = random.randint(0, len(X_train))
    #image = X_train[index]
    
    #plt.subplot(r,5,i+1)
    #plt.axis('off')
    #plt.imshow(image)
    #plt.savefig("Display_Images_try.png")
    
from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda, Activation, Cropping2D
from keras.layers.convolutional import Conv2D
from keras.layers.pooling import MaxPooling2D
from keras.layers import Dropout

model = Sequential()
model.add(Lambda(lambda x: (x / 255.0) - 0.5, input_shape = (160,320,3)))
model.add(Cropping2D(cropping=((70,25), (0,0))))

model.add(Conv2D(24, (5,5), subsample = (2,2), activation = 'relu'))
model.add(Conv2D(36, (5,5), subsample = (2,2), activation = 'relu'))
model.add(Conv2D(48, (5,5), subsample = (2,2), activation = 'relu'))
model.add(Conv2D(64, (3,3), activation = 'relu'))
model.add(Conv2D(64, (3,3), activation = 'relu'))

model.add(Dropout(0.6))
model.add(Flatten())
model.add(Dense(1164,activation = 'relu'))

model.add(Dense(100,activation = 'relu'))
model.add(Dense(50,activation = 'relu'))
model.add(Dense(10,activation = 'relu'))
model.add(Dense(1))

model.summary()      

model.compile(loss='mse',optimizer='adam')
history_object = model.fit(X_train,y_train,validation_split=0.2,shuffle=True, nb_epoch=3, verbose=1)
model.save('model_try.h5')

from keras.models import Model
import matplotlib.pyplot as plt

#history_object = model.fit_generator(train_generator, samples_per_epoch =len(train_samples), validation_data = validation_generator,nb_val_samples len(validation_samples), nb_epoch=5, verbose=1)

### print the keys contained in the history object
print(history_object.history.keys())

### plot the training and validation loss for each epoch
plt.plot(history_object.history['loss'])
plt.plot(history_object.history['val_loss'])
plt.title('model mean squared error loss')
plt.ylabel('mean squared error loss')
plt.xlabel('epoch')
plt.legend(['training set', 'validation set'], loc='upper right')
plt.savefig("Model Mean Square Error Loss_try.png")
#plt.show()

exit()
