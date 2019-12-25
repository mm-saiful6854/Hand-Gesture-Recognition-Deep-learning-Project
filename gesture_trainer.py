# -*- coding: utf-8 -*-
"""
Created on Thu Dec  5 15:34:35 2019

@author: cse.repon
"""

import sys # system functions (ie. exiting the program)
import os # operating system functions (ie. path building on Windows vs. MacOs)
import time # for time operations
import uuid # for generating unique file names
import math # math functions

from IPython.display import display as ipydisplay, Image, clear_output, HTML # for interacting with the notebook better

import numpy as np # matrix operations (ie. difference between two matricies)
import cv2 # (OpenCV) computer vision functions (ie. tracking)
(major_ver, minor_ver, subminor_ver) = (cv2.__version__).split('.')
print('OpenCV Version: {}.{}.{}'.format(major_ver, minor_ver, subminor_ver))

import matplotlib.pyplot as plt # (optional) for plotting and showing images inline

import keras # high level api to tensorflow (or theano, CNTK, etc.) and useful image preprocessing
from keras import backend as K
from keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array
from keras.models import Sequential, load_model, model_from_json
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D
print('Keras image data format: {}'.format(K.image_data_format()))

IMAGES_FOLDER = os.path.join('images') # images for visuals

MODEL_PATH = os.path.join('model')
# MODEL_FILE = os.path.join(MODEL_PATH, 'handrecognition_model.hdf5') # path to model weights and architechture file
MODEL_FILE = os.path.join(MODEL_PATH, 'hand_5_gesture_22dec1.hdf5') # path to model weights and architechture file
MODEL_HISTORY = os.path.join(MODEL_PATH, 'model_history.txt') # path to model training history


classes = {
    0: 'five',
    1: 'grab',
    2: 'next',
    3: 'previous',
    4: 'right'
}




def train_model():
    model = Sequential()
    
    model.add(Conv2D(32, (3, 3), input_shape=(54, 54, 1)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    
    model.add(Conv2D(32, (3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    
    model.add(Conv2D(64, (3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    
    model.add(Flatten())
    model.add(Dense(64))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Dense(5))
    model.add(Activation('softmax'))
    
    model.compile(loss='categorical_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])
    model.summary()
    
    ########################################################
    
    
    batch_size = 16
    
    training_datagen = ImageDataGenerator(
        rotation_range=50,
        width_shift_range=0.1,
        height_shift_range=0.1,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest'
    )
    
    validation_datagen = ImageDataGenerator(zoom_range=0.2, rotation_range=10)
    
    training_generator = training_datagen.flow_from_directory(
        'training_data',
        target_size=(54, 54),
        batch_size=batch_size,
        color_mode='grayscale'
    )
    
    validation_generator = validation_datagen.flow_from_directory(
        'validation_data',
        target_size=(54, 54),
        batch_size=batch_size,
        color_mode='grayscale'
    )
    
    ########################################################
    model.fit_generator(
        generator=training_generator,
        steps_per_epoch=2000 // batch_size,
        epochs=50,
        verbose=1,
        validation_data=validation_generator,
        validation_steps=200 // batch_size,
        workers=32,
    )
    
    model.save(MODEL_FILE)
    return

train_model() 
