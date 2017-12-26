# -*- coding: utf-8 -*-
"""
Created on Thu Dec 21 15:24:12 2017

@author: Richa
"""

import numpy as np
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Dropout, Flatten, Dense
from keras import applications
from keras.regularizers import l2



# dimensions of our images.

img_width, img_height = 150, 150



top_model_weights_path = 'bottleneck_fc_model_uncorrupted.py'
train_data_dir = 'food_dataset/training_set'
validation_data_dir = 'food_dataset/test_set'
nb_train_samples = 2400
nb_validation_samples = 600
epochs = 50
batch_size = 16

def save_bottlebeck_features():
    datagen = ImageDataGenerator(rescale=1. / 255)

    # build the VGG16 network
    model = applications.VGG16(include_top=False, weights='imagenet')

    generator = datagen.flow_from_directory(
        train_data_dir,
        target_size=(img_width, img_height),
        batch_size=batch_size,
        class_mode=None,
        shuffle=False)
    
    bottleneck_features_train = model.predict_generator(
        generator, nb_train_samples // batch_size)
    
   # np.save(bottleneck_features_train)
    #np.load("bottleneck_features_train.npy")
    
    np.save(open('bottleneck_features_train', 'wb'), bottleneck_features_train)

    generator = datagen.flow_from_directory(
        validation_data_dir,
        target_size=(img_width, img_height),
        batch_size=batch_size,
        class_mode=None,
        shuffle=False)

    bottleneck_features_validation = model.predict_generator(
        generator, nb_validation_samples // batch_size)


    np.save(open('bottleneck_features_validation', 'wb'), bottleneck_features_validation)
    #np.save(bottleneck_features_validation)
    #np.load('bottleneck_features_validation.npy', 'w')
     #np.save(open('bottleneck_features_train', 'wb'), bottleneck_features_train)       

def train_top_model():

    train_data = np.load(open('bottleneck_features_train','rb'))
    train_labels = np.array(
        [0] * (nb_train_samples // 2) + [1] * (nb_train_samples // 2))
    
    validation_data = np.load(open('bottleneck_features_validation','rb'))
    validation_labels = np.array(
        [0] * (nb_validation_samples // 2) + [1] * (nb_validation_samples // 2))


    model = Sequential()
    model.add(Flatten(input_shape=train_data.shape[1:]))
    model.add(Dense(256, activation='relu',W_regularizer=l2(0.01)))
    model.add(Dropout(0.5))
    model.add(Dense(1, activation='softmax'))


    model.compile(optimizer='sgd',
                  loss='categorical_crossentropy', metrics=['accuracy'])



    model.fit(train_data, train_labels,
              epochs=epochs,
              batch_size=batch_size,
              validation_data=(validation_data, validation_labels))
    model.save_weights(top_model_weights_path)

save_bottlebeck_features()
train_top_model()