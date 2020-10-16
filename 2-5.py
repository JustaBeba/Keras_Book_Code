# -*- coding: utf-8 -*-
"""
Created on Mon Oct 12 19:24:08 2020

@author: kt510085
"""
from keras import models
from keras import layers

from keras.datasets import mnist

from keras.utils import to_categorical
#Nero Network
network = models.Sequential()
network.add(layers.Dense(512,activation='relu',input_shape=(28 * 28,)))
network.add(layers.Dense(10,activation='softmax'))

network.compile(optimizer='rmsprop',loss='categorical_crossentropy',metrics=['accuracy'])

#dataset_mnist
(train_images, train_labels),(test_images, test_labels)=mnist.load_data()
#image array
train_images = train_images.reshape((60000,28 * 28))
train_images = train_images.astype('float32')/255

test_images = test_images.reshape((10000,28 * 28))
test_images = test_images.astype('float32')/255
#label asset
train_labels = to_categorical(train_labels)
test_labels = to_categorical(test_labels)
