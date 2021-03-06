# -*- coding: utf-8 -*-
"""cifar10_acc_challenge.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1aA8JlpbiIWS3DeSTb18sFFjrT8CcW6dr
"""

from keras.datasets import cifar10
import keras
import keras.utils
(trainX, trainY), (testX, testY) = cifar10.load_data()

trainY = keras.utils.to_categorical(trainY)
testY = keras.utils.to_categorical(testY)
def normalize(X):
    return X/255
trainX = normalize(trainX)
testX = normalize(testX)

from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.layers.core import Dense, Flatten, Activation, Dropout

model = keras.Sequential()
model.add(Convolution2D(32, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same', input_shape=(32, 32, 3)))
model.add(Convolution2D(32, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
model.add(MaxPooling2D((2, 2)))
model.add(Dropout(0.3))
model.add(Convolution2D(64, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
model.add(Convolution2D(64, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
model.add(MaxPooling2D((2, 2)))
model.add(Dropout(0.3))
model.add(Convolution2D(128, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
model.add(Convolution2D(128, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
model.add(MaxPooling2D((2, 2)))
model.add(Dropout(0.3))
model.add(Flatten())
model.add(Dense(100))
model.add(Dropout(0.5))
model.add(Dense(10))
model.add(Activation ('softmax'))

OPT = keras.optimizers.SGD(learning_rate = 0.0001, momentum = 0.9)
model.compile(loss = 'categorical_crossentropy', optimizer = OPT, metrics = ['accuracy'])

epoch = model.fit(trainX, trainY, epochs = 300, validation_data = (testX,testY))
model.evaluate(testX,testY)

import matplotlib.pyplot as plt

plt.plot(epoch.history['val_accuracy'])
plt.plot(epoch.history['accuracy'])
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend(['Test','Train'])
plt.show



import pandas as pd
Accuracies = pd.DataFrame()
Accuracies['Validation'] = epoch.history['val_accuracy']
Accuracies['Training'] = epoch.history['accuracy']
Accuracies.to_excel('CIFAR.xlsx')