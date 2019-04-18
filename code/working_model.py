#!/usr/bin/env python
# coding: utf-8

# ## Import Libraries and Functions
import numpy as np
import time
import tensorflow as tf
import keras
from keras.models import Sequential
from keras.layers import Conv2D, Activation, Dense, MaxPooling2D, Flatten, Dropout
from sklearn.metrics import log_loss
from keras.optimizers import SGD
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import classification_report,confusion_matrix, accuracy_score
import matplotlib.pyplot as plt
from sklearn.metrics import average_precision_score
from load_data import load_resized_data


# ## Constants
img_rows=100 #dimensions of image
img_cols=100
channel = 3 #RGB
num_classes = 2 
batch_size = 128
num_epoch = 4


# ## Convolutional Model

model = Sequential()
model.add(Conv2D(32, (3, 3), input_shape=(100,100,3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(32, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(64, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Flatten())
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(num_classes, activation='softmax'))
model.summary()


# ## Compile Model

model.compile(optimizer="adam",
              loss='categorical_crossentropy',
              metrics=['accuracy'])


# ## Load Data

# ### Malaria Cell Image Dataset Reference: https://ceb.nlm.nih.gov/repositories/malaria-datasets/
X_train, X_valid, Y_train, Y_valid = load_resized_data(img_rows, img_cols)
print("X_train.shape, Y_train.shape")
print(X_train.shape, Y_train.shape)

# ## Training

t=time.time()

hist = model.fit(X_train, Y_train, batch_size=batch_size, epochs=num_epoch, verbose=1, 
                     shuffle=True, validation_data= (X_valid,Y_valid))

print('Training time: %s' % (time.time()-t))


# ## If training did not work as expected I suggest loading in the working weights from below. You should hit 94%-95% validation accuracy

# ## Loading weights
#model = load_model('malaria_model.h5')


# ## Predictions
y_pred = model.predict(X_valid, batch_size=batch_size, verbose=1)
y_pred = np.argmax(y_pred, axis=1)
y_actual = np.argmax(Y_valid, axis=1)
correct = y_actual[y_actual == y_pred]
incorrect = y_actual[y_actual != y_pred]

print("Test Accuracy = ", len(correct)/len(y_actual), "%")
print("Test Inaccuracy = ", len(incorrect)/len(y_actual),"%")


y_pred = model.predict(X_train, batch_size=batch_size, verbose=1)
y_pred = np.argmax(y_pred, axis=1)
y_actual = np.argmax(Y_train, axis=1)

correct = y_actual[y_actual == y_pred]
incorrect = y_actual[y_actual != y_pred]

print("Test Accuracy = ", len(correct)/len(y_actual), "%")
print("Test Inaccuracy = ", len(incorrect)/len(y_actual),"%")


