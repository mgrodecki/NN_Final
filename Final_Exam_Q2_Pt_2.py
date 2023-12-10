## TF - Keras CNN Model 
## Two Conv2D layers, two MaxPooling layers, one Flatten Layer and one Dense Layer
## Patterned after dr. Gates https://gatesboltonanalytics.com/
## ###############################################################################


import tensorflow as tf
from tensorflow import keras
from keras import layers
from sklearn.model_selection import train_test_split
#import tensorflow.keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout, LSTM
from keras.optimizers import Adam
import matplotlib.pyplot as plt
from keras import layers
import numpy as np
import pandas as pd


##########################################
## Using Keras
#################################################

##......................................................
#########################################################
## Build the CNN Model 
#########################################################


CNN_Model = tf.keras.models.Sequential([

  tf.keras.layers.Conv2D(input_shape=(30,30,1), kernel_size=(3,3), strides = (1,1), padding = "same", filters=2, activation="relu"), 
  tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),


  tf.keras.layers.Conv2D(kernel_size=(3,3), strides = (1,1), padding = "same", filters=4, activation="relu"), 
  tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
  
  
  tf.keras.layers.Flatten(),
  tf.keras.layers.Dense(196, activation='relu'), 
  tf.keras.layers.Dense(3, activation='softmax') 
])

    
CNN_Model.summary()


## -------
## Step 2 - 
## Compile the Model
## -------


CNN_Model.compile(
                 loss="categorical_crossentropy",
                 metrics=["accuracy"],
                 optimizer='adam'
                 
                 )


