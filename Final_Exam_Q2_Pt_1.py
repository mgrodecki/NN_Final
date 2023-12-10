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
## Build the ANN Model 
#########################################################


ANN_Model = tf.keras.models.Sequential([
  tf.keras.layers.Dense(4, activation="sigmoid", input_shape=(4,)),
  tf.keras.layers.Dense(3, activation='relu'), 
  tf.keras.layers.Dense(3, activation='softmax')
])
    
ANN_Model.summary()


## -------
## Step 2 - 
## Compile the Model
## -------


ANN_Model.compile(
                 loss="categorical_crossentropy",
                 metrics=["accuracy"],
                 optimizer='adam'
                 
                 )


