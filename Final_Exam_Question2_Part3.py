## Patterned after dr. Gates https://gatesboltonanalytics.com/
## ###########################################################
import tensorflow as tf
from tensorflow import keras
from keras import layers
from sklearn.model_selection import train_test_split
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout, LSTM
from keras.optimizers import Adam
import matplotlib.pyplot as plt
from keras import layers
import numpy as np
import pandas as pd
from keras import utils
import seaborn as sns
from sklearn.metrics import confusion_matrix

## Getting the data loaded and ready


## Dataset
filename="Final_News_DF_Labeled_ExamDataset.csv"

## !! Update this to YOUR path
DF = pd.read_csv("C:/NeuralNetworks/"+str(filename))
print(DF)

#raw data
print("Raw data \n")
print(type(DF))
print(DF.shape)

## Set labels to numerical values
classes = ["politics", "football", "science"]
DF['LABEL'].replace(classes, [0, 1, 2], inplace=True)
print(DF)

#Split into test and train datasets
train, test = train_test_split(DF, test_size=0.2)
print("Train data \n")
print(type(train))
print(train.shape)

print("Test data \n")
print(type(test))
print(test.shape)


## Set y to the label for both test and train datasets
y_test = np.array(test.iloc[:,0]).T
y_test = np.array([y_test]).T
print("y_test is\n", y_test)

y_train = np.array(train.iloc[:,0]).T
y_train = np.array([y_train]).T
print("y_train is\n", y_train)


## Drop the label and assign to x for both test and train datasets
test=test.drop('LABEL', axis=1)
print(test)
print(type(test))
print(test.shape)
x_test = np.array(test)
print("x_test is\n", x_test)
print(type(x_test))
print(x_test.shape)


train=train.drop('LABEL', axis=1)
print(train)
print(type(train))
print(train.shape)
x_train = np.array(train)
print("x_train is\n", x_train)
print(type(x_train))
print(x_train.shape)



NumCols=x_train.shape[1]
print(NumCols)
input_dim = NumCols

NumRows=x_train.shape[0]
print(NumRows)


###############################################
##
## ANN
##
######################################################
ANN_Model = tf.keras.models.Sequential([
  tf.keras.layers.Dense(64, activation="relu", input_shape=(input_dim,)),
  tf.keras.layers.Dropout(0.2), 
  tf.keras.layers.Dense(16, activation='relu'), 
  tf.keras.layers.Dropout(0.2), 
  tf.keras.layers.Dense(3, activation='softmax')
])
    
ANN_Model.summary()

ANN_Model.compile(
                 loss="sparse_categorical_crossentropy",
                 metrics=["accuracy"],
                 optimizer='adam'
                 )

Hist=ANN_Model.fit(x_train, y_train, batch_size=12, epochs=10, validation_data=(x_test, y_test))



###### History and Accuracy
plt.figure(figsize = (8,8))
plt.plot(Hist.history['accuracy'], label='accuracy')
plt.plot(Hist.history['val_accuracy'], label = 'val_accuracy')
plt.plot(Hist.history['loss'], label = 'loss')
plt.plot(Hist.history['val_loss'], label = 'val_loss')
plt.xlabel('Epoch')
plt.ylabel('Accuracy/Loss')
plt.ylim([0.0, 1.2])
plt.legend(loc='lower right')



##Test and Model
Test_Loss, Test_Accuracy = ANN_Model.evaluate(x_test, y_test)

## Predictions
predictions=ANN_Model.predict([x_test])

## Confusion Matrix and Accuracy - and Visual Options
print("The test accuracy is \n", Test_Accuracy)


print("The prediction accuracy via confusion matrix is:\n")
#print(y_test)
#print(predictions)
#print(predictions.shape)
Max_Values = np.squeeze(np.array(predictions.argmax(axis=1)))
#print(Max_Values)
#print(np.argmax([predictions]))
cm = confusion_matrix(Max_Values, y_test)
print(cm)

## Pretty Confusion Matrix
labels = [0, 1, 2]
fig, ax = plt.subplots(figsize=(13,13)) 
sns.heatmap(cm, annot=True, fmt='g', ax=ax, annot_kws={'size': 18})
ax.set_xlabel('True labels') 
ax.set_ylabel('Predicted labels')
ax.set_title('Confusion Matrix: ANN') 
ax.xaxis.set_ticklabels(["politics", "football", "science"],rotation=90, fontsize = 18)
ax.yaxis.set_ticklabels(["politics", "football", "science"],rotation=0, fontsize = 18)




###############################################
##
## CNN
##
######################################################

CNN_Model = tf.keras.models.Sequential([
  tf.keras.layers.Embedding(input_dim=NumRows, output_dim=32, input_length=NumCols),

  tf.keras.layers.Conv1D(kernel_size=3, activation="relu", filters=16), 
  tf.keras.layers.MaxPool1D(pool_size=2), 

  tf.keras.layers.Flatten(),
  tf.keras.layers.Dense(64, activation='relu'), 
  tf.keras.layers.Dropout(0.5),

  tf.keras.layers.Dense(3, activation='softmax') 
])


CNN_Model.summary()


CNN_Model.compile(optimizer='adam',
              loss="sparse_categorical_crossentropy",
              metrics=['accuracy'])


Hist = CNN_Model.fit(x_train, y_train, batch_size=12, epochs=10, validation_data=(x_test, y_test))


###### History and Accuracy
plt.figure(figsize = (8,8))
plt.plot(Hist.history['accuracy'], label='accuracy')
plt.plot(Hist.history['val_accuracy'], label = 'val_accuracy')
plt.plot(Hist.history['loss'], label = 'loss')
plt.plot(Hist.history['val_loss'], label = 'val_loss')
plt.xlabel('Epoch')
plt.ylabel('Accuracy/Loss')
plt.ylim([0.0, 1.2])
plt.legend(loc='lower right')



##Test and Model
Test_Loss, Test_Accuracy = CNN_Model.evaluate(x_test, y_test)

## Predictions
predictions=CNN_Model.predict([x_test])

## Confusion Matrix and Accuracy - and Visual Options
print("The test accuracy is \n", Test_Accuracy)


print("The prediction accuracy via confusion matrix is:\n")
#print(y_test)
#print(predictions)
#print(predictions.shape)
Max_Values = np.squeeze(np.array(predictions.argmax(axis=1)))
#print(Max_Values)
#print(np.argmax([predictions]))
cm = confusion_matrix(Max_Values, y_test)
print(cm)

## Pretty Confusion Matrix
labels = [0, 1, 2]
fig, ax = plt.subplots(figsize=(13,13)) 
sns.heatmap(cm, annot=True, fmt='g', ax=ax, annot_kws={'size': 18})
ax.set_xlabel('True labels') 
ax.set_ylabel('Predicted labels')
ax.set_title('Confusion Matrix: CNN') 
ax.xaxis.set_ticklabels(["politics", "football", "science"],rotation=90, fontsize = 18)
ax.yaxis.set_ticklabels(["politics", "football", "science"],rotation=0, fontsize = 18)




############################################
## LSTM
#############################################
LSTM_Model = tf.keras.models.Sequential([
  tf.keras.layers.Embedding(input_dim=NumRows, output_dim=32, input_length=NumCols),
  tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(25)),
  tf.keras.layers.Dense(3, activation='softmax')
])
    
LSTM_Model.summary()

LSTM_Model.compile(
                 loss="sparse_categorical_crossentropy",
                 metrics=["accuracy"],
                 optimizer='adam'
                 )


Hist=LSTM_Model.fit(x_train, y_train, batch_size=12, epochs=25, validation_data=(x_test, y_test))



###### History and Accuracy
plt.figure(figsize = (8,8))
plt.plot(Hist.history['accuracy'], label='accuracy')
plt.plot(Hist.history['val_accuracy'], label = 'val_accuracy')
plt.plot(Hist.history['loss'], label = 'loss')
plt.plot(Hist.history['val_loss'], label = 'val_loss')
plt.xlabel('Epoch')
plt.ylabel('Accuracy/Loss')
plt.ylim([0.0, 1.2])
plt.legend(loc='lower right')



##Test and Model
Test_Loss, Test_Accuracy = LSTM_Model.evaluate(x_test, y_test)

## Predictions
predictions=LSTM_Model.predict([x_test])

## Confusion Matrix and Accuracy - and Visual Options
print("The test accuracy is \n", Test_Accuracy)


print("The prediction accuracy via confusion matrix is:\n")
#print(y_test)
#print(predictions)
#print(predictions.shape)
Max_Values = np.squeeze(np.array(predictions.argmax(axis=1)))
#print(Max_Values)
#print(np.argmax([predictions]))
cm = confusion_matrix(Max_Values, y_test)
print(cm)

## Pretty Confusion Matrix
labels = [0, 1, 2]
fig, ax = plt.subplots(figsize=(13,13)) 
sns.heatmap(cm, annot=True, fmt='g', ax=ax, annot_kws={'size': 18})
ax.set_xlabel('True labels') 
ax.set_ylabel('Predicted labels')
ax.set_title('Confusion Matrix: LSTM') 
ax.xaxis.set_ticklabels(["politics", "football", "science"],rotation=90, fontsize = 18)
ax.yaxis.set_ticklabels(["politics", "football", "science"],rotation=0, fontsize = 18)



print("Done!")