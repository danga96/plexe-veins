import os
import sys
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from keras import optimizers
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers import Dropout
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import GRU
from keras.layers.convolutional import Conv1D
from keras.layers.convolutional import MaxPooling1D
from keras.layers.embeddings import Embedding
from keras.preprocessing import sequence
from keras.constraints import maxnorm
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
import tensorflow as tf    

# fix random seed for reproducibility
tf.random.set_seed(2)
np.random.seed(7)
#os.system("taskset -p -c 0 %d" % os.getpid())

csv_path = "../../RollingDB/Rdistance.csv"
attacks = pd.read_csv(csv_path)

X = attacks.drop(['Detection'], axis=1).values

Y = attacks["Detection"].values



scaler = StandardScaler()
X_one_column = X.reshape([-1,1])
result_one_column = scaler.fit_transform(X_one_column)
X = (X - scaler.mean_)/(scaler.var_**0.5)


X = X.reshape(X.shape[0],10,1)

# create model
model = Sequential()

model.add(GRU(64, input_shape=(10,1), activation='relu', kernel_initializer='uniform'))
model.add(Dropout(0.2))
model.add(Dense(1, activation='sigmoid', kernel_initializer='uniform'))
# Compile model
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
# Fit the model
history = model.fit(X, Y, validation_split=0.33, epochs=100, batch_size=32, verbose=1)
# list all data in history
print(history.history.keys())
# summarize history for accuracy
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
# summarize history for loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()