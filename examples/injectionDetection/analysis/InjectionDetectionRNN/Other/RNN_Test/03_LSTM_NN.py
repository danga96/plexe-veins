import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers import Dropout
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers import LSTM
from keras.layers import GRU
from keras.layers import Bidirectional
from keras.layers import SimpleRNN
from keras.layers import TimeDistributed
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


csv_path = "../../RollingDB/Rdistance.csv"
attacks = pd.read_csv(csv_path)
#X = attacks.drop(attacks.columns[[3,4,5,6,-1]], axis=1).values
X = attacks.drop(['Detection'], axis=1).values
Y = attacks["Detection"].values


X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size = 0.3, shuffle=True)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# create the model
#np.set_printoptions(threshold=np.inf)

#y_train = np_utils.to_categorical(y_train)
#y_test = np_utils.to_categorical(y_test)

model = Sequential()
X_test = X_test.reshape((X_test.shape[0],X_test.shape[1],1))
#y_test = y_test.reshape((y_test.shape[0],1))

X_train = X_train.reshape((X_train.shape[0],X_train.shape[1],1))
#y_train = y_train.reshape((y_train.shape[0],1))
print(y_train[0])
print("shape_X",X_train.shape,"shape_Y",y_train.shape)
#exit()
#####################################################################
n_timesteps, n_features, n_outputs = X_train.shape[1], X_train.shape[2], y_train.shape[0]
model.add(GRU(128, input_shape=(X_train.shape[1:]), activation='relu'))
#model.add(Bidirectional(LSTM(128, return_sequences=True), input_shape=(X_train.shape[1:])))
#model.add(Bidirectional(GRU(64, return_sequences=True)))
#model.add(LSTM(128, input_shape=(X_train.shape[1:]), activation='relu', return_sequences=True))
model.add(Dropout(0.1))
"""
model.add(LSTM(256, activation='relu',return_sequences=True))
model.add(Dropout(0.1))

model.add(LSTM(128, activation='relu',return_sequences=True))
model.add(Dropout(0.1))
"""
"""
model.add(GRU(128, activation='relu', return_sequences=True))
model.add(Dropout(0.8))

model.add(LSTM(64, activation='relu', return_sequences=True))
model.add(Dropout(0.8))
"""
#model.add(GRU(64, activation='relu'))
#model.add(Dropout(0.1))
#model.add(TimeDistributed(Dense(1)))
#model.add(TimeDistributed(Dense(1, activation='sigmoid')))

model.add(Dense(512, activation='relu', kernel_initializer='uniform'))
model.add(Dropout(0.1))

model.add(Dense(256, activation='relu', kernel_initializer='uniform'))
model.add(Dropout(0.1))

model.add(Dense(128, activation='relu', kernel_initializer='uniform'))
model.add(Dropout(0.1))

model.add(Dense(64, activation='relu', kernel_initializer='uniform'))
model.add(Dropout(0.1))

#####################################################################
"""
model.add(Dense(256, input_dim = X_train.shape[1], activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.5))
"""

model.add(Dense(1, activation='sigmoid', kernel_initializer='uniform'))

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
print(model.summary())
history = model.fit(X_train, y_train, epochs=1, batch_size=32)
# Final evaluation of the model
scores = model.evaluate(X_test, y_test, verbose=1)
print("Accuracy: %.2f%%" % (scores[1]*100))
"""
# summarize history for accuracy
plt.plot(history.history['accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
# summarize history for loss
plt.plot(history.history['loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
"""
Y_pred = model.predict(X_test)
Y_pred_round = [1 * (x[0]>=0.5) for x in Y_pred]
print(len(Y_pred_round))
print(len(y_test))
from sklearn.metrics import confusion_matrix
confusion_matrix = confusion_matrix(y_test, Y_pred_round)
print(confusion_matrix)
