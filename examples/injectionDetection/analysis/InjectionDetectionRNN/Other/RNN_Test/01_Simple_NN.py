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

csv_path = "/home/tesi/src/plexe-veins/examples/injectionDetection/analysis/Other/Rolling/KFdistance.csv"
attacks = pd.read_csv(csv_path)
#X = attacks.drop(attacks.columns[[3,4,5,6,-1]], axis=1).values
#KF distance,V2X-KF distance,V2X-KF speed,Radar distance,Radar-KF distance,Radar-V2X speed,Radar-KF speed,Detection
X = attacks.drop(['Detection'], axis=1).values
#X = attacks.drop(attacks.columns[[1,2,5,6,-1]], axis=1).values

Y = attacks["Detection"].values


X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size = 0.3, shuffle=True)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

"""
np.set_printoptions(threshold=np.inf)
print(X_train,"\n\n")
X_train = sequence.pad_sequences(X_train, maxlen=max_review_length)
print(X_train)
exit()
X_test = sequence.pad_sequences(X_test, maxlen=max_review_length)
"""
# create the model
#np.set_printoptions(threshold=np.inf)

#y_train = np_utils.to_categorical(y_train)
#y_test = np_utils.to_categorical(y_test)

model = Sequential()
model.add(Dense(512, input_dim = X_train.shape[1], activation='relu', kernel_initializer='uniform'))
model.add(Dropout(0.1))

#model.add(Dense(256, activation='relu', kernel_initializer='uniform'))
#model.add(Dropout(0.1))

model.add(Dense(128, activation='relu', kernel_initializer='uniform'))
model.add(Dropout(0.1))

model.add(Dense(64, activation='relu', kernel_initializer='uniform'))
model.add(Dropout(0.1))
"""
model.add(Dense(40, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(30, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(20, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(10, activation='relu'))
"""
model.add(Dense(1, activation='sigmoid', kernel_initializer='uniform'))
_optimizer = optimizers.Adam(learning_rate=0.001)
model.compile(loss='binary_crossentropy', optimizer=_optimizer, metrics=['accuracy'])
print(model.summary())
history = model.fit(X_train, y_train, epochs=2, batch_size=32)
# Final evaluation of the model
scores = model.evaluate(X_test, y_test, verbose=1)
print("Accuracy: %.2f%%" % (scores[1]*100))

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

Y_pred = model.predict(X_test)
Y_pred_round = [1 * (x[0]>=0.5) for x in Y_pred]
print(len(Y_pred_round))
print(len(y_test))
from sklearn.metrics import confusion_matrix
confusion_matrix = confusion_matrix(y_test, Y_pred_round)
print(confusion_matrix)
