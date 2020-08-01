# -*- coding: utf-8 -*-
# Use scikit-learn to grid search the batch size and epochs
import numpy as np
import pandas as pd
import time
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.wrappers.scikit_learn import KerasClassifier
from keras import optimizers
#from keras.optimizers import SGD
#from keras.optimizers import Adam
# Function to create model, required for KerasClassifier
start_time = time.time()
def _get_optimizer(_optimizer, learn_rate, momentum):
    #optimizer = [ 'SGD', 'RMSprop', 'Adagrad', 'Adadelta', 'Adam', 'Adamax', 'Nadam']
    switcher = {
        'SGD': optimizers.SGD(learning_rate=learn_rate,momentum=momentum),
        'RMSprop': optimizers.RMSprop(learning_rate=learn_rate,momentum=momentum),
        'Adagrad': optimizers.Adagrad(learning_rate=learn_rate),
        'Adadelta': optimizers.Adadelta(learning_rate=learn_rate),
        'Adam': optimizers.Adam(learning_rate=learn_rate),
        'Adamax': optimizers.Adamax(learning_rate=learn_rate),
        'Nadam': optimizers.Nadam(learning_rate=learn_rate)
    }
    return switcher.get(_optimizer, 0)
def create_model(dense_layer=0, dropout_rate=0.0, activation = 'relu', optimizer='Adam', momentum = 0.0, learn_rate=0.001, init_mode='uniform'):
    # create model
    model = Sequential()
    """
    if dense_layer==0:
        first_layer_size=16 if dense_layer==0 else dense_layer[0]
    else:
        first_layer_size = dense_layer[0]
    """
    first_layer_size= 16 if dense_layer==0 else dense_layer[0]

    model.add(Dense(first_layer_size, input_dim=7, kernel_initializer=init_mode, activation=activation))
    model.add(Dropout(dropout_rate))
    for layer_size in dense_layer[1:]:
        model.add(Dense(layer_size, kernel_initializer=init_mode, activation= activation))
        model.add(Dropout(dropout_rate))
    #  model.add(Dense(neurons, input_dim=7, activation='relu'))
    #  model.add(Dropout(dropout_rate))
    model.add(Dense(1, kernel_initializer=init_mode, activation='sigmoid'))
    optimizer = _get_optimizer(optimizer, learn_rate, momentum)
    # Compile model
    model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])
    return model
# fix random seed for reproducibility
np.random.seed(7)
# load dataset
csv_path = "/home/tesi/src/plexe-veins/examples/injectionDetection/analysis/Other/DB.csv"
attacks = pd.read_csv(csv_path)
# split into input (X) and output (Y) variables
X = attacks.drop(['Detection'], axis=1).values
Y = attacks["Detection"].values
#normalize data
scaler = StandardScaler()
X = scaler.fit_transform(X)
# create model
model = KerasClassifier(build_fn=create_model, verbose=0)
# define the grid search parameters
dense_size_candidates = [(512,256,128,64,), (256, 128, 64,), (128,128,64,64,32,32,16,8,), (256,128,128,64), (64,32,16,8,4,2,)]
########################################################
# Use scikit-learn to grid search 
activation =  ['relu', 'tanh', 'sigmoid', 'hard_sigmoid', 'linear', 'softmax', 'softplus', 'softsign']
momentum = [0.0, 0.2, 0.4, 0.6, 0.8, 0.9]
learn_rate = [0.001, 0.01, 0.1, 0.2, 0.3]
dropout_rate = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
init = ['uniform', 'lecun_uniform', 'normal', 'zero', 'glorot_normal', 'glorot_uniform', 'he_normal', 'he_uniform']
optimizer = [ 'SGD', 'RMSprop', 'Adagrad', 'Adadelta', 'Adam', 'Adamax', 'Nadam']

####################################################################
batch_size = [16, 32, 64, 128]
epochs = [100,250,500]
param_grid = dict(
    dense_layer = dense_size_candidates,
    activation = activation,
    momentum = momentum,
    learn_rate = learn_rate,  
    dropout_rate = dropout_rate,
    init_mode = init,
    optimizer = optimizer,
    batch_size=batch_size,
    epochs=epochs
    )
grid = GridSearchCV(estimator=model, param_grid=param_grid, n_jobs=-1, cv=5, verbose = 2)
grid_result = grid.fit(X, Y)
# summarize results
print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
means = grid_result.cv_results_['mean_test_score']
stds = grid_result.cv_results_['std_test_score']
params = grid_result.cv_results_['params']
for mean, stdev, param in zip(means, stds, params):
    print("%f (%f) with: %r" % (mean, stdev, param))
    
print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
print("--- %s s ---" % ((time.time() - start_time)))
