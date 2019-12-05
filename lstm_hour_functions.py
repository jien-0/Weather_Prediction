import os
os.chdir("/Users/haimannmok/Desktop/Lisa/Careers/SkillingUp/MachineLearning/Mastering_Machine_Learning/Projects/04 Deep Learning/LSTM_temperature/")

import numpy as np
from pandas import DataFrame
from pandas import concat
import errno
from sklearn import preprocessing
from keras.models import Sequential
from keras.layers import LSTM
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import RepeatVector
from keras.layers import TimeDistributed
from keras.optimizers import SGD
import keras.backend as K
from math import sqrt
from sklearn.metrics import mean_squared_error


############## FRAME A TIME SERIES AS A SUPERVISED LEARNING DATASET ##############

def series_to_supervised(data, n_in=1, n_out=1, dropnan=True):

	n_vars = 1 if type(data) is list else data.shape[1]
	df = DataFrame(data)
	cols, names = list(), list()
	for i in range(n_in, 0, -1):
		cols.append(df.shift(i))
		names += [('var%d(t-%d)' % (j+1, i)) for j in range(n_vars)]
	for i in range(0, n_out):
		cols.append(df.shift(-i))
		if i == 0:
			names += [('var%d(t)' % (j+1)) for j in range(n_vars)]
		else:
			names += [('var%d(t+%d)' % (j+1, i)) for j in range(n_vars)]
	agg = concat(cols, axis=1)
	agg.columns = names
	if dropnan:
		agg.dropna(inplace=True)
	return agg


############## DATASET REFRAMING INTO TIMESTEP MATRIX ##############

def timestep_matrix(timesteps, dataset):
    dataset = dataset.iloc[:, np.r_[1:2, 5:len(dataset.columns)]] # Remove irrelevant columns - location, date, appliance
    values = dataset.astype('float32').values
    features = values.shape[1]  # No. of features include energy usage

    # Normalise values
    scaler = preprocessing.MinMaxScaler(feature_range=(0, 1))
    scaled = scaler.fit_transform(values)

    # Reframe dataset into time-step sequences
    reframed = series_to_supervised(scaled, timesteps, 1)  # 7 preceding timesteps, 1 output time-step
    column_drop = (np.array(range(features)) * -1)[1:features]
    reframed.drop(reframed.columns[column_drop], axis=1, inplace=True)

    # Transform commands for output values
    output_y = values[:, 0]
    output_y = output_y.reshape(output_y.shape[0], 1)
    scaled_y = scaler.fit_transform(output_y)
    # y = scaler.inverse_transform(scaled_y)

    return features, scaler, reframed, output_y, scaled_y

################### PREPARE TRAIN AND TEST SET ####################

# Use test period as n days prior to the last day
# Train period is all days prior to test period

def train_test(reframed, test_obs, timesteps, features):
    # split into test and training datasets
    data_test = reframed.values[-test_obs:]  # <- extract last n timesteps
    data_train = reframed.values[:-test_obs]

    # split out features and output
    test_X = data_test[:, :-1]  # <-removes most recent 24 hours
    test_Y = data_test[:, -1:]

    train_X = data_train[:, :-1]
    train_Y = data_train[:, -1:]

    # reshape features data into 3D arrays (samples, time-steps, features)
    train_X = train_X.reshape(train_X.shape[0], timesteps, features)
    test_X = test_X.reshape(test_X.shape[0], timesteps, features)

    return train_X, train_Y, test_X, test_Y


################### PREPARE TRAIN SET ####################

def train(reframed, timesteps, features):

    # split out features and output
    train_X = reframed.values[:, :-1]
    train_Y = reframed.values[:, -1:]

    # reshape features data into 3D arrays (samples, time-steps, features)
    train_X = train_X.reshape(train_X.shape[0], timesteps, features)

    return train_X, train_Y


################### LSTM MODEL ARCHITECT ####################


def lstm_model_temp(timesteps, features, neurons, dropout, learning, momentum, decay, init, activation ) :
   model = Sequential()
   model.add(LSTM(neurons, return_sequences=False, input_shape = (timesteps, features), kernel_initializer = init, activation = activation))
   model.add(Dropout(dropout))
   model.add(Dense(1))
   SGD( lr = learning, momentum = momentum, decay = decay, nesterov = False)
   model.compile(optimizer = 'adam', loss = 'mse')
   return model

