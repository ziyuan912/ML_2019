import csv
import numpy as np
import os
import keras
from keras.models import Sequential, load_model
from keras.layers import Dense, LSTM
import argparse
from util import *

parser = argparse.ArgumentParser()
parser.add_argument('mode', choices=['train','predict'])
parser.add_argument('-output', help= 'Submission name', default= 'RNN.csv')
args = parser.parse_args()

def WMAE(y_true, y_pred):
	losses = keras.losses.mean_absolute_error(y_true[:, 0], y_pred[:, 0])*200 + keras.losses.mean_absolute_error(y_true[:, 1], y_pred[:, 1])*1 + keras.losses.mean_absolute_error(y_true[:, 2], y_pred[:, 2])*300
	return losses

def train(model_name):
	x_train, y_train, x_valid, y_valid = get_train_data(0.2)
	x_train2 = []
	x_valid2 = []
	for i in range(50):
		x_train2.append(x_train[:, 5000+100*i:5000+100*(i + 1)])
		x_valid2.append(x_valid[:, 5000+100*i:5000+100*(i + 1)])
	x_train2 = np.array(x_train2)
	x_valid2 = np.array(x_valid2)
	model1 = load_model('RNN_model_weights.h5', custom_objects={'WMAE': WMAE})
	x_train3 = model1.predict(x_train[:, :200])
	x_valid3 = model1.predict(x_valid[:, :200])
	model = [Sequential() for i in range(50)]
	for i in range(50):
		model[i].add(LSTM(9, activation = 'relu', input_shape = (10, 10), return_sequences = False))
		model[i].add(Dense(3, activation = 'linear'))
		x_train_temp = x_train2[i].reshape(x_train.shape[0], 10, 10)
		x_valid_temp = x_valid2[i].reshape(x_valid.shape[0], 10, 10)

		#model.output_shape
		model[i].compile(loss = WMAE, optimizer = 'adam', metrics = ['accuracy'])
		reduce_lr = keras.callbacks.ReduceLROnPlateau(monitor='val_loss', patience=10, mode='auto')
		model[i].fit(x_train_temp, y_train, verbose = 1, epochs = 10, callbacks=[reduce_lr], validation_data=(x_valid_temp, y_valid))

		score = model[i].evaluate(x_valid_temp, y_valid,verbose=1)
		print(i, 'th iteration: ', score)
		model[i].save('VAC_' + str(i) + '.h5')
		y_train_pred = model[i].predict(x_train_temp)
		y_valid_pred = model[i].predict(x_valid_temp)
		x_train3 = np.column_stack((x_train3, y_train_pred))
		x_valid3 = np.column_stack((x_valid3, y_valid_pred))

	del x_train2, x_valid2, x_train, x_valid
	model_final = Sequential()
	model_final.add(Dense(50, activation = 'relu', input_shape = (51, )))
	model_final.add(Dense(100, activation = 'relu'))
	model_final.add(Dense(100, activation = 'relu'))
	model_final.add(Dense(3, activation = 'linear'))
	model_final.compile(loss = WMAE, optimizer = 'adam', metrics = ['accuracy'])
	reduce_lr = keras.callbacks.ReduceLROnPlateau(monitor='val_loss', patience=10, mode='auto')
	model[i].fit(x_train3, y_train, verbose = 1, epochs = 50, callbacks=[reduce_lr], validation_data=(x_valid3, y_valid))
	score = model_final.evaluate(x_valid3, y_valid, verbose = 1)
	print('final score:', score)
	model_final.save('final.h5')
	del x_train3, x_valid3
	return

def predict(model_name, file_name):
	x_test = get_test_data()
	x_test = x_test[:, :5000]
	x_test = x_test.reshape(x_test.shape[0], 50, 100)
	model = load_model(model_name, custom_objects={'WMAE': WMAE})
	pred = model.predict(x_test)
	del x_test
	write_submission(pred, file_name)


if __name__ == '__main__':
	if args.mode == 'train':
		print('- TRAIN -')
		train(args.model)
	else:
		print('- PREDICT -')
		predict(args.model, args.output)
