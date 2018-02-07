import tflearn
import tensorflow as tf 
import os
'''
即将更新:
o音频对比机制
'''



first_net2_batch_size = 64
first_net2_input_size = 20
first_net2_height_size = 0
first_net2_classes = 10
first_net2_learing_rate = 0.0001
#first_net2_batch = word_first_net2_batch = speech_data.mfcc_batch_generator(first_net2_batch_size) #-------------声音数据
first_net2_X,first_net2_Y = next(first_net2_batch)
first_net2_trainX,first_net2_trainY = first_net2_X,first_net2_Y
first_net2_testX,first_net2_testY = first_net2_X,first_net2_Y



first_net3_learing_rate = 0.001

first_net3_classes = 10
first_net3_input_size = 0
first_net3_batch_size = 0
first_net3_height_size = 0
#first_net3_batch = #模拟生理数据
first_net3_X,first_net3_Y = next(first_net3_batch)
first_net3_trainX,first_net3_trainY = first_net3_X,first_net3_Y
first_net3_testX,first_net3_testY = first_net3_X,first_net3_Y



first_net4_learing_rate = 0.001
first_net4_classes = 10
first_net4_input_size = 0
first_net4_batch_size = 0
first_net4_height_size = 0
#听觉系统



#first_net4_batch = #模拟生理数据
first_net4_X,first_net4_Y = next(first_net4_batch)
first_net4_trainX,first_net4_trainY = first_net4_X,first_net4_Y
first_net4_testX,first_net4_testY = first_net4_X,first_net4_Y

class Building_Auditory_System(object):
	def __init__(self):
		pass
	def Building():
		#LSTM1
		first_net2 = tflearn.input_data([None,first_net2_input_size,first_net2_height_size])
		first_net2 = tflearn.lstm(first_net2, 128, dropout = 0.8)
		first_net2 = tflearn.fully_connected(first_net2,first_net2_classes, activation = 'softmax')
		first_net2 = tflearn.regression(first_net2, optimizer = 'adam', learning_rate = first_net2_learing_rate,loss = 'categorical_crossentropy') 
		


		#LSTM2
		first_net3 = tflearn.input_data([None,first_net3_input_size,first_net3_height_size])
		first_net3 = tflearn.lstm(first_net3, 128, dropout = 0.8)
		first_net3 = tflearn.fully_connected(first_net3,first_net3_classes, activation = 'softmax')
		first_net3 = tflearn.regression(first_net3, optimizer = 'adam', learning_rate = first_net3_learing_rate,loss = 'categorical_crossentropy')
		


		#LSTM3
		first_net4 = tflearn.input_data([None,first_net4_input_size,first_net4_height_size])
		first_net4 = tflearn.lstm(first_net4, 128, dropout = 0.8)
		first_net4 = tflearn.fully_connected(first_net4,first_net4_classes, activation = 'softmax')
		first_net4 = tflearn.regression(first_net4, optimizer = 'adam', learning_rate = first_net4_learing_rate,loss = 'categorical_crossentropy')


	def Audio_Compare():
		pass


	def Training(net2 = False,net3 = False,net4 = False):

		if net2 is True:
			first_model2 = tflearn.DNN(first_net2, tensorboard_verbose = 0)
			while 1:
				first_model2.fit(first_net2_trainX,first_net2_trainY, n_epoch = 10, validation_set = (first_net2_testX,first_net2_testY),show_metric = True, batch_size = first_net2_batch_size)
				_first_net2_y = first_model2.predict(first_net2_X)
			first_model2.save("first_net2.first_model2")

		if net3 is True:
			first_model3 = tflearn.DNN(first_net3, tensorboard_verbose = 0)
			while 1:
				first_model3.fit(first_net3_trainX,first_net3_trainY, n_epoch = 10, validation_set = (first_net3_testX,first_net3_testY),show_metric = True, batch_size = first_net3_batch_size)
				_first_net3_y = first_model3.predict(first_net3_X)
			first_model3.save("first_net3.first_model3")

		if net4 is True:
			first_model4 = tflearn.DNN(first_net4, tensorboard_verbose = 0)
			while 1:
				first_model4.fit(first_net4_trainX,first_net4_trainY, n_epoch = 10, validation_set = (first_net4_testX,first_net4_testY),show_metric = True, batch_size = first_net4_batch_size)
				_first_net4_y = first_model4.predict(first_net4_X)
			first_model4.save("first_net4.first_model4")
