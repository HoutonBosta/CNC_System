#神经连接

import tensorflow as tf 
import tflearn
import CNN_model as CNN
import LSTM_model as LSTM
import DQN_model as DQN

class connection(object):
	def __init__(self,Eval,DQN_action):
		self.eval = Eval
		self.action = DQN_action

	def connect_front(self):#联系前一层单元的function
		pass


	def connect_next(self):#后一层的connection_function联系单元
		pass

	def forget_function(self):
		pass