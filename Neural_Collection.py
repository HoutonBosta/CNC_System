#神经集合生成脚本

import tensorflow as tf 
import tflearn
import Visual_System
import Auditory_System
import Physiological_System
import Tactile_System
import DQN_model as DQN
import Connection
import multiprocessing as mp

class Building_Network(object):
	def	__init__(self,V,A,P,T,N):
		self.v = V
		self.a = A
		self.p = P
		self.t = T
		self.n = N

	def Visual_Network(self):
		
		with Visual_System.Building_Visual_System() as build_V:
			build_V.Building_CNN()
			build_V.Training()
			build_V.Building_CRNN()

		'''
		for i in range(self.b):

			with CNN.Building_CNN() as build_cnn:
				build_cnn.Building()


			with LSTM.Building_LSTM() as build_lstm:
				build_lstm.Building()

			with DQN.DQNPrioritizedReplay(_):
				pass

			with Connection.connection(_) as build_connection_besic:
				build_connection_besic.connect_next()
		'''
	def Auditory_Network(self):
		pass


	def Physiological_Network(self):
		pass

	def Tactile_Network(self):
		pass


	def Middle_Building(self):
		for i in range(self.m):
			with Connection.connection(_) as build_connection_middle:
				build_connection_middle.connect_front()

			with DQN.DQNPrioritizedReplay(_):
				pass

			with DQN.DQNPrioritizedReplay(_):
				pass

			with DQN.DQNPrioritizedReplay(_):
				pass

			with DQN.DQNPrioritizedReplay(_):
				pass



	def Next_Building(self):
		pass




if __name__ == '__main__':
	pass