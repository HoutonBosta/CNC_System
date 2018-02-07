'''
from __future__ import division, print_function, absolute_import
import tensorflow as tf 
import os
import tflearn
from tflearn.layers.core import input_data, dropout, fully_connected  
from tflearn.layers.conv import conv_2d, max_pool_2d  
from tflearn.layers.normalization import local_response_normalization  
from tflearn.layers.estimator import regression
'''
from __future__ import division, print_function, absolute_import
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.estimator import regression
import tflearn
import numpy as np 
import DQN_model.py 
import Visual_System.py
import Auditory_System.py 
import Tactile_System.py 
import Physiological_System.py




#环境特征全局变量
#基本情感
Threats_V = 0
Favorable_V = 0
Sex_V = 0
Fear_V = 0

Threats_H = 0
Favorable_H = 0
Sex_H = 0
Fear_H = 0

Threats_B = 0
Favorable_B = 0
Sex_B = 0
Fear_B = 0

Threats_T = 0
Favorable_T = 0
Sex_T = 0
Fear_T = 0


Innate_first = 5
Innate_Second = 3


class Nerve_group(object):
	def visual_system():
		pass

	def auditory_system():
		pass

	def tactile_system():
		pass

	def physiological_system():
		pass


	def Deep_Network_Create():
		pass

	def front_connection_function():
		pass

	def behind_connection_function():
		pass

class Innate(object):#先天性本能行为
	def  __init__(self,innate_first,innate_Second,init = False):
		V,H,B,T = self.read_eval()


		#检测文件夹 & 创造文件夹
		if os.path.exists('/CNC_system/Data_Cloud') is False:
			os.mkdir('/CNC_system/Data_Cloud')
			print('==Data Cloud Created successfully==')
		
		if os.path.exists('/CNC_system/Online_even') is False:
			os.mkdir('/CNC_system/Online_even')
			print('==Online even Created successfully==')
		
		if os.path.exists('/CNC_system/Online_even/Visual') is False:
			os.mkdir('/CNC_system/Online_even/Visual')
			#print('====')

		if os.path.exists('/CNC_system/Online_even/Hearing') is False:
			os.mkdir('/CNC_system/Online_even/Hearing')
			#print('====')

		if os.path.exists('/CNC_system/Online_even/Body') is False:
			os.mkdir('/CNC_system/Online_even/Body')
			#print('====')

		if os.path.exists('/CNC_system/Online_even/Tactile') is False:
			os.mkdir('/CNC_system/Online_even/Tactile')
			#print('====')
		
		if os.path.exists('/CNC_system/Initialize_the_feature_dataset') is False:
			os.mkdir('/CNC_system/Initialize_the_feature_dataset')					
	    
	    #生成初级先天性神经网络层
		for _ in range(innate_first):
			self.First_Network_Create(V,H,B,T)

		#生成高级先天性神经网络层
		for _ in range(innate_Second):
			self.Second_Network_Create()






	def read_eval(self): #从Online_eval文件夹中读入eval
		#读入视觉
		visual = Open('/Online_even/Visual/visual').read()
		#读入听觉
		hearing = Open('/Online_even/Hearing/hearing').read()
		#读入生理
		body = Open('/Online_even/Body/body').read()
		#读入触觉
		tactile = Open('/Online_even/Tactile/tactile').read()
		
		return visual,hearing,body,tactile


class Acquired(object):#后天性意识
	def __init__(self):
		pass

	def Basic_Network_Create(self):
		
		pass





class Robot_Communication(object):#云端与机器人通讯交接
	def eval_memory_change():#更新Online中的eval
		pass
	def Data_Cloud_Add_I():#基于先天行为添加记忆
		pass

#INNATE = Innate(Innate_first,Innate_Second)

