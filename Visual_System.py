#视觉系统

'''
即将更新:
o全局HSV突兀注意机制
o运动解释完善
o多进程并行计算机制
o运动注意机制
'''
'''
图像像素读取方法
img=Image.open("demo.jpg")
RGB_array=img.load()
RGB_array[x,y]
'''

import tflearn
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.estimator import regression
import cv2
import numpy as np
from PIL import Image




# Data loading and preprocessing
#import tflearn.datasets.oxflower17 as oxflower17
#X, Y = oxflower17.load_data(one_hot=True)

# Building 'VGG first_Net1'

first_net1_learing_rate = 0.01
#first_net1_X,first_net1_Y,first_net1_testX,first_net1_testY = '数据集'
camera = cv2.VideoCapture(0)
width = int(camera.get(3))
height = int(camera.get(4))





class Building_Visual_System(object):
	

	def Building_CNN():#the besic image valu
		first_net1 = input_data(shape=[None, 224, 224, 3])

		first_net1 = conv_2d(first_net1, 64, 3, activation='relu')
		first_net1 = conv_2d(first_net1, 64, 3, activation='relu')
		first_net1 = max_pool_2d(first_net1, 2, strides=2)

		first_net1 = conv_2d(first_net1, 128, 3, activation='relu')
		first_net1 = conv_2d(first_net1, 128, 3, activation='relu')
		first_net1 = max_pool_2d(first_net1, 2, strides=2)

		first_net1 = conv_2d(first_net1, 256, 3, activation='relu')
		first_net1 = conv_2d(first_net1, 256, 3, activation='relu')
		first_net1 = conv_2d(first_net1, 256, 3, activation='relu')
		first_net1 = max_pool_2d(first_net1, 2, strides=2)

		first_net1 = conv_2d(first_net1, 512, 3, activation='relu')
		first_net1 = conv_2d(first_net1, 512, 3, activation='relu')
		first_net1 = conv_2d(first_net1, 512, 3, activation='relu')
		first_net1 = max_pool_2d(first_net1, 2, strides=2)

		first_net1 = conv_2d(first_net1, 512, 3, activation='relu')
		first_net1 = conv_2d(first_net1, 512, 3, activation='relu')
		first_net1 = conv_2d(first_net1, 512, 3, activation='relu')
		first_net1 = max_pool_2d(first_net1, 2, strides=2)

		first_net1 = fully_connected(first_net1, 4096, activation='relu')
		first_net1 = dropout(first_net1, 0.5)
		first_net1 = fully_connected(first_net1, 4096, activation='relu')
		first_net1 = dropout(first_net1, 0.5)
		first_net1 = fully_connected(first_net1, 17, activation='softmax')

		first_net1 = regression(first_net1, optimizer='rmsprop',
                     		loss='categorical_crossentropy',
                     		learning_rate=0.0001)

		
	def Building_CRNN():#express action
		firstFrame = None
		while True:
			(grabbed, frame) = camera.read()
			gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
			gray = cv2.GaussianBlur(gray, (21, 21), 0)
  			
			if firstFrame is None:
				firstFrame = gray
				continue
            
			frameDelta = cv2.absdiff(firstFrame, gray)
			thresh = cv2.threshold(frameDelta, 25, 255, cv2.THRESH_BINARY)[1]
			# thresh = cv2.adaptiveThreshold(frameDelta,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,\
			# cv2.THRESH_BINARY,11,2)
			# thresh = cv2.adaptiveThreshold(frameDelta,255,cv2.ADAPTIVE_THRESH_MEAN_C,\
			#           cv2.THRESH_BINARY,11,2)
			thresh = cv2.dilate(thresh, None, iterations=2)
			(_, cnts, _) = cv2.findContours(thresh.copy(), cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
			for c in cnts:
				if cv2.contourArea(c) < 10000:
					continue
					(x, y, w, h) = cv2.boundingRect(c)

					cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

					cv2.imshow("Security Feed", frame)
  
					firstFrame = gray.copy()

    		#camera.release()
			#cv2.destroyAllWindows()
	
	

	def array_convolution(array,x,y):#array:目标矩阵
		array_0 = array
		array_shape = array.shape
		array_member = len(array_shape)
		#array_ = np.zeros(((array_shape[0],array_shape[1],_))) #生成二维数组容器
		#array_2 = np.zeros((3,3))
		if array_member is 2:		
			#初始化:分成100等份,每次循环都再分100等份
			w = array_shape[0] / x
			h = array_shape[1] / y
			part_size = w * h
			array_ = np.zeros((((x,y,w,h))))
			for x0 in range(x):
				for y0 in range(y):
					array_part = np.zeros((w,h))
					for x_ in range(w):
						for y_ in range(h):
							X = x_ + x0 * w
							Y = y_ + y0 * h
							array_part[x_,y_] = array_0[X,Y] #part矩阵(2维)
					array_[x0,y0] = array_part
			return array_,part_size
		

		if array_member is 3:
			w = array_shape[0] / x
			h = array_shape[1] / y
			part_size = w * h
			array_ = np.zeros(((((x,y,w,h,3)))))
			for x0 in range(x):
				for x0 in range(y):
					array_part = np.zeros(((w,h,3)))
					for x_ in range(w):
						for y_ in range(h):
							for d in range(3):
								X = x_ + x0 * w
								Y = y_ + y0 * w
								array_part[x_,y_] = array_0[X,Y]#part矩阵(3维)
					array_[x0,y0] = array_part
			return array_,part_size

				




	def RGB_Detection():
		#img=Image.open("demo.jpg")
		RGB_plus_ = []
		RGB_plus_array = []
		RGB_array=img.load() #获取图像RGB矩阵

		#计算亮度
		w = img.size[0]
		h = img.size[1]
		RGB_light = np.zeros((w,h)) #图像亮度容器
		RGB_class = np.zeros(((w,h,3))) #图像颜色容器

		#填充容器
		for x in range(w):
			for y in range(h):
				r,g,b = RGB_array[x,y]
				rgb = r + g + b
				RGB_light[x,y] = rgb
				RGB_class[x,y] = [r,g,b]

		#对二维数组分区
		RGB_lib,part_shape = array_convolution(RGB_light,10,10)
		#对三维数组分区
		RGB_lib_,part_shape_ = array_convolution(RGB_class,10,10)


		#计算每个区域的平均光亮度
		average_light = np.zeros((10,10))
		for x in range(10):
			for y in range(10):

				average_light[x,y] = sum(RGB_lib[x,y]) / part_shape

		#计算每个区域的平均颜色

		w = RGB_class.shape[0] / 10
		h = RGB_class.shape[1] / 10

		average_color_r = np.zeros(((10,10,w,h)))
		average_color_g = np.zeros(((10,10,w,h)))
		average_color_b = np.zeros(((10,10,w,h)))


		#填充平均颜色容器
		for x in range(10):
			for y in range(10):
				
				array_part_r = np.zeros((w,h))
				array_part_g = np.zeros((w,h))
				array_part_b = np.zeros((w,h))

				for w_ in range(w):
					for h_ in range(h):
						X = w_ + x * w
						Y = h_ + y * w	
						array_part_r[w_,h_] = RGB_class[x,y,X,Y,0]
						array_part_g[w_,h_] = RGB_class[x,y,X,Y,1]
						array_part_b[w_,h_] = RGB_class[x,y,X,Y,2]

				average_color_r[x,y] = array_part_r
				average_color_g[x,y] = array_part_g
				average_color_b[x,y] = array_part_b

		#平均颜色容器求平均
		average_color_rgb = np.zeros(((10,10,3)))
		for x in range(10):
			for y in range(10):
				average_color_rgb[x,y,0] = sum(average_color_r[x,y]) / part_shape
				average_color_rgb[x,y,1] = sum(average_color_r[x,y]) / part_shape
				average_color_rgb[x,y,2] = sum(average_color_r[x,y]) / part_shape



		#计算光亮对比度{寻找较亮的与较暗的区域(=)计算极端区域的距离(=)给出对比度较大的坐标  } ？--> 周边值算法
		#计算总体平均光亮度
		for x in range(10):
			for y in range(10):
				average_light_all = average_light_all + average_light[x,y]

		average_light_total = average_light_all / 100
		'''
		#计算平均颜色
		average_color_rgb_total = [0,0,0]
		for x in range(10):
			for y in range(10):
				average_color_rgb_total[0] = average_color_rgb_total[0] + average_color_rgb[x,y,0]
				average_color_rgb_total[1] = average_color_rgb_total[1] + average_color_rgb[x,y,1]
				average_color_rgb_total[2] = average_color_rgb_total[2] + average_color_rgb[x,y,2]

		average_color_rgb_total[0] = average_color_rgb_total[0] / 100
		average_color_rgb_total[1] = average_color_rgb_total[1] / 100
		average_color_rgb_total[2] = average_color_rgb_total[2] / 100
		'''

		return average_light_total,average_light,average_color_rgb



	def RGB_Attention(light_total,Light_array,RGB_array_):
		#创建注意区域容器
		light_loss = np.zeros((10,10))
		rgb_loss = np.zeros(((10,10,3)))
		#计算区域与平均光亮的差距并找出
		for x in range(10):
			for y in range(10):
				light_loss[x,y] = abs(Light_array[x,y] - light_total)
		'''		
		for x in range(10):
			for y in range(10):
				rgb_batch[x,y,1] = abs(RGB_array[x,y,1] - )
		'''
		attention_light = np.zeros((10,10))
		for x in range(10):
			for y in range(10):
				attention_light[x,y] = light_loss[x,y] / 255


		return attention_light








		#根据找出的近似色计算出临近区域的对比度
		#根据临近的对比度进行评估(评估值为-10~10的浮点数)




	def Training():
		# Training
		model = tflearn.DNN(first_net1, checkpoint_path='model_vgg',
                    		max_checkpoints=1, tensorboard_verbose=0)
		model.fit(X, Y, n_epoch=500, shuffle=True,
          		show_metric=True, batch_size=32, snapshot_step=500,
          		snapshot_epoch=False, run_id='vgg_oxflowers17')
