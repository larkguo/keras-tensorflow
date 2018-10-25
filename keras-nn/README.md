
# Keras搭建神经网络进行疾病预测


## 1. 概要

### 
   使用大量病人化验数据和信息训练神经网络模型，使用训练好的模型进行新病人疾病预测。
   总体过程：
![image](https://github.com/larkguo/keras-tensorflow/blob/master/keras-nn/data/train-predict.png)


## 2. 模型训练

### 

   训练过程：
   
![image](https://github.com/larkguo/keras-tensorflow/blob/master/keras-nn/data/steps.png) 

   网络原理：
   
![image](https://github.com/larkguo/keras-tensorflow/blob/master/keras-nn/data/network.png) 

   准备数据,pima-indians-diabetes.csv为糖尿病大数据：
   
![image](https://github.com/larkguo/keras-tensorflow/blob/master/keras-nn/data/pima-indians-diabetes.PNG)

    每一行有以下9项：
		1. Number of times pregnant 怀孕的次数
		2. Plasma glucose concentration a 2 hours in an oral glucose tolerance test 在口服葡萄糖耐量试验中2小时的血浆葡萄糖浓度
		3. Diastolic blood pressure (mm Hg) 舒张压（mm Hg）
		4. Triceps skin fold thickness (mm) 三头肌皮褶厚度（mm）
		5. 2-Hour serum insulin (mu U/ml) 2小时血清胰岛素（mu U / ml）
		6. Body mass index (weight in kg/(height in m)^2) 体重指数（体重（kg）/身高（m）^ 2）
		7. Diabetes pedigree function 糖尿病谱系功能
		8. Age (years) 年龄（岁）
		9. Class variable (0 or 1) 类变量（0为费糖尿病，1为糖尿病）
		
训练代码见[model.py](https://github.com/larkguo/keras-tensorflow/blob/master/keras-nn/model.py):
    
		# -*- coding: utf-8 -*-

		# 多层感知神经网络,创建预测糖尿病模型
		from keras.models import Sequential
		from keras.layers import Dense
		from keras.layers import Dropout
		from keras.models import model_from_json
		import numpy
		import matplotlib.pyplot as plt
		from keras.callbacks import TensorBoard

		# fix random seed for reproducibility
		numpy.random.seed(7)

		# 1. 准备数据
		# 数据预处理：导入数据集，区分训练和验证数据
		dataset = numpy.loadtxt("data/pima-indians-diabetes.csv",delimiter=",")
		# split into input (X) and output (Y) variables
		X = dataset[:,0:8] # pima-indians-diabetes.csv 每行前8个值
		Y = dataset[:,8] # pima-indians-diabetes.csv 每行第9个值

		# 2. 定义模型
		# 创建多层感知机模型：输入层有8个入参，隐含层具有12个神经元，激活函数采用的是 relu;
		# 输出层具有1个神经元，激活函数采用的是sigmoid
		model = Sequential()
		model.add(Dense(12, input_dim=8, kernel_initializer='uniform', activation='relu'))
		#Dropout 应用于输入，防止过拟合，此处丢弃的输入比例为0.1
		model.add(Dropout(0.1))
		model.add(Dense(8, kernel_initializer='uniform', activation='relu'))
		model.add(Dropout(0.1))
		model.add(Dense(1, kernel_initializer='uniform', activation='sigmoid'))
		model.summary()

		# 3. 编译模型
		# 编译过程将我们所定义的简单的图层序列模型转换成一系列可以高效执行的矩阵
		# 优化器和损失函数类型分别为：adam 和 binary_crossentropy
		# keras默认测量loss损失率，还可手动添加accuracy准确性。
		model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

		# 4. 训练模型
		# 使用训练数据集不断调整网络上各节点的权重
		# 网络模型会使用反向传播算法进行训练，并根据编译模型时指定的优化算法和损失函数进行优化。
		# 训练周期为150，每次数据量为10
		# history contains summary of the training loss and metrics recoded each epoch
		history = model.fit(X,Y,epochs=150,batch_size=10,verbose=0,
		                    callbacks=[TensorBoard(log_dir='data')])

		# 5. 保存模型
		# 模型为JSON或YAML格式,模型权重保存为HDF5格式
		model_json = model.to_json()
		with open("model.json", "w") as json_file:
		    json_file.write(model_json)
		model.save_weights("model.h5")
		print("Saved model to disk")

		# 6. 评估模型
		# 评估loss损失率和accuracy准确率
		scores = model.evaluate(X, Y, verbose=0)
		print("%s:%.2f, %s:%.2f%%"
		      %(model.metrics_names[0],scores[0],model.metrics_names[1],scores[1]*100))

		# 7.进行预测
		# 用训练好的模型在新的数据上进行预测
		# predictions是预测返回的结果，数据格式与输出层的输出格式相同。
		predictions = model.predict(X) #predicting Y only using X
		# Round predictions
		rounded = [int(numpy.round(x, 0)) for x in predictions]
		accuracy = numpy.mean(rounded == Y)
		print("Prediction Accuracy: %.2f%%" % (accuracy*100))

		'''
		# 绘制fit训练历史数据acc准确率，loss损失
		print(history.history.keys())
		plt.plot(history.history['acc'])
		plt.title('model accuracy')
		plt.ylabel('accuracy')
		plt.xlabel('epoch')
		plt.legend(['train'], loc='upper left')
		plt.show()
		plt.plot(history.history['loss'])
		plt.title('model loss')
		plt.ylabel('loss')
		plt.xlabel('epoch')
		plt.legend(['train'], loc='upper left')
		plt.show()
		'''

###
		启动TensorBoard：
![image](https://github.com/larkguo/keras-tensorflow/blob/master/keras-nn/data/tensorboard-start.PNG) 
	
		TensorBoard查看神经网络模型，可看到每层dense有反馈：
![image](https://github.com/larkguo/keras-tensorflow/blob/master/keras-nn/data/tensorboard-graphs.png) 
	
		训练后生成model.json模型和model.h5权重，模型准确率达 79.04%:
![image](https://github.com/larkguo/keras-tensorflow/blob/master/keras-nn/data/model.PNG) 
		

 
## 3. 使用模型预测疾病

###
    加载model.json模型和model.h5权重，对两组新病人数据进行疾病预测，同时可修正模型再次预测:
![image](https://github.com/larkguo/keras-tensorflow/blob/master/keras-nn/data/use-model.PNG) 

训练代码见[use_model.py](https://github.com/larkguo/keras-tensorflow/blob/master/keras-nn/use_model.py):
		#!/usr/bin/python
		# -*- coding: utf-8 -*-

		from keras.models import Sequential
		from keras.layers import Dense
		from keras.models import model_from_json
		import numpy as np

		#============================ 1. 使用训练好的模型进行预测 ============================
		# 准备两组待预测的病人数据
		A = [[5,140,72,38,0,33.8,0.626,51],[1,85,66,29,0,26.6,0.351,31]]
		X = np.array(A)

		# 加载模型
		json_file = open('model.json', 'r')
		loaded_model_json = json_file.read()
		json_file.close()
		loaded_model = model_from_json(loaded_model_json)
		loaded_model.load_weights("model.h5")
		print("Loaded model from disk")

		#预测新数据的得糖尿病的几率
		predictions = loaded_model.predict(X)
		print (predictions)
		#============================== 2. 再次训练和预测 ==============================
		# X两组数据都设置为1糖尿病
		print ("retrain:")
		Y = np.array([1,1])
		loaded_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
		history = loaded_model.fit(X, Y, epochs=150, batch_size=10, verbose=0)

		#预测新数据的得糖尿病的几率
		predictions = loaded_model.predict(X)
		print (predictions)
