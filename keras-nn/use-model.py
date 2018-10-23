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

