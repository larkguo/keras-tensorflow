# 多层感知神经网络,预测糖尿病
from keras.models import Sequential
from keras.layers import Dense
from keras.models import model_from_json
import numpy
import os

# fix random seed for reproducibility
numpy.random.seed(1)

# 1. 导入数据
dataset = numpy.loadtxt("pima-indians-diabetes.csv",delimiter=",")
# split into input (X) and output (Y) variables
X = dataset[:,0:8] # pima-indians-diabetes.csv 每行前8个值
Y = dataset[:,8] # pima-indians-diabetes.csv 每行第9个值

# 2. 定义模型
# 输入层有8个入参，隐含层具有12个神经元，激活函数采用的是relu;
# 输出层具有1个神经元，激活函数采用的是sigmoid
model = Sequential()
model.add(Dense(12, input_dim=8, kernel_initializer='uniform', activation='relu'))
model.add(Dense(8, kernel_initializer='uniform', activation='relu'))
model.add(Dense(1, kernel_initializer='uniform', activation='sigmoid'))

# 3. 编译模型
#优化器和损失函数类型分别为：adam和binary_crossentropy
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# 4. 训练模型
#训练周期为150，每次数据量为10
model.fit(X, Y, epochs=150, batch_size=10, verbose=0)

# 5. 评估模型
scores = model.evaluate(X, Y, verbose=0)
print("%s:%.2f, %s:%.2f%%"
      %(model.metrics_names[0],scores[0],model.metrics_names[1],scores[1]*100))

# 6.进行预测
# 用训练好的模型在新的数据上进行预测
# predictions是预测返回的结果，数据格式与输出层的输出格式相同。
predictions = model.predict(X) #predicting Y only using X
# Round predictions
rounded = [int(numpy.round(x, 0)) for x in predictions]
accuracy = numpy.mean(rounded == Y)
print("Prediction Accuracy: %.2f%%" % (accuracy*100))

# 7. 保存模型
model_json = model.to_json()
with open("model.json", "w") as json_file:
    json_file.write(model_json)
# serialize weights to HDF5
model.save_weights("model.h5")
print("Saved model to disk")

# 8. 加载模型
# load json and create model
json_file = open('model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)
# load weights into new model
loaded_model.load_weights("model.h5")
print("Loaded model from disk")

# evaluate loaded model on test data
loaded_model.compile(loss='binary_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
score = loaded_model.evaluate(X, Y, verbose=0)
print("%s:%.2f, %s:%.2f%%"
      %(model.metrics_names[0],scores[0],model.metrics_names[1],scores[1]*100))

