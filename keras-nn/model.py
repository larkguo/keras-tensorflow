# 多层感知神经网络,创建预测糖尿病模型
from keras.models import Sequential
from keras.layers import Dense
from keras.models import model_from_json
import numpy
import matplotlib.pyplot as plt

# fix random seed for reproducibility
numpy.random.seed(7)

# 1. 导入数据
dataset = numpy.loadtxt("data/pima-indians-diabetes.csv",delimiter=",")
# split into input (X) and output (Y) variables
X = dataset[:,0:8] # pima-indians-diabetes.csv 每行前8个值
Y = dataset[:,8] # pima-indians-diabetes.csv 每行第9个值

# 2. 定义模型
# 创建多层感知机模型：输入层有8个入参，隐含层具有12个神经元，激活函数采用的是 relu;
# 输出层具有1个神经元，激活函数采用的是sigmoid
model = Sequential()
model.add(Dense(12, input_dim=8, kernel_initializer='uniform', activation='relu'))
model.add(Dense(8, kernel_initializer='uniform', activation='relu'))
model.add(Dense(1, kernel_initializer='uniform', activation='sigmoid'))

# 3. 编译模型
#编译过程将我们所定义的简单的图层序列模型转换成一系列可以高效执行的矩阵
#优化器和损失函数类型分别为：adam 和 binary_crossentropy
#keras默认测量loss损失率，还可手动添加accuracy准确性。
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# 4. 训练模型
#使用训练数据集不断调整网络上各节点的权重
#网络模型会使用反向传播算法进行训练，并根据编译模型时指定的优化算法和损失函数进行优化。
#训练周期为150，每次数据量为10
#history contains summary of the training loss and metrics recoded each epoch
history = model.fit(X, Y, epochs=150, batch_size=10, verbose=0)

# 5. 保存模型
model_json = model.to_json()
with open("model.json", "w") as json_file:
    json_file.write(model_json)
# serialize weights to HDF5
model.save_weights("model.h5")
print("Saved model to disk")

# 6. 评估模型
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


