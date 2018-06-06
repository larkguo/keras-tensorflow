# 根据之前训练的模型，预测新数据得糖尿病的几率
from keras.models import Sequential
from keras.layers import Dense
from keras.models import model_from_json
import numpy as np

# 准备两组数据
A = [[5,140,72,38,0,33.8,0.626,51],[1,85,66,29,0,26.6,0.351,31]]
X = np.array(A)

# 加载模型
# load json and create model
json_file = open('model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)
# load weights into new model
loaded_model.load_weights("model.h5")
print("Loaded model from disk")

#预测新数据的得糖尿病的几率
predictions = loaded_model.predict(X)
print (predictions)

