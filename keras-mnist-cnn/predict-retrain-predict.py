# encoding: utf-8
'''
三个过程：
    1. 使用之前的训练模型，结果把图片6识别为5
    2. 使用该图片6 继续训练
    3. 再次识别正确

debug信息：
    ./debug/6-1.png 图中数字为:'5'
    x_train.shape: (1, 28, 28) x_train samples: 1
    y_train.shape: (1,) y_train: [6.]
    Epoch 1/2

    1/1 [==============================] - 1s 554ms/step - loss: 1.3771 - acc: 0.0000e+00
    Epoch 2/2

    1/1 [==============================] - 0s 25ms/step - loss: 0.0046 - acc: 1.0000
    _________________________________________________________________
    Layer (type)                 Output Shape              Param #
    =================================================================
    conv2d_1 (Conv2D)            (None, 26, 26, 32)        320
    _________________________________________________________________
    conv2d_2 (Conv2D)            (None, 24, 24, 64)        18496
    _________________________________________________________________
    max_pooling2d_1 (MaxPooling2 (None, 12, 12, 64)        0
    _________________________________________________________________
    dropout_1 (Dropout)          (None, 12, 12, 64)        0
    _________________________________________________________________
    flatten_1 (Flatten)          (None, 9216)              0
    _________________________________________________________________
    dense_1 (Dense)              (None, 128)               1179776
    _________________________________________________________________
    dropout_2 (Dropout)          (None, 128)               0
    _________________________________________________________________
    dense_2 (Dense)              (None, 10)                1290
    =================================================================
    Total params: 1,199,882
    Trainable params: 1,199,882
    Non-trainable params: 0
    _________________________________________________________________
    ./debug/6-1.png 图中数字为:"6"
'''
import os
import cv2
import numpy as np
import keras
from keras.models import Sequential
from keras.models import model_from_json
from keras.utils import np_utils

#变量
file = './debug/6-1.png'    #被识别的数字图片
imgs =1                        #被识别的图片总数
img_rows, img_cols = 28, 28    # input image dimensions
channels=1                     #通道数1:只有灰度
model_json = 'model.json'    #网络结构
model_h5 = 'model.h5'        #参数权重
epochs = 2                     #分几组训练
batch_size = 1                 #每组训练图片数
num_classes = 10               #0-9手写数字一个有10个类别
digit = 6                      #正确的数字标签，再次训练时使用

############################# 1.使用之前的训练模型，结果把数字图片6识别为5 #############################
# 读取model
model = model_from_json(open(model_json).read())
model.load_weights(model_h5)

# 读取灰度图片，调整大小为28*28,与训练时一致
pic = cv2.imread(file,cv2.IMREAD_GRAYSCALE)
pic = cv2.resize(pic, (img_rows, img_cols))
pic = pic.astype('float32')

#shape与训练时conv2d_1_input一致 reshape(图片数1，行，列，通道数1只有灰度，)
pic_reshape = pic.reshape(imgs,img_rows,img_cols,channels)
print("%s 图中数字为:'%d'"%(file,model.predict_classes(pic_reshape, verbose=0)[0]))

############################# 2. 使用该数字图片 继续训练 #############################
# 准备训练数据
x_train =  np.empty((imgs,img_rows,img_cols),dtype="float32")
y_train =  np.empty((imgs),dtype="float32")
x_train[0,:,:] = np.asarray(pic,dtype="float32")
y_train[:] = digit
print('x_train.shape:',x_train.shape,'samples:',x_train.shape[0])
print('y_train.shape:',y_train.shape,"y_train:",y_train)
x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, channels)
x_train = x_train.astype('float32')
x_train /= 255
print('x_train.shape:',x_train.shape,'samples:',x_train.shape[0])

# convert class vectors to binary class matrices, One-Hot编码
y_train = keras.utils.to_categorical(y_train, num_classes)
print('y_train.shape:',y_train.shape,"y_train:",y_train)

# 定义优化器，loss function，训练过程中计算准确率
model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=keras.optimizers.Adadelta(),
              metrics=['accuracy'])

#再次训练模型
model.fit(x_train, y_train,batch_size,epochs,verbose=1)

#打印模型，未变化
#model.summary()
############################# 3. 再次识别正确 ##############################

print("%s 图中数字为:\"%d\""%(file,model.predict_classes(pic_reshape, verbose=0)[0]))

#显示图片，再次保存训练结果
'''
#窗口显示数据
cv2.namedWindow("Image1")
cv2.imshow("Image1", pic)
#等待按键，退出显示
cv2.waitKey(0)
cv2.destroyAllWindows()

#保存网络结构
model_json = model.to_json()
with open("model.json", "w") as json_file:
  json_file.write(model_json)

#保存权重参数
model.save_weights("model.h5")
'''



