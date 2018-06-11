# -*- coding: utf-8 -*-
'''
Trains a simple convnet on the MNIST dataset.
Gets to 99.21% test accuracy after 12 epochs

ref:
    http://www.python36.com/mnist-handwritten-digits-classification-using-keras/
    http://www.python36.com/deploy-keras-model-to-production-using-flask/

run:
    [root@localhost keras-mnist-cnn]# python mnist_cnn.py
    Using TensorFlow backend.
    input_shape: (28, 28, 1)
    x_train shape: (60000, 28, 28, 1)
    60000 train samples
    10000 test samples
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
    Train on 60000 samples, validate on 10000 samples
    Epoch 1/12
    60000/60000 [==============================] - 65s 1ms/step - loss: 0.2801 - acc: 0.9131 - val_loss: 0.0603 - val_acc: 0.9798
    Epoch 2/12
    60000/60000 [==============================] - 65s 1ms/step - loss: 0.0925 - acc: 0.9720 - val_loss: 0.0402 - val_acc: 0.9861
    Epoch 3/12
    60000/60000 [==============================] - 65s 1ms/step - loss: 0.0701 - acc: 0.9796 - val_loss: 0.0344 - val_acc: 0.9887
    Epoch 4/12
    60000/60000 [==============================] - 65s 1ms/step - loss: 0.0568 - acc: 0.9828 - val_loss: 0.0331 - val_acc: 0.9888
    Epoch 5/12
    60000/60000 [==============================] - 65s 1ms/step - loss: 0.0487 - acc: 0.9855 - val_loss: 0.0306 - val_acc: 0.9892
    Epoch 6/12
    60000/60000 [==============================] - 65s 1ms/step - loss: 0.0442 - acc: 0.9866 - val_loss: 0.0286 - val_acc: 0.9906
    Epoch 7/12
    60000/60000 [==============================] - 65s 1ms/step - loss: 0.0390 - acc: 0.9884 - val_loss: 0.0292 - val_acc: 0.9904
    Epoch 8/12
    60000/60000 [==============================] - 65s 1ms/step - loss: 0.0358 - acc: 0.9891 - val_loss: 0.0271 - val_acc: 0.9916
    Epoch 9/12
    60000/60000 [==============================] - 65s 1ms/step - loss: 0.0319 - acc: 0.9898 - val_loss: 0.0290 - val_acc: 0.9910
    Epoch 10/12
    60000/60000 [==============================] - 65s 1ms/step - loss: 0.0307 - acc: 0.9907 - val_loss: 0.0267 - val_acc: 0.9917
    Epoch 11/12
    60000/60000 [==============================] - 65s 1ms/step - loss: 0.0266 - acc: 0.9920 - val_loss: 0.0257 - val_acc: 0.9918
    Epoch 12/12
    60000/60000 [==============================] - 65s 1ms/step - loss: 0.0274 - acc: 0.9918 - val_loss: 0.0243 - val_acc: 0.9921
    Test loss: 0.02432824007576928
    Test accuracy: 0.9921
'''

from __future__ import print_function
import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras import backend as K
from keras.models import model_from_json


batch_size = 128
num_classes = 10
epochs = 12

# input image dimensions
img_rows, img_cols = 28, 28

# 1. 导入数据
# the data, split between train and test sets
(x_train, y_train), (x_test, y_test) = mnist.load_data()
print('x_train:',x_train.shape)
print(x_train)
'''
#显示第n张数据图
import cv2
n = 0
pic = x_train[n].reshape(28, 28)
cv2.imshow("demo", pic)
cv2.waitKey(0)
cv2.destroyAllWindows()
'''

if K.image_data_format() == 'channels_first': #通道数在前
    x_train = x_train.reshape(x_train.shape[0], 1, img_rows, img_cols)
    x_test = x_test.reshape(x_test.shape[0], 1, img_rows, img_cols)
    input_shape = (1, img_rows, img_cols)
else: #通道数在后
    x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
    x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)
    input_shape = (img_rows, img_cols, 1)

x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255

# input_shape: (28, 28, 1)
# x_train shape: (60000, 28, 28, 1)
#60000 train samples
#10000 test samples
print('input_shape:', input_shape)
print('x_train shape:', x_train.shape)
print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')

# convert class vectors to binary class matrices, One-Hot编码
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)

#创建模型
model = Sequential()
model.add(Conv2D(32, kernel_size=(3, 3),
                 activation='relu',
                 input_shape=input_shape))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(num_classes, activation='softmax'))
model.summary()

#定义优化器，loss function，训练过程中计算准确率
model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=keras.optimizers.Adadelta(),
              metrics=['accuracy'])

#训练模型
model.fit(x_train, y_train,
          batch_size=batch_size,
          epochs=epochs,
          verbose=1,
          validation_data=(x_test, y_test))

#评估模型
score = model.evaluate(x_test, y_test, verbose=0)
#Test loss: 0.02432824007576928
#Test accuracy: 0.9921
print('Test loss:', score[0])
print('Test accuracy:', score[1])

#保存网络结构
model_json = model.to_json()
with open("model.json", "w") as json_file:
  json_file.write(model_json)

#保存权重参数
model.save_weights("model.h5")

