#imports
import numpy as np
from mnist import MNIST
import os
import keras
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout, MaxPooling2D, Conv2D, Flatten
from keras import optimizers 
from keras import backend as K
from keras.utils import plot_model
from keras.callbacks import TensorBoard
import pydot
import graphviz

#laod the MNIST Data
#downloaded from yann.lecun.com
#use the MNIST library to import images from idx1/3-ubyte data
mndata = MNIST('/Users/natelinden/anaconda3/lib/python3.6/site-packages/mnist/samples/')
Train_images, Train_labels = mndata.load_training()
Test_images, Test_labels = mndata.load_testing()

#convert to numpy array
Train_images = np.array(Train_images)
Test_images = np.array(Test_images)
Train_labels = np.array(Train_labels)
Test_labels = np.array(Test_labels)

# input image dimensions
img_rows, img_cols = 28, 28

# the data, split between train and test sets
if K.image_data_format() == 'channels_first':
    Train_imagesCNN = Train_images.reshape(Train_images.shape[0], 1, img_rows, img_cols)
    Test_imagesCNN = Test_images.reshape(Test_images.shape[0], 1, img_rows, img_cols)    
    input_shapeCNN = (1, img_rows, img_cols)
else:
    Train_imagesCNN = Train_images.reshape(Train_images.shape[0],  img_rows, img_cols,1)
    Test_imagesCNN = Test_images.reshape(Test_images.shape[0], img_rows, img_cols,1)    
    input_shapeCNN = (img_rows, img_cols,1)
Train_imagesCNN = Train_imagesCNN.astype('float32')
Test_imagesCNN = Test_imagesCNN.astype('float32')
Train_imagesCNN /= 255
Test_imagesCNN /= 255 

Train_images = Train_images.astype('float32')
Test_images = Test_images.astype('float32')

Train_images /= 255
Test_images /= 255 
#convert labels to categorical vectors
num_classes = 10
Train_labels = keras.utils.to_categorical(Train_labels,num_classes)
Test_labels = keras.utils.to_categorical(Test_labels,num_classes)

#create keras model
#CNN Model
model1 = Sequential()

model1.add(Conv2D(32, kernel_size=(3, 3),
                 activation='relu',
                 input_shape=input_shapeCNN))
model1.add(Conv2D(64, (3, 3), activation='relu'))
model1.add(MaxPooling2D(pool_size=(2, 2)))
model1.add(Dropout(0.25))
model1.add(Flatten())
model1.add(Dense(128, activation='relu'))
model1.add(Dropout(0.5))
model1.add(Dense(num_classes, activation='softmax'))

tbCallBack1 = TensorBoard(log_dir='./Graph1', histogram_freq=0, write_graph=True, write_images=True)
tbCallBack1.set_model(model1)
model1.summary()
plot_model(model1, to_file='model.png',show_layer_names=True, show_shapes=True)

model1.compile(loss='categorical_crossentropy', optimizer=optimizers.Adadelta(),metrics=['accuracy'])
history = model1.fit(Train_imagesCNN, Train_labels,
	batch_size=128,
	epochs=50,
	verbose=1,
	validation_data=(Test_imagesCNN,Test_labels),
	callbacks=[tbCallBack1])
score1 = model1.evaluate(Test_imagesCNN, Test_labels,verbose=0)
print('Test loss:', score1[0])
print('Test accuracy: ', score1[1])
"""OTHER NNN"""
#Simple Feed Forward Model
model2 = Sequential()

model2.add(Dense(512,activation='relu',input_shape=(784,)))
model2.add(Dropout(0.2))
model2.add(Dense(512,activation='tanh'))
model2.add(Dropout(0.3))
model2.add(Dense(512,activation='relu'))
model2.add(Dropout(0.2))
model2.add(Dense(num_classes,activation='softmax'))

tbCallBack2 = TensorBoard(log_dir='./Graph2', histogram_freq=0, write_graph=True, write_images=True)
tbCallBack2.set_model(model2)
model2.summary()
plot_model(model2, to_file='model.png',show_layer_names=True, show_shapes=True)
sgd = optimizers.SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
model2.compile(loss='categorical_crossentropy', optimizer=sgd,metrics=['accuracy']) 
history = model2.fit(Train_images, Train_labels,
	batch_size=128,
	epochs=50,
	verbose=1,
	validation_data=(Test_images,Test_labels),
	callbacks=[tbCallBack2])
score2 = model2.evaluate(Test_images, Test_labels,verbose=0)
print('Test loss:', score2[0])
print('Test accuracy: ', score2[1])