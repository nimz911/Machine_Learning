# -*- coding: utf-8 -*-

# Importing the Keras libraries and packages
from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense
from keras.layers import BatchNormalization
from keras import backend as K
from keras.layers import Dropout
from keras.constraints import maxnorm
from keras.optimizers import SGD
import keras
import time
from datetime import datetime
import pandas as pd
import os
import scipy
K.set_image_dim_ordering('tf')
from sklearn.metrics import classification_report,confusion_matrix
import numpy as np
import matplotlib.pyplot as plt

#%%

start_time = []
end_time = []
total_time = []
start_time = time.time()

train_files = os.listdir('d:/data/machine learning/hw2/reduced/trainSet/')
X_train = []
x = []
for file in train_files:
    x = scipy.misc.imread('d:/data/machine learning/hw2/reduced/trainSet/' + file)
    X_train.append(x)
X_train = np.asarray(X_train).reshape((30000, 3, 32, 32))


test_files = os.listdir('d:/data/machine learning/hw2/reduced/testSet/')
X_test = []
x = []
for file in test_files:
    x = scipy.misc.imread('d:/data/machine learning/hw2/reduced/testSet/' + file)
    X_test.append(x)
X_test = np.asarray(X_test).reshape((10000, 3, 32, 32))

X_train = X_train / 255.0
X_test = X_test / 255.0

test_files = pd.DataFrame({'test_files': test_files})
y_test = test_files['test_files'].str.split('_', 3, expand=True)[1].astype(int)
y_test = np.array(y_test).reshape((10000,1))

train_files = pd.DataFrame({'train_files': train_files})
y_train = train_files['train_files'].str.split('_', 3, expand=True)[2].astype(int)
y_train = np.array(y_train).reshape((30000,1))

from keras.utils.np_utils import to_categorical
y_train_binary = to_categorical(y_train)
y_test_binary = to_categorical(y_test)

end_time = time.time()
total_time = end_time - start_time
print ("end time: " + str(datetime.today()))
print ("total run time: " + str(total_time))


#%%
start_time = []
end_time = []
total_time = []
start_time = time.time()
f = open('D:/data/machine learning/HW2/CNN_report.txt','a',1)

epochs = 32
batch_size = 64
lrate = 0.003

f.write('start time: ' + str(datetime.today())+'\n'+'\n')
f.write('classifier parameters:'+'\n')
f.write('\t'+'# batch_size: '+str(batch_size)+'\n')
f.write('\t'+'# number of epochs: '+str(epochs)+'\n')
f.write('\t'+'# learning rate: '+str(lrate)+'\n'+'\n')
# Initialising the CNN
classifier = Sequential()
# Step 1 - Convolution
classifier.add(Conv2D(128, (2, 2), input_shape = (3, 32, 32), padding='same', activation = 'relu', kernel_constraint=maxnorm(3)))
#classifier.add(Conv2D(16, (2, 2), input_shape = (3, 32, 32), activation = 'relu', kernel_constraint=maxnorm(3)))
classifier.add(Dropout(0.2))
# Step 2 - Pooling
classifier.add(MaxPooling2D(pool_size = (2, 2), data_format="channels_first"))
classifier.add(BatchNormalization())
# Adding a second convolutional layer
classifier.add(Conv2D(128, (2, 2), data_format="channels_first", padding='same', activation = 'relu', kernel_constraint=maxnorm(3)))
classifier.add(MaxPooling2D(pool_size = (2, 2)))
classifier.add(BatchNormalization())
# Step 3 - Flattening
classifier.add(Flatten())
# Step 4 - Full connection
classifier.add(Dense(units = 256, activation = 'relu', kernel_constraint=maxnorm(3)))
classifier.add(Dropout(0.2))
classifier.add(Dense(units = 10, activation = 'softmax'))

# Compiling the CNN


decay = lrate/epochs
sgd = keras.optimizers.SGD(lr=lrate, momentum=0.9, decay=decay, nesterov=False)
classifier.compile(optimizer =sgd , loss = 'categorical_crossentropy', metrics = ['accuracy'])
classifier.fit(X_train, y_train_binary, batch_size = batch_size, epochs = epochs)
y_pred = classifier.predict(X_test)
y_pred = np.argmax(y_pred,1)
f.write('classification_report: '+'\n'+str(classification_report(y_test,y_pred,digits=3))+str('-'*84)+'\n'+'\n')
scores = classifier.evaluate(X_test, y_test_binary, verbose=0)
print("Accuracy: %.2f%%" % (scores[1]*100))
cm = confusion_matrix(y_test, y_pred)
f.write('confusion matrix:'+'\n'+str(cm)+'\n'+str('-'*84)+'\n'+'\n')
f.write('Accuracy: '+str(scores[1]*100)+'%'+'\n')
end_time = time.time()
total_time = end_time - start_time
f.write('total run time: '+str(total_time)+'\n'+'end time: ' + str(datetime.today())+'\n'+str('#'*125)+'\n'+'\n')
f.close()

#%%

fig_size = plt.rcParams["figure.figsize"]
fig_size[0] = 10
fig_size[1] = 7
plt.rcParams["figure.figsize"] = fig_size
plt.matshow(cm)
plt.title('Confusion matrix')
plt.colorbar()
plt.ylabel('True labels')
plt.xlabel('Predicted labels')
plt.show()


#%%
