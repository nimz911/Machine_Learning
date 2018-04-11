# -*- coding: utf-8 -*-
"""
Created on Wed Apr 11 13:11:40 2018

@author: nimrod
"""

# Importing  packages
from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense
from keras import backend as K
from keras.layers import Dropout
from keras.constraints import maxnorm
from keras.optimizers import SGD
import keras
import pandas as pd
K.set_image_dim_ordering('tf')
import numpy as np


#%%

test = pd.read_csv('test.csv')
train = pd.read_csv('train.csv')

X_train = train.loc[:, 'pixel0':'pixel783'].values
X_test = test.loc[:, 'pixel0':'pixel783'].values
y_train = train['label'].values

X_train = X_train.reshape(-1, 28,28, 1)
X_test = X_test.reshape(-1, 28,28, 1)

X_train = X_train / 255.0
X_test = X_test / 255.0


from keras.utils.np_utils import to_categorical
y_train_binary = to_categorical(y_train)
#y_test_binary = to_categorical(y_test)

from sklearn.model_selection import train_test_split
X_train,X_val,y_train,y_val = train_test_split(X_train, y_train_binary, test_size=0.2, random_state=13)

#%%



#%%
from keras.layers.advanced_activations import LeakyReLU
'''
classifier = Sequential()
classifier.add(Conv2D(32, kernel_size=(3, 3),activation='linear',input_shape=(28,28,1),padding='same'))
classifier.add(LeakyReLU(alpha=0.1))
classifier.add(MaxPooling2D((2, 2),padding='same'))
classifier.add(Conv2D(64, (3, 3), activation='linear',padding='same'))
classifier.add(LeakyReLU(alpha=0.1))
classifier.add(MaxPooling2D(pool_size=(2, 2),padding='same'))
classifier.add(Conv2D(128, (3, 3), activation='linear',padding='same'))
classifier.add(LeakyReLU(alpha=0.1))                  
classifier.add(MaxPooling2D(pool_size=(2, 2),padding='same'))
classifier.add(Flatten())
classifier.add(Dense(128, activation='linear'))
classifier.add(LeakyReLU(alpha=0.1))                  
classifier.add(Dense(10, activation='softmax'))
'''
epochs = 20
batch_size = 64
lrate = 0.0003

classifier = Sequential()
classifier.add(Conv2D(64, kernel_size=(3, 3),activation='linear',padding='same',input_shape=(28,28,1)))
classifier.add(LeakyReLU(alpha=0.1))
classifier.add(MaxPooling2D((2, 2),padding='same'))
classifier.add(Dropout(0.3))

classifier.add(Conv2D(128, (3, 3), activation='linear',padding='same'))
classifier.add(LeakyReLU(alpha=0.1))
classifier.add(MaxPooling2D(pool_size=(2, 2),padding='same'))
classifier.add(Dropout(0.3))

classifier.add(Conv2D(256, (3, 3), activation='linear',padding='same'))
classifier.add(LeakyReLU(alpha=0.1))                  
classifier.add(MaxPooling2D(pool_size=(2, 2),padding='same'))
classifier.add(Dropout(0.45))
classifier.add(Flatten())

classifier.add(Dense(256, activation='linear'))
classifier.add(LeakyReLU(alpha=0.1))           
classifier.add(Dropout(0.35))

classifier.add(Dense(10, activation='softmax'))

decay = lrate/epochs
adam = keras.optimizers.Adam()

#%%
classifier.compile(loss=keras.losses.categorical_crossentropy,optimizer=adam,metrics=['accuracy'])

#%%
model_train = classifier.fit(X_train, y_train, batch_size=batch_size,epochs=epochs,verbose=1,validation_data=(X_val, y_val))
#%%
test_eval = classifier.evaluate(X_val, y_val, verbose=0)
#%%
print('Test loss:', round(test_eval[0],5))
print('Test accuracy:', round(test_eval[1],5))
#%%
predicted_classes = classifier.predict(X_test)
y_pred = np.argmax(predicted_classes, axis=1)
pred_df = pd.DataFrame({'Label':y_pred}, index=None)
pred_df.index.name = 'ImageID'
pred_df.index = pred_df.index + 1
pred_df.to_csv('submission.csv')
