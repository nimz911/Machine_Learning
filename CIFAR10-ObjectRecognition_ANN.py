# -*- coding: utf-8 -*-

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import time
from datetime import datetime
import os
import scipy
from sklearn.metrics import classification_report,confusion_matrix
from sklearn.model_selection import train_test_split
import keras
from keras.models import Sequential
from keras.layers import Dense
from sklearn.utils import shuffle

#%%
#create txt file with all data
file_list = []
print ('start time: ' + str(datetime.today()))
start_time = time.time()
files = os.listdir('d:/data/machine learning/hw2/bmp_30k/')
array = open('d:/data/machine learning/hw2/bmp_pixels.txt','w')
for file in files:
    x = scipy.misc.imread('d:/data/machine learning/hw2/bmp_30k/' + file)
    flatten_x = x.flatten()
    array.write(str(flatten_x.tolist()) + '\n')
    file_list.append(file)
array.close()

#%%
# Data preparation - from csv generated from bmp_pixels.txt
data = []
file_list = []
start_time = []
end_time = []
total_time = []

print ('start time: ' + str(datetime.today()))
start_time = time.time()
files = os.listdir('c:/data/ML/hw2/bmp_30k/')
array = open('c:/data/ML/hw2/bmp_pixels.txt','w')
for file in files:
    x = scipy.misc.imread('c:/data/ML/hw2/bmp_30k/' + file)
    flatten_x = x.flatten()
    array.write(str(flatten_x.tolist()) + '\n')
    file_list.append(file)
array.close()

files = open('c:/data/ML/hw2/files_list.txt','w')
files.write(str(file_list))
files.close()

#%%

start_time = []
end_time = []
total_time = []

print ('start time: ' + str(datetime.today()))
start_time = time.time()

df = pd.read_csv('D:/data/machine learning/HW2/pics_30k - reduced.csv', header=None)
X = df.iloc[:,2:3074].values.astype(float) # create DataFrame of PREDICTORS
#y = df['class'].values.astype(int) # create DataFrame of LABELS
y = pd.get_dummies(df[1]).values.astype(float) # create matrix of dummies
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 300)
y_test = np.argmax(y_test,axis=1)
y_test_dummies = pd.get_dummies(y_test).values.astype(float) # create matrix of dummies

# Feature normalize inputs to 0.0-1.0
X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
X_train = X_train / 255.0
X_test = X_test / 255.0

'''
# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)
'''

end_time = time.time()
total_time = end_time - start_time
print ("end time: " + str(datetime.today()))
print ("total run time: " + str(total_time))


#%%

batch_size = [250,500]
epochs = [25]
units = [50,1500]
decay = 0.001 / 100
sgd = keras.optimizers.SGD(lr=0.0001, momentum=0.95, decay=decay, nesterov=False)
Adam = keras.optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
activations1 = ['relu']
activations2 = 'softmax'

f = open('D:/data/machine learning/HW2/classification_report.txt','a',1)

for unit in units:
    for size in batch_size:
        for n_epochs in epochs:
            for activation1 in activations1:
                start_time = []
                end_time = []
                total_time = []
                start_time = time.time()
                f.write('start time: ' + str(datetime.today())+'\n'+'\n')
                f.write('DataFrame shape: '+ str(df.shape) + '\n')
                f.write('test size: '+str(len(y_test)) + '\n')
                f.write('train size: '+ str(len(y_train)) + '\n')
                f.write('classifier parameters:'+'\n')
                f.write('\t'+'# 1st activation: '+str(activation1)+'\n')
                f.write('\t'+'# 2nd activation: softmax'+'\n')
                f.write('\t'+'# optimizer: SGD - '+'\t'+str(sgd.get_config())+'\n')
                f.write('\t'+'# batch_size: '+str(size)+'\n')
                f.write('\t'+'# number of epochs: '+str(n_epochs)+'\n')
                f.write('\t'+'# hidden layer size: '+str(unit)+'\n'+'\n')
                f.write(str('-'*84)+'\n')
                classifier = Sequential()
                classifier.add(Dense(units= unit, kernel_initializer = "glorot_uniform", activation = activation1, input_dim = 3072))
                classifier.add(Dense(units = 9, kernel_initializer = "glorot_uniform", activation = 'softmax'))
                classifier.compile(optimizer=sgd, loss = "categorical_crossentropy", metrics = ["accuracy"])
                classifier.fit(X_train, y_train, validation_data=(X_test, y_test_dummies), batch_size = size, epochs = n_epochs)
                y_pred = classifier.predict(X_test)
                y_pred = np.argmax(y_pred,axis=1)
                print(classification_report(y_test,y_pred,digits=3))
                f.write('classification_report: '+'\n'+str(classification_report(y_test,y_pred,digits=3))+str('-'*84)+'\n'+'\n')
                cm = confusion_matrix(y_test, y_pred)
                end_time = time.time()
                total_time = end_time - start_time
                f.write('confusion matrix:'+'\n'+str(cm)+'\n'+str('-'*84)+'\n')
                scores = classifier.evaluate(X_test, y_test_dummies, verbose=0)
                print("Accuracy: %.2f%%" % (scores[1]*100))
                f.write('Accuracy: '+str(scores[1]*100)+'%'+'\n')
                f.write('total run time: '+str(total_time)+'\n'+'end time: ' + str(datetime.today())+'\n'+str('#'*125)+'\n'+'\n')
                time.sleep(2)
                
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

    
  
    
