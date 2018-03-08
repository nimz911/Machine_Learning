# -*- coding: utf-8 -*-
"""
Created on Thu Mar  8 11:59:16 2018

@author: nimrod
"""

import sklearn
import matplotlib.pyplot as plt
from sklearn import cross_validation
from sklearn.svm import SVC
import pandas as pd
import time
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report,confusion_matrix
from sklearn.neural_network import MLPClassifier
import numpy as np
#%%
# Data preparation
df = pd.read_csv('d:/data/machine learning/wpbc.csv', header=None) # R=1 / N=0
df.loc[df[34] == '?', 34] = 0  # repalce '?' in last column with 0
df[34] = df[34].astype('float64') # set data type as float
array = df.values # create array from DataFrame values
X = pd.DataFrame(array[:,3:35]) # create DataFrame of PREDICTORS
y = pd.Series(array[:,1]) # create DataFrame of LABELS

# split the data to train & test 
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 300)

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)
#%%
# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# Fitting Kernel SVM to the Training set
from sklearn.svm import SVC
classifier = SVC(kernel = 'sigmoid', random_state = 0, C=0.1,gamma=0.00001)
classifier.fit(X_train, y_train)

# Predicting the Test set results
y_pred = classifier.predict(X_test)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
classifier.score(X_test,y_test)
#%%
# create models list with different kernels & parameters

models = [] 
#cache_size = range(100,300,100)
#degree = range(1,10,2)
#gamma = np.arange(0.0001, 0.5, 0.001)
#C = np.arange(0.1, 1.0, 0.01)
C = []
gamma = []
for i in range(0,13):
    C.append(10**i)
    gamma.append(10**(-i))
    

for j in range(0,len(C)):
    for k in range(0,len(gamma)):
        models.append(('SVM_rbf', SVC(gamma=gamma[k],C=C[j])))
        models.append(('SVM_sigmoid', SVC(kernel='sigmoid', gamma=gamma[k],C=C[j])))


# evaluate each model in turn 
results = []
names = []
scoring = 'accuracy'
#%%

# prepare configuration for cross validation test harness
num_instances = len(X)
num_folds = range(3,12,2)
seed = 7
#%%
summary = []
msg = []
model_cache = []
model_degree = []
model_kernel = []
model_gamma = []
model_C = []

startTime = time.time()
for name, model in models:
    for i in range(0,len(num_folds)):
        start_time = time.time()
        kfold = cross_validation.KFold(n=num_instances, n_folds=num_folds[i], random_state=seed) 
        cv_results = cross_validation.cross_val_score(model, X, y, cv=kfold, scoring=scoring)
        results.append(cv_results) 
        names.append(name)
        model_cache.append(model.cache_size)
        model_degree.append(model.degree)
        model_kernel.append(model.kernel)
        model_gamma.append(model.gamma)
        model_C.append(model.C)
        end_time = time.time()
        total_time = end_time - start_time
        msg = "model name: %s, n_fold: %s, results mean: %f, results std: %f, time: %f" % (name,num_folds[i], cv_results.max(), 
                                                                                           cv_results.std(), total_time)
        summary.append(msg)

endTime = time.time()
totalTime = endTime - startTime
print(startTime, endTime,totalTime)
#%%

# create results summary DataFrame
newlist=[]
for item in summary:
    newlist.append(item.split(','))
df = pd.DataFrame.from_records(newlist)

name_df = df[0].str.split(': ', 1, expand=True)
fold_df = df[1].str.split(': ', 1, expand=True)
mean_df = df[2].str.split(': ', 1, expand=True)
std_df = df[3].str.split(': ', 1, expand=True)
time_df = df[4].str.split(': ', 1, expand=True)

summary_df = pd.DataFrame({'model_name': name_df[1],'n_fold': fold_df[1], 'results_mean': mean_df[1],
              'results_std':std_df[1], 'run_time':time_df[1],'model_cache':model_cache, 'model_degree':model_degree,
                           'model_kernel':model_kernel, 'model_gamma':model_gamma})
    
#%%
# create models list with different kernels & parameters
solver = ['lbfgs', 'sgd', 'adam']
alpha  = [0.1,0.05,0.001,0.5,0.005]
learning_rate = ['constant', 'invscaling', 'adaptive']

MLP_models = []
for i in range(0,len(solver)):
    for j in range(0,len(alpha)):
        for k in range(0,len(learning_rate)):
                 MLP_models.append(MLPClassifier(solver=solver[i], alpha=alpha[j],
                                                 learning_rate=learning_rate[k],hidden_layer_sizes=(1000,500), random_state=100))


#%%
models = []
score = []
kernels = ['sigmoid','linear','rbf']
for C in np.arange(0.1, 10, 0.01):
    for kernel in kernels:
        clf = SVC(C=C,kernel=kernel,decision_function_shape='ovo')
        clf.fit(X_train,y_train)
        score.append(clf.score(X_test,y_test))
        print clf.kernel, clf.coef0, clf.C, round(clf.score(X_test,y_test),5)
