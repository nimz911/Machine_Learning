# -*- coding: utf-8 -*-
"""
Created on Tue Apr  3 11:58:10 2018
@author: nimrod

kaggle titanic challenge
"""
import pandas as pd
import numpy as np
#%%
df = pd.read_csv('train.csv')
df = df.fillna(df.mean())
df = df[pd.notnull(df['Embarked'])] # Remove 2 'Embarked' NAN rows

#%%
df['Embarked_int'] = df['Embarked'].astype('category').cat.codes
df['Sex_int'] = df['Sex'].astype('category').cat.codes
df['Mr'] = df['Name'].str.contains(' Mr. ').astype('category').cat.codes
df['Mrs'] = df['Name'].str.contains(' Mrs. ').astype('category').cat.codes
df['Miss'] = df['Name'].str.contains(' Miss. ').astype('category').cat.codes
df['Ms'] = df['Name'].str.contains(' Ms. ').astype('category').cat.codes
df['Mr'] = df['Mr'] * df.columns.get_loc("Mr")
df['Mrs'] = df['Mrs'] * df.columns.get_loc("Mrs")
df['Miss'] = df['Miss'] * df.columns.get_loc("Miss")
df['Ms'] = df['Ms'] * df.columns.get_loc("Ms")
df['title'] = df['Mr'] + df['Mrs'] + df['Ms'] + df['Miss']
df['title'] = df['title'].astype('category').cat.codes


X = df[['Age','Sex_int','SibSp','Pclass','Parch','Embarked_int','Fare','title']].values
y = df['Survived'].values
#%%

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 0)

#%%
# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
x_train = sc.fit_transform(x_train)
x_test = sc.transform(x_test)

#%%
# Fitting Logistic Regression to the Training set
from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression(random_state = 0)
classifier.fit(x_train, y_train)

#%%
# Applying Grid Search to find the best model and the best parameters
from sklearn.model_selection import GridSearchCV
#C = [0.047,0.048,0.049,0.05,0.051]
C = np.arange(0.045,0.06,0.00001)
penalty1 = ['l1']
penalty2 = ['l2']
multi_class = ['ovr']
solver1 = ['liblinear']
solver2 = ['lbfgs','sag','newton-cg']
#tol = [0.00001,0.00005,0.001]
max_iter = [150,250,500]
parameters = [{'C': C, 'penalty':penalty1, 'solver':solver1, 'multi_class':multi_class, 'max_iter':max_iter},
              {'C':C, 'penalty':penalty2,'solver':solver2, 'multi_class':multi_class, 'max_iter':max_iter}]
grid_search = GridSearchCV(estimator = classifier,
                           param_grid = parameters,
                           scoring = 'accuracy',
                           cv = 20,
                           n_jobs = 4)
grid_search = grid_search.fit(x_train, y_train)
best_accuracy = grid_search.best_score_
best_parameters = grid_search.best_params_

#%%
# Read data
test = pd.read_csv('test.csv')
train = pd.read_csv('train.csv')
train = train.fillna(train.mean())

#%%
# Check for missing data in test features
round((test.isnull().sum() / test.shape[0]) * 100,4)
#%%
# Check for missing data in train features
round((train.isnull().sum() / train.shape[0]) * 100,4)
#%%
# Taking care of missing data
#test['Age'].fillna((test['Age'].mean()), inplace=True)
#train['Age'].fillna((train['Age'].mean()), inplace=True)
test = test.fillna(test.mean())
train = train.fillna(train.mean())
train = train[pd.notnull(train['Embarked'])] # Remove 2 'Embarked' NAN rows

#%%
# Encoding categorical data
test['Embarked_int'] = test['Embarked'].astype('category').cat.codes
train['Embarked_int'] = train['Embarked'].astype('category').cat.codes
test['Sex_int'] = test['Sex'].astype('category').cat.codes
train['Sex_int'] = train['Sex'].astype('category').cat.codes

test['Mr'] = test['Name'].str.contains(' Mr. ').astype('category').cat.codes
test['Mrs'] = test['Name'].str.contains(' Mrs. ').astype('category').cat.codes
test['Miss'] = test['Name'].str.contains(' Miss. ').astype('category').cat.codes
test['Ms'] = test['Name'].str.contains(' Ms. ').astype('category').cat.codes
test['Mr'] = test['Mr'] * df.columns.get_loc("Mr")
test['Mrs'] = test['Mrs'] * df.columns.get_loc("Mrs")
test['Miss'] = test['Miss'] * df.columns.get_loc("Miss")
test['Ms'] = test['Ms'] * df.columns.get_loc("Ms")
test['title'] = test['Mr'] + test['Mrs'] + test['Ms'] + test['Miss']
test['title'] = test['title'].astype('category').cat.codes

train['Mr'] = train['Name'].str.contains(' Mr. ').astype('category').cat.codes
train['Mrs'] = train['Name'].str.contains(' Mrs. ').astype('category').cat.codes
train['Miss'] = train['Name'].str.contains(' Miss. ').astype('category').cat.codes
train['Ms'] = train['Name'].str.contains(' Ms. ').astype('category').cat.codes
train['Mr'] = train['Mr'] * df.columns.get_loc("Mr")
train['Mrs'] = train['Mrs'] * df.columns.get_loc("Mrs")
train['Miss'] = train['Miss'] * df.columns.get_loc("Miss")
train['Ms'] = train['Ms'] * df.columns.get_loc("Ms")
train['title'] = train['Mr'] + train['Mrs'] + train['Ms'] + train['Miss']
train['title'] = train['title'].astype('category').cat.codes
#%%
x_test = test[['Age','Sex_int','SibSp','Pclass','Parch','Embarked_int','Fare']].values
x_train = train[['Age','Sex_int','SibSp','Pclass','Parch','Embarked_int','Fare']].values
y_train = train['Survived'].values

#%%
# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
x_train = sc.fit_transform(x_train)
x_test = sc.transform(x_test)

#%%
# Fitting Logistic Regression to the Training set
from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression(**grid_search.best_params_, random_state = 0)
classifier.fit(x_train, y_train)


#%%
# Predicting the Test set results
y_pred = classifier.predict(x_test)
#%%

predicion = pd.DataFrame({'PassengerId':test['PassengerId'], 'Survived':y_pred},index=None)
predicion.to_csv('submission.csv', index=False)
#%%


# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix, accuracy_score, r2_score, classification_report
cm = confusion_matrix(y_test, y_pred)
accuracy = accuracy_score(y_test, y_pred, normalize=True, sample_weight=None)
r2 = r2_score(y_test, y_pred)
classification_report = classification_report(y_test,y_pred,digits=3)
print('accuracy: '+str(accuracy)+'\nr2: '+str(r2)+'\nclassification_report:\n\n'+classification_report)

#%%
import matplotlib.pyplot as plt
def plot_confusion_matrix(cm, title='Confusion matrix', cmap=plt.cm.Oranges):
    import itertools
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    ax = plt.gca()
    ax.set_xticklabels((ax.get_xticks() +1).astype(str))

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    

print(cm)
fig, ax = plt.subplots()
plot_confusion_matrix(cm)

plt.show()

#%%

