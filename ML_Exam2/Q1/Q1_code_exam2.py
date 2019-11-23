# -*- coding: utf-8 -*-
"""
Created on Mon Nov 18 22:14:27 2019

@author: hp
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix
from sklearn.tree import export_graphviz
from sklearn.externals.six import StringIO
from IPython.display import Image
from pydotplus import graph_from_dot_data
from sklearn.metrics import accuracy_score
from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.model_selection import train_test_split
import statistics
from pylab import rcParams
rcParams['figure.figsize'] = 10,10
# =============================================================================
# import os
# os.environ['PATH'] = os.environ['PATH']+';'+os.environ['CONDA_PREFIX']+r"\Library\bin\graphviz"
# =============================================================================
def plotdecision_boundary(x1_dec,x2_dec,y,x1_1,x2_1,x1_2,x2_2,clf1):
    x_min, x_max = x1_dec.min() - 1, x1_dec.max() + 1
    y_min, y_max = x2_dec.min() - 1, x2_dec.max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.1),np.arange(y_min, y_max, 0.1))

    Z = clf1.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    fig = plt.figure()
    ax1 = fig.add_subplot(111)
    ax1.scatter(x1_1,x2_1,s=10,color='black',marker="x",label='Class 1')
    ax1.scatter(x1_2,x2_2,s=10,c='r',marker="o",label='Class -1')
    ax1.contourf(xx, yy, Z, alpha=0.4,label='Decision Boundary')
    plt.title('scatter plot of class +1 & class -1')
    plt.xlabel('Feature value x1')
    plt.ylabel('Feature value x2')
    plt.legend(loc='upper left');
    plt.show()
    return;
    
col_names = ['x1', 'x2', 'label']
data = pd.read_csv("Q1.csv", header=None, names=col_names) 
data.head()
feature_cols = ['x1', 'x2']
x = data[feature_cols] # Features
x1_dec=x['x1']
x2_dec=x['x2']
y = data['label'] # Target variable
indices=np.where(y==1)
X1=x.loc[indices]
indices=np.where(y==-1)
X2=x.loc[indices]
x1_1=X1['x1']
x2_1=X1['x2']
x1_2=X2['x1']
x2_2=X2['x2']

X_test=x[0:100]
X_train=x[100:1000]
y_test=y[0:100]
y_train=y[100:1000]
clf = DecisionTreeClassifier(criterion="entropy", max_depth=11,min_impurity_decrease =0.01)
clf = clf.fit(X_train,y_train)
y_pred = clf.predict(X_test)
print("Accuracy of ID3 classifier:",accuracy_score(y_test, y_pred))
dot_data = StringIO()
export_graphviz(clf, out_file=dot_data,  
                filled=True, rounded=True,
                special_characters=True, feature_names = feature_cols,class_names=['1','-1'])
graph = graph_from_dot_data(dot_data.getvalue())  
graph.write_png('DecisionTree.png')
Image(graph.create_png())
confusion_ID3=confusion_matrix(y_test, y_pred)
plotdecision_boundary(x1_dec,x2_dec,y,x1_1,x2_1,x1_2,x2_2,clf)

clfbagg = BaggingClassifier(base_estimator=clf,n_estimators=7,bootstrap=True)
clfbagg = clfbagg.fit(X_train,y_train)
y_bagg = np.zeros((100,7))
for i in range(7):
    clfbagg.estimators_[i]=clfbagg.estimators_[i].fit(X_train,y_train)
    y_pred_bagg=clfbagg.estimators_[i].predict(X_test)
    y_bagg[:,i]=y_pred_bagg
print(y_bagg)
y_bagging_output=np.zeros((100,1))
for i in range(100):
    y_bagging_output[i,0]=statistics.mode(y_bagg[i,:])
print(y_bagging_output)
print("Accuracy of Bagging classifier:",accuracy_score(y_test, y_bagging_output))  
confusion_bagging=confusion_matrix(y_test, y_bagging_output)
plotdecision_boundary(x1_dec,x2_dec,y,x1_1,x2_1,x1_2,x2_2,clfbagg)
clboost = AdaBoostClassifier(base_estimator=clf, n_estimators=7, learning_rate=1.0, algorithm='SAMME', random_state=None)
clboost = clboost.fit(X_train,y_train)
y_predboost=clboost.predict(X_test)
confusion_boost=confusion_matrix(y_test, y_predboost)
print("Accuracy of Adaboost classifier:",accuracy_score(y_test, y_predboost))
plotdecision_boundary(x1_dec,x2_dec,y,x1_1,x2_1,x1_2,x2_2,clboost)
print(clboost.estimator_weights_)