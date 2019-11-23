# -*- coding: utf-8 -*-
"""
Created on Thu Nov 21 00:56:14 2019

@author: hp
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pylab import rcParams
rcParams['figure.figsize'] = 10,10

col_names = ['Time', 'h_train', 'b_train']
data = pd.read_csv("Q2train.csv", header=None, names=col_names) 
data.head()
feature_cols = ['Time', 'h_train', 'b_train']
x = data[feature_cols] # Features
htrain=x['h_train']
btrain=x['b_train']
T = x['Time'] # Target variable
fig = plt.figure() 
ax1 = fig.add_subplot(111)
ax1.scatter(htrain,btrain,s=30,color='black',marker="*",label='measured position')
plt.plot(htrain,btrain,linestyle='dashed')
plt.title('Object Measurement Sequence')
plt.xlabel('Object Measurement longitude')
plt.ylabel('Object Measurement latitude')
plt.legend(loc='upper left');
plt.show()
#Initial estimate x[0]
data_train= np.loadtxt('Q2train.csv',delimiter=',')
data_test= np.loadtxt('Q2test.csv',delimiter=',')
#state=[data_train[0,1],2,2,data_train[0,2],2,2]



def crossval(data_train,data_test,k,s):
    #Input Matricies
    state=[0,0,0,0,0,0]
    xi=[htrain[0],2,2,btrain[0],2,2]
    A=[[1,2,2,0,0,0],[0,1,2,0,0,0],[0,0,1,0,0,0],[0,0,0,1,2,2],[0,0,0,0,1,2],[0,0,0,0,0,1]]
    A=np.array(A)
    C=[[1,0,0,0,0,0],[0,0,0,1,0,0]]
    C=np.array(C)
    Pi=[[0.1,0,0,0,0,0],[0,0.1,0,0,0,0],[0,0,0,0,0,0],[0,0,0,0.1,0,0],[0,0,0,0,0.1,0],[0,0,0,0,0,0]]
    Pi=np.array(Pi)
    Q= k*np.eye(6)
    R=s*np.eye(2)
    Pred_train=[]
    
    for i in range(100):
        Xnp= np.dot(A,state) #X time update
        Pnp=np.dot(np.dot(A,Pi),A.T)+Q #P time update
        Kdot=np.linalg.inv((np.dot(np.dot(C,Pnp),C.T)+R))
        K=np.dot(np.dot(Pnp,C.T),Kdot) #Calculate Kalman Gain
        MEA=[data_train[i,1],2,2,data_train[i,2],2,2]
        yt=np.dot(C,MEA) #Noisy measurements from Q1train.csv
        Xp= Xnp + np.dot(K,(yt-np.dot(C,Xnp))) #X estimate (Measurement Update)
        Pred_train.append(Xp) #Store Estimate
        P=Pnp-np.dot(K,np.dot(C,Pnp)) #P measurement update
        state=Xp #Update new previous state
        Pi=P #update initial P
    
    Pred_test=[]
    CV=np.zeros((10,10))
    for i in range(100):
        Xnp= np.dot(A,state)
        Pnp=np.dot(np.dot(A,Pi),A.T)+Q
        Kdot=np.linalg.inv((np.dot(np.dot(C,Pnp),C.T)+R))
        K=np.dot(np.dot(Pnp,C.T),Kdot)
        MEA=[data_test[i,1],2,2,data_test[i,2],2,2]
        yt=np.dot(C,MEA)
        Xp= Xnp + np.dot(K,(yt-np.dot(C,Xnp)))
        Pred_test.append(Xp)
        P=Pnp-np.dot(K,np.dot(C,Pnp))
        state=Xp
        Pi=P 
    Crossval=0
    for i in range(100):
        MEA=[data_test[i,1],2,2,data_test[i,2],2,2]
        Crossval=Crossval+(abs(Pred_test[i][0]-MEA[0])+(abs(Pred_test[i][3]-MEA[3])))**2
    Crossval_score=Crossval/100
    return [Crossval_score,Pred_train,Pred_test]

scores=[]

for k in range(1,10):
    for s in range(1,10):
        score,Pred_train,Pred_test=crossval(data_train,data_test,k,s)
        scores.append([k,s,score])
        
scores = np.array(scores)
min_score=np.argmin(scores, axis=0)[2]
print('The optimal pair (K,S) which gives the minimum cross validation metric is:')
print(scores[min_score,0])
print(scores[min_score,1])
print('The value of minimum cross validation metric is:')
print(scores[min_score,2])

xlist = np.linspace(1,9,9)
ylist = np.linspace(1,9,9)
X, Y = np.meshgrid(xlist, ylist)
Z = scores[:, 2].reshape(9, 9)
#Contour Plot 
plt.contour(X,Y,Z)
plt.plot(9,1,'o',ms=20)
plt.xlabel('K Values')
plt.ylabel('S Values')
plt.show()
score,Pred_train,Pred_test=crossval(data_train,data_test,9,1)

Predtrain=np.zeros([100,2])
Predtest=np.zeros([100,2])
C=[[1,0,0,0,0,0],[0,0,0,1,0,0]]
C=np.array(C)

for i in range(100):
    Predtrain[i,:]=np.squeeze(np.dot(C,np.expand_dims(Pred_train[i],1)))
    Predtest[i,:]=np.squeeze(np.dot(C,np.expand_dims(Pred_test[i],1)))
    
# Scatter Plot of Train and Test data along with plot of estmated train and test data

plt.scatter(data_train[:,1],data_train[:,2],s=30,color='black',marker="*",label='train data')
plt.scatter(data_test[:,1],data_test[:,2],s=30,color='red',marker="o",label='test data')
plt.plot(Predtrain[:,0],Predtrain[:,1],linestyle='dashed',color='black',label='train estimate')
plt.plot(Predtest[:,0],Predtest[:,1],linestyle='dashed',color='red',label='test estimate')
plt.title('Object Measurement Sequence')
plt.xlabel('Object Measurement longitude')
plt.ylabel('Object Measurement latitude')
plt.legend(loc='upper left');
plt.show()