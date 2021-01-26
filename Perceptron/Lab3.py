# -*- coding: utf-8 -*-
"""
Created on Wed Mar 25 23:49:51 2020

@author: Erdinc
"""
"""
plt.scatter((data.iloc[:4,0]), (data.iloc[:4,2]), marker='o', label='A')
plt.scatter((data.iloc[5:,0]), (data.iloc[5:,2]), marker='x', label='B')
plt.xlabel('petal length')
plt.ylabel('sepal length')
plt.legend()
plt.show()

"""
import numpy as np # linear algebra
import pandas as pd
import matplotlib.pyplot as plt

#%%
data = pd.read_csv("data.csv", sep=",")

#%%
def stepFunction(z):
    if z>0:
        y_pred=1
    elif z<=0:
        y_pred=0
    return y_pred

#%%
def initializeWeightNBias(size):
    weight=np.random.rand(1,size)
    bias=1
    return weight,bias

#%%
def perceptronStep(xdata,weight,bias):
    lr=0.1
    error=0
    y = data.iloc[:,-1]
    x = data.iloc[:,:-1]
    x_nd=x.to_numpy()
    for i in range(len(xdata)):
        total=(np.dot(x_nd[i,:],weight.T)+bias)        
        y_pred=stepFunction(total)    
        if (y[i]-y_pred==1):
            weight+=(x_nd[i]*lr).T
            bias+=lr
            error+=1
        elif(y[i]-y_pred==-1):
            weight-=(x_nd[i]*lr).T
            bias-=lr
            error+=1
    return weight,bias,error
               
#%%
def trainPerceptron(data):
    index=[]
    cost=[]
    x = data.iloc[:,:-1]
    weight,bias=initializeWeightNBias(x.shape[1])  
    for iterator in range(10):
        index.append(iterator)
        weight,bias,error = perceptronStep(data,weight,bias)
        cost.append(error)
    return index,cost

#%%
xindex,ycost=trainPerceptron(data)
plt.plot(xindex,ycost)
plt.xticks(xindex,rotation="vertical")
plt.xlabel("Number of Iteration")
plt.ylabel("Cost")
plt.show()