# -*- coding: utf-8 -*-
"""
Created on Sat Apr  4 19:41:24 2020

@author: Erdinc
XOR
0 0 --> 0 
0 1 --> 1 
1 0 --> 1 
1 1 --> 0 

"""

import numpy as np
import matplotlib.pyplot as plt 

#%%Sigmoid : Activation Function
def sigmoid(i):
    return 1 / (1+np.exp(-i))

#%%Sigmoid derivative for backprop
def sigmoidDeriv(i):
    return sigmoid(i)*(1-sigmoid(i))
#%%Forward Function    
def forward(i,w1,w2,predict=False):
    a1 = np.matmul(i,w1)
    z1=sigmoid(a1)
    
    #create and add bias
    bias =np.ones((len(z1),1))
    z1=np.concatenate((bias,z1),axis=1)
    a2=np.matmul(z1,w2)
    z2 = sigmoid(a2)
    if predict:
        return z2
    return a1,z1,a2,z2

#backprop function
def backprop(a2,z0,z1,z2,o):
    delta2=z2-o
    D2=np.matmul(z1.T,delta2)
    delta1 = (delta2.dot(w2[1:,:].T))*sigmoidDeriv(a1)
    D1 = np.matmul(z0.T,delta1)
    return delta2,D1,D2

#%%
#First column = bias 
X =np.array([[1,1,0],
             [1,0,1],
             [1,0,0],
             [1,1,1]])
    # Output 
o = np.array([[1],[1],[0],[0]])

w1 = np.random.randn(3,5)
w2 = np.random.randn(6,1) 
#%%    

lr=0.09  
cost=[]
epochs=15000
m=len(X)
for i in range(epochs):
    a1,z1,a2,z2 = forward(X,w1,w2)
    delta2,D1,D2 = backprop(a2,X,z1,z2,o)
        
    w1 -= lr*(1/m)*D1
    w2 -= lr*(1/m)*D2
        
    c=np.mean(np.abs(delta2))
    cost.append(c)
        #if i % 1000 == 0 :
         #   print("Iteration :  {i}. Error : {c}")
print("Training Complete")
z3= forward(X,w1,w2,True)
print("Percentages: ")
print(z3)
print("Predictions: ")
print(np.round(z3))
    
plt.plot(cost)
plt.show()    


