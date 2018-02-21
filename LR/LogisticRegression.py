# -*- coding: utf-8 -*-
"""
Created on Tue Feb 20 20:35:49 2018

@author: Auser
"""

from math import *
from numpy import *   


    
def sigmoid(X):
    return 1.0 / ( 1.0 + exp (-1.0 * X ) )
    
    

def compute_cost(theta,X,y):
    ''' one row for a sample '''
    ''' y is a label column '''
    ''' theta is a parameter column '''
    N = X.shape[0]
    J = -(1/N) * transpose(y).dot(log(sigmoid(X.dot(theta)))) + transpose(1-y).dot(log(1-sigmoid(X.dot(theta))))
    return J[0][0]
    
    
    
def compute_grad(theta,X,y):
    N = X.shape[0]
    grad = (1/N) * transpose(X).dot(sigmoid(X.dot(theta))- y) 
    return grad



def gradient_descent(theta,X,y,alpha,max_step):
    for i in range(max_step):
        grad = compute_grad(theta,X,y)
        theta = theta - alpha * grad
    return theta



def stochastic_gradient_descent(theta,X,y,alpha,max_step):
    N = X.shape[0]
    for i in range(max_step):
        for j in range(N):
            grad = compute_grad(theta,matrix(X[j,:]),y[j])
            theta = theta - alpha * grad
    return theta



def LR(X,y,alpha,max_step,method):
    m,n=shape(X)
    theta=ones((n,1))
    if method in ["GD","gd"]:
        return gradient_descent(theta,X,y,alpha,max_step)
    elif  method in ["SGD","sgd"]:
        return stochastic_gradient_descent(theta,X,y,alpha,max_step)
    else:
        raise ValueError
    
    

def predict(theta,X):
    ''' x is a sample row '''
    m=shape(X)[0]
    result=zeros((m,1))
    for i in range(m):
        prob = sigmoid(X[i,:].dot(theta))
        if prob > 0.5:
            result[i] = 1
        else:
            result[i] = 0
    return result