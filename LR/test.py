# -*- coding: utf-8 -*-
"""
Created on Tue Feb 20 22:04:11 2018

@author: Auser
"""

from LogisticRegression import *
from Metric import *

import numpy as np
import matplotlib.pyplot as plt

''' data generation '''
mu = np.array([[1,1],[5,5]])
Sigma = np.array([[1, 0], [0, 1]])
sampleNo = 20

data_1 = np.random.multivariate_normal(mu[0],Sigma,sampleNo)
data_2 = np.random.multivariate_normal(mu[1],Sigma,sampleNo)
label_1 = np.ones((sampleNo,1))
label_2 = np.zeros((sampleNo,1))

data = np.row_stack((data_1,data_2))
data = np.column_stack((data,np.ones((40,1)))) # bias
label = np.row_stack((label_1,label_2))


''' shuffle the data '''
d_l = np.column_stack((data,label))
np.random.shuffle(d_l)
sdata = d_l[:,0:3]
slabel_0 = d_l[:,3]
slabel = transpose(np.matrix(slabel_0))


''' train classifier '''
theta = LR(sdata,slabel,0.01,10,"SGD")
# print(theta)
result = predict(theta,sdata)
print("No. of correct =",sum(result==slabel))
print("Accuracy =",accuracy(result,slabel))
print("Precision =",precision(result,slabel))
print("Recall =",recall(result,slabel))
print("F1score =",F1score(result,slabel))



''' plot '''
#u = linspace(-1,8,100)  
#v = linspace(-1,8,100)  
#z = zeros(shape=(len(u), len(v)))  
#for i in range(len(u)):  
#    for j in range(len(v)):  
#        z[i, j]=predict(theta,np.matrix(np.array((u[i],v[j],1))))
##z = z.T
#plt.contourf(u, v, z)  
#plt.scatter(sdata[:, 0], sdata[:, 1],c=slabel_0)


''' Ax + By + C = 0 '''
''' y = -(Ax+C)/B '''
x = arange(-2.0,8.0,0.1)
y = -(theta[0]*x+theta[2])/theta[1]
plt.plot(x,y.T,color="yellow")
plt.scatter(sdata[:, 0], sdata[:, 1],c=slabel_0)

