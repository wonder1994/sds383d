# -*- coding: utf-8 -*-
"""
Created on Mon Feb 26 22:52:34 2018

@author: xinjie
"""

import pandas as pd
import numpy as np
from scipy.optimize import minimize


# define posterior function
def posterior(beta, X, y):
    mu = np.zeros(len(beta))
    var = 1
    result = - np.sum(np.exp(np.dot(X, beta))) + np.dot(y, np.dot(X, beta))
    result = result - np.dot(beta - mu, beta - mu)/(2 * var) 
    return result
def negposterior(beta, X, y):
    result = -posterior(beta, X, y)
    return result
# define precision matrix
def precision(beta, X, y):
    var = 1
    Lambda = np.identity(len(beta)) / var
    for i in range(len(y)):
        Lambda = Lambda + (np.exp(np.dot(X[i, :], beta))) * np.outer(X[i, :], X[i, :])
    return Lambda
# preprocessing data
df = pd.read_csv(r'/home/xinjie/Dropbox/courses/2018_Spring/STAT_MODEL/sds383d/R/tea_discipline_oss.csv')
df1 = df[['ACTIONS','GRADE']]
ind = df1['ACTIONS'] >0
data = df1[ind]
#data = df1.values[ind,:].astype(float)
n = data.shape[0]
X = np.vstack((np.ones(n),data['GRADE'].values)).T
X = X.astype(np.int)
y = data['ACTIONS'].values
res = minimize(negposterior, x0 = np.array([1,1]), args = (X,y), method='Nelder-Mead', tol=1e-6)
# the posterior mode
print('The posterior mode is', res.x)
#The posterior mode is [ 9.2008958   0.91760304]
Lambda = precision(res.x, X, y)
#[[  1355.55503888   9227.3900972 ]
# [  9227.3900972   74464.20455676]]
var1 = 1/Lambda[0, 0]
var2 = 1/Lambda[1, 1]
#0.000737705199215
#1.34292712311e-05

9.14657434
9.25521726


0.91027384
0.92493223
