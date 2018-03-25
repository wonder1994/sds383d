# -*- coding: utf-8 -*-
"""
Created on Mon Feb 26 20:53:55 2018

@author: xinjie
"""
import pandas as pd
import numpy as np
from scipy.optimize import minimize


# define posterior function
def posterior(beta, X, y):
    mu = np.zeros(len(beta))
    var = 1
    result = np.dot(y, np.log(1 + np.exp(-np.dot(X, beta)))) + np.dot(1 - y, np.log(1 + np.exp(np.dot(X, beta))))
    result = - result - np.dot(beta - mu, beta - mu)/(2 * var) 
    return result
def negposterior(beta, X, y):
    result = -posterior(beta, X, y)
    return result
# define precision matrix
def precision(beta, X, y):
    var = 1
    Lambda = np.identity(len(beta)) / var
    for i in range(len(y)):
        Lambda = Lambda + y[i] * np.exp(np.dot(X[i, :], beta))/((1 + np.exp(np.dot(X[i, :], beta)))**2) * np.outer(X[i, :], X[i, :])
        Lambda = Lambda + (1 - y[i]) * np.exp(-np.dot(X[i, :], beta))/((1 + np.exp(-np.dot(X[i, :], beta)))**2) * np.outer(X[i, :], X[i, :])
    return Lambda
# preprocessing data
df = pd.read_csv(r'/home/xinjie/Dropbox/courses/2018_Spring/STAT_MODEL/sds383d/R/pima.csv')
df1 = df[['age','class_variable']]
n = df1.shape[0]
X = np.vstack((np.ones(n),df1['age'].values)).T
y = df1['class_variable'].values
res = minimize(negposterior, x0 = np.array([1,1]), args = (X,y), method='Nelder-Mead', tol=1e-6)
# the posterior mode
print('The posterior mode is', res.x)
Lambda = precision(res.x, X, y)
var1 = 1/Lambda[0, 0]
var2 = 1/Lambda[1, 1]
