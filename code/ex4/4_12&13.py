#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr  3 21:16:07 2018

@author: xinjie
"""

import pandas as pd
import numpy as np
from numpy.linalg import inv, det, norm
import matplotlib.pyplot as plt
from scipy.optimize import minimize
df = pd.read_csv(r'/home/xinjie/Dropbox/courses/2018_Spring/STAT_MODEL/sds383d/code/ex4/iris.csv')
index = np.array(df.loc[:, 'Species'] != 'virginica')
x = np.array(df.iloc[index, 1:5])
y = np.array(df.iloc[index, 5] == 'versicolor')
y = y * 1

def kernel(x, y, l):
    n = x.shape[0]
    m = y.shape[0]
    result = np.zeros((n,m))
    for i in range(n):
        for j in range(m):
            result[i,j] = kernel_comp(x[i,:], y[j, :], l)
    return result
def kernel_comp(x, y, l):
    return np.exp(-norm(x-y)**2/(2*l**2))

K = kernel(x, x, 1)
K_inv = inv(K + 0.01 * np.identity(K.shape[0]))
def posterior(f, K_inv, y): 
    return np.dot(np.log(1 + np.exp(-f)), -y) - np.dot(np.log(1 + np.exp(f)), 1 - y) - 0.5 * np.dot(f, np.dot(K_inv, f))
def negposterior(f, K_inv, y):
    result = -posterior(f, K_inv, y)
    return result

res = minimize(negposterior, x0 = np.ones(100), args = (K_inv,y), method='BFGS', options={'xtol': 1e-8, 'disp': True})
pre_prob = 1./(1 + np.exp(-res.x))

plt.subplot(1, 2, 1)
plt.scatter(x[:,0], x[:, 1], c=pre_prob, s=20, edgecolors='black')
plt.xlabel(list(df)[1])
plt.ylabel(list(df)[2])
plt.title("Gaussian Process Prediction")
plt.gray()
plt.subplot(1, 2, 2)
plt.scatter(x[:,0], x[:, 1], c=y, s=20, edgecolors='black')
plt.xlabel(list(df)[1])
plt.ylabel(list(df)[2])
plt.title("True classification")
plt.subplots_adjust(wspace=0.5)


plt.subplot(1, 2, 1)
plt.scatter(x[:,2], x[:, 3], c=pre_prob, s=20, edgecolors='black')
plt.xlabel(list(df)[3])
plt.ylabel(list(df)[4])
plt.title("Gaussian Process Prediction")
plt.gray()
plt.subplot(1, 2, 2)
plt.scatter(x[:,2], x[:, 3], c=y, s=20, edgecolors='black')
plt.xlabel(list(df)[3])
plt.ylabel(list(df)[4])
plt.title("True classification")
plt.subplots_adjust(wspace=0.5)

## prediction for new x*
numpoints = 20
x = np.array(df.iloc[index, 1:3])
tmp = np.meshgrid(*[np.linspace(i,j, numpoints) for i, j in zip(np.amin(x, axis = 0), np.amax(x, axis = 0))])
x_star = np.zeros((numpoints**2, 2))
x_star[:,0] = np.reshape(tmp[0], (1,-1))
x_star[:,1] = np.reshape(tmp[1], (1,-1))
l = 1
K = kernel(x, x, l)
K_star = kernel(x_star, x, l)
K_starstar = kernel(x_star, x_star, l)
K_inv = inv(K + 0.0001*np.identity(K.shape[0]))

mean_star = K_star @ K_inv @ res.x
sd_star = np.power(np.diag(K_starstar - K_star @ K_inv @ K_star.T), 1/2)
mean_matrix = np.reshape(mean_star, (numpoints, numpoints))
prob_matrix = 1./(1 + np.exp(-mean_matrix))
sd_matrix = np.reshape(sd_star, (numpoints, numpoints))

plt.pcolormesh(prob_matrix,  cmap='RdBu_r')
plt.colorbar()
curves = 10
m = max([max(row) for row in prob_matrix])
levels = np.arange(0, m, (1 / float(curves)) * m)
plt.contour(prob_matrix, colors="white", levels=levels)
plt.title('Prediction with Gaussian processes')
plt.xlabel(list(df)[1])
plt.ylabel(list(df)[2])
plt.show()

plt.pcolormesh(sd_matrix,  cmap='RdBu_r')
plt.colorbar()
curves = 10
m = max([max(row) for row in sd_matrix])
levels = np.arange(0, m, (1 / float(curves)) * m)
plt.contour(sd_matrix, colors="white", levels=levels)
plt.title('Uncertainty(sd) of predictions with Gaussian processes')
plt.xlabel(list(df)[1])
plt.ylabel(list(df)[2])
plt.show()


#
#
## 4_13 Hessian
#H = np.diag(y * np.exp(res.x) / (np.power(1 + np.exp(res.x), 2)) + (1 - y) * np.exp(-res.x) / (np.power(1 + np.exp(-res.x) ,2))) + (K_inv)
#H_inv = inv(H)
#std = np.sqrt(np.diag(H_inv))
#
#plt.scatter(x[:,2], x[:, 3], c=std, s=20, edgecolors='black')
#plt.xlabel(list(df)[3])
#plt.ylabel(list(df)[4])
#plt.title("Standard deviation of Gaussian Process Prediction")




