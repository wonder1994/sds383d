#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr  9 21:09:48 2018

@author: xinjie
"""

import pandas as pd
import numpy as np
from numpy.linalg import inv, det, norm
import matplotlib.pyplot as plt
from scipy.optimize import minimize
df = pd.read_csv(r'/home/xinjie/Dropbox/courses/2018_Spring/STAT_MODEL/sds383d/code/ex4/iris.csv')
x = np.array(df.iloc[:, 1:3])
y = np.zeros(3 * len(x))
y[0:len(x)] = df.iloc[:, 5] == 'setosa'
y[len(x):2*len(x)] = df.iloc[:, 5] == 'versicolor'
y[2*len(x):3*len(x)] = df.iloc[:, 5] == 'virginica'

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
K_inv = inv(K + 0.001*np.identity(K.shape[0]))
n = len(x)
def posterior(f, K_inv, y, n): 
    result = -0.5 * np.dot(f[0:n], np.dot(K_inv, f[0:n]))-0.5 * np.dot(f[n:2*n], np.dot(K_inv, f[n:2*n]))-0.5 * np.dot(f[2*n:3*n], np.dot(K_inv, f[2*n:3*n]))
    result += np.dot(y, f)
    for i in range(n):
        result -= np.log(np.exp(f[i]) + np.exp(f[i+n]) + np.exp(f[i + 2*n]))
    return result
def negposterior(f, K_inv, y, n):
    result = -posterior(f, K_inv, y, n)
    return result
def derivative(f, K_inv, y, n):
    result = np.zeros(len(f))
    result[0:n] = np.dot(K_inv, f[0:n]) 
    result[n:2*n] = np.dot(K_inv, f[n:2*n]) 
    result[2*n:3*n] = np.dot(K_inv, f[2*n:3*n])
    pi = np.zeros(len(f))
    expf = np.exp(f)
    tmp = (expf[0:n] + expf[n:2*n] + expf[2*n:3*n])
    pi[0:n] = expf[0:n] / tmp
    pi[n:2*n] = expf[n:2*n] / tmp
    pi[2*n:3*n] = expf[2*n:3*n] / tmp
    result  += pi - y
    return result
res = minimize(negposterior, x0 = np.ones(450), args = (K_inv,y,n), method='BFGS', jac=derivative, options={'disp': True})


expf = np.exp(res.x)
predict1 = expf[0:n] / (expf[0:n] + expf[n:2*n] + expf[2*n:3*n])
predict2 = expf[n:2*n]/ (expf[0:n] + expf[n:2*n] + expf[2*n:3*n])
predict3 = expf[2*n:3*n] / (expf[0:n] + expf[n:2*n] + expf[2*n:3*n])
plt.plot(predict1)
plt.plot(predict2)
plt.plot(predict3)
## prediction for new x*
numpoints = 20
x = np.array(df.iloc[:, 1:3])
tmp = np.meshgrid(*[np.linspace(i,j, numpoints) for i, j in zip(np.amin(x, axis = 0), np.amax(x, axis = 0))])
x_star = np.zeros((numpoints**2, 2))
x_star[:,0] = np.reshape(tmp[0], (1,-1))
x_star[:,1] = np.reshape(tmp[1], (1,-1))
l = 1
K = kernel(x, x, l)
K_star = kernel(x_star, x, l)
K_starstar = kernel(x_star, x_star, l)
K_inv = inv(K + 0.0001*np.identity(K.shape[0]))

mean_star1 = K_star @ K_inv @ res.x[0:n]
mean_star2 = K_star @ K_inv @ res.x[n:2*n]
mean_star3 = K_star @ K_inv @ res.x[2*n:3*n]
sd_star = np.power(np.diag(K_starstar - K_star @ K_inv @ K_star.T), 1/2)
mean_matrix1 = np.exp(np.reshape(mean_star1, (numpoints, numpoints)))
mean_matrix2 = np.exp(np.reshape(mean_star2, (numpoints, numpoints)))
mean_matrix3 = np.exp(np.reshape(mean_star3, (numpoints, numpoints)))
sum_mean_matrix = mean_matrix1 + mean_matrix2 + mean_matrix3
prob_matrix1 = mean_matrix1 / sum_mean_matrix
prob_matrix2 = mean_matrix2 / sum_mean_matrix
prob_matrix3 = mean_matrix3 / sum_mean_matrix
sd_matrix = np.reshape(sd_star, (numpoints, numpoints))

plt.subplots_adjust(hspace=0.5,
                    wspace=0.35)
plt.subplot(2,2,1)
plt.pcolormesh(prob_matrix1,  cmap='RdBu_r')
plt.colorbar()
curves = 10
m = max([max(row) for row in prob_matrix1])
levels = np.arange(0, m, (1 / float(curves)) * m)
plt.contour(prob_matrix1, colors="white", levels=levels)
plt.title('Probabilities in class ' + 'setosa')
#plt.xlabel(list(df)[1])
#plt.ylabel(list(df)[2])

plt.subplot(2,2,2)
plt.pcolormesh(prob_matrix2,  cmap='RdBu_r')
plt.colorbar()
curves = 10
m = max([max(row) for row in prob_matrix2])
levels = np.arange(0, m, (1 / float(curves)) * m)
plt.contour(prob_matrix2, colors="white", levels=levels)
plt.title('Probabilities in class ' + 'versicolor')
#plt.xlabel(list(df)[1])
#plt.ylabel(list(df)[2])

plt.subplot(2,2,3)
plt.pcolormesh(prob_matrix3,  cmap='RdBu_r')
plt.colorbar()
curves = 10
m = max([max(row) for row in prob_matrix3])
levels = np.arange(0, m, (1 / float(curves)) * m)
plt.contour(prob_matrix3, colors="white", levels=levels)
plt.title('Probabilities in class ' + 'virginica')
#plt.xlabel(list(df)[1])
#plt.ylabel(list(df)[2])


plt.subplot(2,2,4)
plt.pcolormesh(sd_matrix,  cmap='RdBu_r')
plt.colorbar()
curves = 10
m = max([max(row) for row in sd_matrix])
levels = np.arange(0, m, (1 / float(curves)) * m)
plt.contour(sd_matrix, colors="white", levels=levels)
plt.title('Uncertainty(sd) of predictions')
#plt.xlabel(list(df)[1])
#plt.ylabel(list(df)[2])

