#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr  3 10:45:07 2018

@author: xinjie
"""
import pandas as pd
import numpy as np
from numpy.linalg import inv, det
import matplotlib.pyplot as plt
from scipy.optimize import minimize
df = pd.read_csv(r'/home/xinjie/Dropbox/courses/2018_Spring/STAT_MODEL/sds383d/code/ex4/faithful.csv')

global y
y = df['eruptions'].values
global x
x = df['waiting'].values
global A
A = np.ones([len(x), len(x)]) * x
A = A - A.T
def obj(theta):
    l2 = theta[0]
    alpha2 = theta[1] 
    sigma2 = theta[2]
    K = kernel(x, x, l2, alpha2, sigma2)
    K_inv = inv(K)
    # alpha = K_inv @ y
    return -(-0.5 * np.dot(y, np.dot(K_inv,y)) - 0.5 * np.log(det(K)))
    
def kernel(x1, x2, l2, alpha2, sigma2):
    n = len(x1)
    m = len(x2)
    result = np.zeros((n,m))
    for i in range(n):
        for j in range(m):
            result[i,j] = alpha2 * kernel_comp(x1[i], x2[j], l2)
    return result + sigma2 * np.identity(result.shape[0])
def kernel2(x1, x2, l2, alpha2):
    n = len(x1)
    m = len(x2)
    result = np.zeros((n,m))
    for i in range(n):
        for j in range(m):
            result[i,j] = alpha2 * kernel_comp(x1[i], x2[j], l2)
    return result
def kernel_comp(x1, x2, l2):
    return np.exp(-(x1-x2)**2/(2*l2))

theta0 = [10,10,10]
res = minimize(obj, theta0, method = 'nelder-mead', options={'xtol': 1e-08, 'disp': True})


def obj_der(theta):
    l2 = theta[0]
    alpha2 = theta[1] 
    sigma2 = theta[2]
    result = np.zeros(len(theta))
    K = kernel(x, x, l2, alpha2, sigma2)
    K_tmp = kernel(x, x, l2, 1, 0)
    K_inv = inv(K)
    alpha = K_inv @ y
    tmp = (np.outer(alpha, alpha) - K_inv)
    result[0] = - 0.5 * np.trace(alpha2 * tmp @ (K_tmp * np.power(A, 2)/(2 * l2 ** 2)))
    result[1] = - 0.5 * np.trace(tmp @ K_tmp )
    result[2] = - 0.5 * np.trace(tmp)
    # alpha = K_inv @ y
    return result

res = minimize(obj, theta0, method='BFGS', jac=obj_der, options={'disp': True})

# Out[146]: array([ 4.82043894,  7.27105305,  0.15036074]) start from [1,1,1]
# Out[146]: array([  1.66300361e+02,   7.10354094e+00,   1.37495514e-01]) start from [10,10,10]
l2 = 4.82043894
alpha2 = 7.27105305
sigma2 = 0.15036074

x_star = np.arange(0, 100, 0.5)
n = len(x)
l = 5
K = kernel(x, x, l2, alpha2, sigma2)
K_star = kernel2(x_star, x, l2, alpha2)
K_starstar = kernel2(x_star, x_star, l2, alpha2)
K_inv = inv(K )

mean_star = K_star @ K_inv @ y
sd_star = np.power(np.diag(K_starstar - K_star @ K_inv @ K_star.T), 1/2)

fig = plt.figure()
plt.plot(x, y, 'r.', markersize=10, label=u'Observations')
plt.plot(x_star, mean_star, 'b-', label=u'Prediction')
plt.fill(np.concatenate([x_star, x_star[::-1]]),
        np.concatenate([mean_star - 1.9600 * sd_star,
                       (mean_star + 1.9600 * sd_star)[::-1]]),
        alpha=.5, fc='b', ec='None', label='95% confidence interval')
plt.xlabel('$x$')
plt.ylabel('$f(x)$')
plt.legend(loc='upper left')
plt.title("Gaussian Process Prediction with lengthscale = " + str(l2))


















