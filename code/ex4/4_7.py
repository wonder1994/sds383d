#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr  3 15:42:40 2018

@author: xinjie
"""

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
# k = 16 and 5
k = 5
global y
y = df['eruptions'].values[10*k:10*(k+1)]
global x
x = df['waiting'].values[10*k:10*(k+1)]
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

theta0 = [1,1,1]
res = minimize(obj, theta0, method = 'nelder-mead', options={'xtol': 1e-08, 'disp': True})
res.x

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

# Out[146]: array([  1.27596238e+02,   1.02164179e+01,   6.51988054e-02]) start from 50-60
# [  5.49491049,  10.84382461,  -0.60791999]
l2, alpha2, sigma2 = res.x
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
#plt.title("Gaussian Process Prediction trained with randomly selected data 1")



# plot negative likelihood. 
x_grid = np.arange(-4, 10, 0.1)
y_grid = np.arange(-4, 10, 0.1)
x_grid = np.exp(x_grid)
y_grid = np.exp(y_grid)
xv, yv = np.meshgrid(x_grid, y_grid)
likelihood4 = np.zeros(xv.shape)
for i in range(likelihood4.shape[0]):
    for j in range(likelihood4.shape[1]):
        likelihood4[i,j] = obj(np.array([xv[i,j], 10 ,yv[i,j]]))
likelihood4 = pd.DataFrame(likelihood4)
likelihood4.index = np.round(x_grid, 2)
likelihood4.columns = np.round(y_grid, 2)



import seaborn as sns
ax = sns.heatmap(likelihood4)


