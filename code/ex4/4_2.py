#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 26 19:56:42 2018

@author: xinjie
"""

import pandas as pd
import numpy as np
from numpy.linalg import inv
import matplotlib.pyplot as plt
def kernel(x, y):
    n = len(x)
    m = len(y)
    result = np.zeros((n,m))
    for i in range(n):
        for j in range(m):
            result[i,j] = kernel_comp(x[i], y[j])
    return result
def kernel_comp(x, y):
    return 1+ x*y + (x*y)**2 + (x*y)**3

df = pd.read_csv(r'/home/xinjie/Dropbox/courses/2018_Spring/STAT_MODEL/sds383d/code/ex4/faithful.csv')

y = df['eruptions'].values
x = df['waiting'].values
plt.scatter(x, y)
x_star = np.arange(40, 100, 0.5)
n = len(x)
Phi = np.zeros((4, n))
Phi[0,:] = 1
Phi[1,:] = x
Phi[2,:] = np.power(x, 2)
Phi[3,:] = np.power(x, 3)
Phi_star = np.zeros((4, len(x_star)))
Phi_star[0,:] = 1
Phi_star[1,:] = x_star
Phi_star[2,:] = np.power(x_star, 2)
Phi_star[3,:] = np.power(x_star, 3)
A = Phi @ Phi.T + np.identity(4)
A_inv = inv(A)
mean_star = Phi_star.T @ A_inv @ Phi @ y
sd_star = np.power(np.diag(Phi_star.T @ A_inv @ Phi_star), 0.5)
#
#K = kernel(x, x)
#K_star = kernel(x_star, x)
#K_starstar = kernel(x_star, x_star)
#K_inv = inv(K + 500*np.identity(K.shape[0]))
#
#mean_star = K_star @ K_inv @ y
#sd_star = np.power(np.diag(K_starstar - K_star @ K_inv @ kernel(x, x_star)), 1/2)

fig = plt.figure()
plt.plot(x, y, 'r.', markersize=10, label=u'Observations')
plt.plot(x_star, mean_star, 'b-', label=u'Prediction')
plt.fill(np.concatenate([x_star, x_star[::-1]]),
        np.concatenate([mean_star - 1.9600 * sd_star,
                       (mean_star + 1.9600 * sd_star)[::-1]]),
        alpha=.5, fc='b', ec='None', label='95% confidence interval')
plt.xlabel('$x$')
plt.ylabel('$f(x)$')
plt.ylim(0, 6)
plt.legend(loc='upper left')

