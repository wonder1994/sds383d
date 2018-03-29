#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 26 22:01:16 2018

@author: xinjie
"""

import pandas as pd
import numpy as np
from numpy.linalg import inv
import matplotlib.pyplot as plt
def kernel(x, y, l):
    n = len(x)
    m = len(y)
    result = np.zeros((n,m))
    for i in range(n):
        for j in range(m):
            result[i,j] = kernel_comp(x[i], y[j], l)
    return result
def kernel_comp(x, y, l):
    return np.exp(-(x-y)**2/(2*l**2))

df = pd.read_csv(r'/home/xinjie/Dropbox/courses/2018_Spring/STAT_MODEL/sds383d/code/ex4/faithful.csv')

y = df['eruptions'].values
x = df['waiting'].values
plt.scatter(x, y)
x_star = np.arange(0, 100, 0.5)
n = len(x)
l = 5
K = kernel(x, x, l)
K_star = kernel(x_star, x, l)
K_starstar = kernel(x_star, x_star, l)
K_inv = inv(K + 1*np.identity(K.shape[0]))

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
plt.title("Gaussian Process Prediction with lengthscale = " + str(l))
