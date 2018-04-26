#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 26 21:45:28 2018

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
l=50
x_star = np.arange(0, 100, 0.1)
cov = kernel(x_star, x_star, l)
y_star1 = np.random.multivariate_normal(np.zeros(x_star.shape), cov)
y_star2 = np.random.multivariate_normal(np.zeros(x_star.shape), cov)
y_star3 = np.random.multivariate_normal(np.zeros(x_star.shape), cov)
y_star4 = np.random.multivariate_normal(np.zeros(x_star.shape), cov)
y_star5 = np.random.multivariate_normal(np.zeros(x_star.shape), cov)
fig = plt.figure()
plt.plot(x_star, y_star1, 'r-')
plt.plot(x_star, y_star2, 'b-')
plt.plot(x_star, y_star3, 'g-')
plt.plot(x_star, y_star4, 'y-')
plt.plot(x_star, y_star5, 'c-')
plt.xlabel('$x$')
plt.ylabel('$f(x)$')
plt.title("5 samples of function with l = " + str(l))





