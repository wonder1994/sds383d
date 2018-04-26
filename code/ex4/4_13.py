#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr  4 09:31:13 2018

@author: xinjie
"""

plt.contour(x[:,0], x[:, 1], pre_prob)

delta = 0.025
a = np.arange(-3.0, 3.0, delta)
b = np.arange(-2.0, 2.0, delta)
X, Y = np.meshgrid(a, b)
Z1 = np.exp(-X**2 - Y**2)
Z2 = np.exp(-(X - 1)**2 - (Y - 1)**2)
Z = (Z1 - Z2) * 2
CS = plt.contour(X, Y, Z)