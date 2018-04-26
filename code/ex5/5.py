#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Apr 14 19:50:36 2018

@author: xinjie
"""
import os
os.chdir(r'/home/xinjie/Dropbox/courses/2018_Spring/STAT_MODEL/sds383d/code/ex5/')
from sklearn import preprocessing
import pandas as pd
import numpy as np
from numpy.linalg import inv, det, norm
import matplotlib.pyplot as plt
from scipy.optimize import minimize
df = pd.read_csv(r'/home/xinjie/Dropbox/courses/2018_Spring/STAT_MODEL/sds383d/code/ex5/restaurants.csv')

def gibbs(x, n):
    burnin = 1000
    w_samples = 0.5 * np.ones(n + burnin)
    mu1_samples = np.zeros(n + burnin)
    mu2_samples = np.zeros(n + burnin)
    lambda_samples1 = np.ones(n + burnin)
    lambda_samples2 = np.ones(n + burnin)
    x_samples = np.ones(n + burnin)
    d = np.zeros(len(x))
    for i in range(1, n + burnin):
        prob1 = w_samples[i-1] * np.array(list(map(lambda r: normpdf(r, mu1_samples[i-1], 1 / lambda_samples1[i-1]), x)))
        prob2 = (1 - w_samples[i-1]) *np.array(list(map(lambda r: normpdf(r, mu2_samples[i-1], 1 / lambda_samples2[i-1]), x)))
        d = np.random.binomial(1, prob2/(prob1 + prob2)) + 1
        index1 = d == 1
        index2 = d == 2
        n1 = sum(index1)
        n2 = sum(index2)
        w_samples[i] = np.random.beta(1 + n1, 1 + n2)
        lambda_samples1[i] = np.random.gamma(1 + n1/2, 1 / (1 + (sum(np.power(x[index1] - mu1_samples[i-1],2)) )/2))
        lambda_samples2[i] = np.random.gamma(1 + n2/2, 1 / (1 + (sum(np.power(x[index2] - mu2_samples[i-1],2)) )/2))
        mu1_samples[i] = np.random.normal(sum(x[index1])/(1/(100*lambda_samples1[i]) + n1), np.sqrt(1/(1/100 + n1 * lambda_samples1[i])))
        mu2_samples[i] = np.random.normal(sum(x[index2])/(1/(100*lambda_samples2[i]) + n2), np.sqrt(1/(1/100 + n2 * lambda_samples2[i])))
        tmp = np.random.binomial(1, w_samples[i])
        x_samples[i] = tmp * np.random.normal(mu1_samples[i], np.sqrt(1/lambda_samples1[i])) + (1 - tmp) * np.random.normal(mu2_samples[i], np.sqrt(1/lambda_samples2[i]))
    return w_samples[burnin: ], mu1_samples[burnin: ], mu2_samples[burnin: ],lambda_samples1[burnin: ], lambda_samples2[burnin: ], d, x_samples[burnin:, ]
def normpdf(x, mu, sigma2):
    return np.exp(-(x-mu)**2/(2*sigma2))/np.sqrt(2*np.pi*sigma2)
data = df.values[:, 1:]
data[:,0] = preprocessing.scale(data[:,0])
y = data[:, 0]
w_samples, mu1_samples, mu2_samples, lambda_samples1,lambda_samples2, d, x_samples = gibbs(y, 1000)
d = d - 1
np.sum(data[:, 1]==(d))
np.sum(np.logical_and(data[:, 1]==0, d==0))
np.sum(np.logical_and(data[:, 1]==0, d==1))
np.sum(np.logical_and(data[:, 1]==1, d==0))
np.sum(np.logical_and(data[:, 1]==1, d==1))

plt.figure(figsize=(15,8))
plt.subplots_adjust(hspace=0.4)
plt.subplot(2,3,1)
plt.hist(mu1_samples)
plt.xlabel('$\mu_1$')
plt.subplot(2,3,2)
plt.hist(mu2_samples)
plt.xlabel('$\mu_2$')
plt.subplot(2,3,3)
plt.hist(1/lambda_samples1)
plt.xlabel('$\sigma_1$')
plt.subplot(2,3,4)
plt.hist(1/lambda_samples2)
plt.xlabel('$\lambda_2$')
plt.subplot(2,3,5)
plt.hist(w_samples)
plt.xlabel('$w$')
plt.subplot(2,3,6)
plt.hist(x_samples, bins = 30,ls='dashed', alpha = 0.5, lw=3, color="b", normed =1)
plt.hist(y, bins =30, ls='dotted', alpha = 0.5, lw=3, color="r", normed = 1)
plt.legend(['Fitted model','data'])
plt.savefig(r'5.png')

