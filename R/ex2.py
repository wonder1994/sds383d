#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 16 11:37:26 2018

@author: xinjie
"""
# import libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

## sampling function
def sampler1(data, Lambda, K, a, b, mu, nsamples):
    n, p = data.shape
    p = p - 1
    y = data[:, 0]
    y.shape = (n, 1)
    X = data[:, 1:]
    beta = np.zeros((nsamples, p))
    omega = np.zeros(nsamples)
    omega.shape = (nsamples, 1)
    a_n = a + n/2
    b_n = b + (np.transpose(mu)@K@(mu) + np.transpose(y)@Lambda@y- np.transpose(K@mu+np.transpose(X)@Lambda@y)@np.linalg.inv(K + np.transpose(X)@Lambda@X)@(K@mu + np.transpose(X)@Lambda@y))/2 
    for i in range(nsamples):
        # first sample omega
        omega[i] = np.random.gamma(shape = a_n, scale = 1/b_n)      
        Lambda_n = omega[i]*K + omega[i]*np.transpose(X)@Lambda@X
        Cov_n = np.linalg.inv(Lambda_n)
        mu_n = Cov_n@(omega[i]*K@mu + omega[i]*np.transpose(X)@Lambda@y)
        beta[i, :] = np.random.multivariate_normal(mu_n[:, 0], Cov_n)
    return beta, omega


def sampler2(data, K, a, b, mu, tau, nsamples):
    n, p = data.shape
    burnin = 2000
    p = p - 1
    y = data[:, 0]
    y.shape = (n, 1)
    X = data[:, 1:]
    beta = np.zeros((nsamples+burnin, p))
    omega = np.zeros(nsamples+burnin)
    omega.shape = (nsamples+burnin, 1)
    sample_Lambda = np.zeros((nsamples+burnin, n))
    Lambda = np.identity(n)
    for i in range(nsamples + burnin):
        # first sample omega
        a_n = a + n/2
        b_n = b + (np.transpose(mu)@K@(mu) + np.transpose(y)@Lambda@y- np.transpose(K@mu+np.transpose(X)@Lambda@y)@np.linalg.inv(K + np.transpose(X)@Lambda@X)@(K@mu + np.transpose(X)@Lambda@y))/2 
        omega[i] = np.random.gamma(shape = a_n, scale = 1/b_n)      
        Lambda_n = omega[i]*K + omega[i]*np.transpose(X)@Lambda@X
        Cov_n = np.linalg.inv(Lambda_n)
        mu_n = Cov_n@(omega[i]*K@mu + omega[i]*np.transpose(X)@Lambda@y)
        beta[i, :] = np.random.multivariate_normal(mu_n[:, 0], Cov_n)
        y_bar = X@beta[i,:]
        y_bar.shape = (n, 1)
        sample_Lambda[i, :] = np.random.gamma(tau+1/2, tau+omega[i]*(y-y_bar)**2/2).flatten()
        Lambda = np.diag(sample_Lambda[i,:])
    return beta[burnin:, :], omega[burnin:], sample_Lambda[burnin:,:]


def sampler3(data, a, b, mu, tau, nsamples):
    n, p = data.shape
    burnin = 2000
    p = p - 1
    y = data[:, 0]
    y.shape = (n, 1)
    X = data[:, 1:]
    beta = np.zeros((nsamples+burnin, p))
    omega = np.zeros(nsamples+burnin)
    omega.shape = (nsamples+burnin, 1)
    sample_Lambda = np.zeros((nsamples+burnin, n))
    sample_K = np.zeros((nsamples+burnin, p))
    Lambda = np.identity(n)
    K = np.identity(p)
    for i in range(nsamples + burnin):
        # first sample omega
        a_n = a + n/2
        b_n = b + (np.transpose(mu)@K@(mu) + np.transpose(y)@Lambda@y- np.transpose(K@mu+np.transpose(X)@Lambda@y)@np.linalg.inv(K + np.transpose(X)@Lambda@X)@(K@mu + np.transpose(X)@Lambda@y))/2 
        omega[i] = np.random.gamma(shape = a_n, scale = 1/b_n)      
        Lambda_n = omega[i]*K + omega[i]*np.transpose(X)@Lambda@X
        Cov_n = np.linalg.inv(Lambda_n)
        mu_n = Cov_n@(omega[i]*K@mu + omega[i]*np.transpose(X)@Lambda@y)
        beta[i, :] = np.random.multivariate_normal(mu_n[:, 0], Cov_n)
        y_bar = X@beta[i,:]
        y_bar.shape = (n, 1)
        sample_Lambda[i, :] = np.random.gamma(tau+1/2, tau+omega[i]*(y-y_bar)**2/2).flatten()
        Lambda = np.diag(sample_Lambda[i,:])
        sample_K[i,:] = np.random.gamma(tau+1/2, tau + omega[i]*(beta[i,:])**2/2)
        K = np.diag(sample_K[i,:])
    return beta[burnin:, :], omega[burnin:], sample_Lambda[burnin:,:], sample_K[burnin:,:]



## preprocessing data
df = pd.read_csv(r'/home/xinjie/Dropbox/courses/2018_Spring/STAT_MODEL/sds383d/R/dental.csv')
df = df.drop(['Unnamed: 0', 'Subject'], axis = 1)
ind = df['Sex'] == 'Male'
df['Sex'] = ind.astype('int')
[nrow, ncol] = df.shape
data = df.values
X_0 = np.ones((nrow,1))
data = np.hstack((data, X_0))

## setting the priors
Lambda = np.identity(nrow)
K = np.identity(ncol)
nsamples = 1000
a = 1
b = 1
tau = 1
mu = np.zeros(ncol)
mu.shape = (ncol, 1)

## fixed Lambda
beta1, omega1 = sampler1(data, Lambda, K, a, b, mu, nsamples)

## put a prior on Lambda
beta2, omega2, lambda2 = sampler2(data, K, a, b, mu, tau, nsamples)
## only male
beta2, omega2, lambda2 = sampler2(data[ind,:], K, a, b, mu, tau, nsamples)
##  only femalE
beta2, omega2, lambda2 = sampler2(data[-ind,:], K, a, b, mu, tau, nsamples)

# put a prior on K
beta3, omega3, lambda3, k3 = sampler3(data, a, b, mu, tau, nsamples)







## least squares
X = data[:, 1:]
y = data[:, 0]
beta_ls = np.linalg.inv(np.transpose(X)@X)@(np.transpose(X)@y)
## ridge regression
beta_ridge = np.linalg.inv(np.identity(ncol) + np.transpose(X)@X)@(np.transpose(X)@y)




## plot
plt.hist(beta1[:, 0], color = 'y')
plt.plot([0.660185,0.660185],[0,250], lw = 2)
plt.annotate('ridge regression', xy=(0.911105, 200), xytext=(1.1, 200),
            arrowprops=dict(facecolor='black', shrink=0.05),
            )
plt.annotate('ls regression', xy=(0.660185, 220), xytext=(0.8, 220),
            arrowprops=dict(facecolor='black', shrink=0.01),
            )
plt.plot([0.911105,0.911105],[0,250], lw = 2, color = 'red')
plt.title(r"Histogram for the coefficient of age")

plt.hist(beta1[:, 1], color = 'y')
plt.plot([2.32102,2.32102],[0,250], lw = 2)
plt.annotate('ls regression', xy=(2.32102, 220), xytext=(2.8, 220),
            arrowprops=dict(facecolor='black', shrink=0.01),
            )
plt.plot([2.50672,2.50672],[0,250], lw = 2, color = 'red')
plt.annotate('ridge regression', xy=(2.50672, 200), xytext=(3, 200),
            arrowprops=dict(facecolor='black', shrink=0.05),
            )
plt.title(r"Histogram for the coefficient of sex")

plt.hist(beta1[:, 2], color = 'y')
plt.plot([15.3857,15.3857],[0,250], lw = 2)
plt.annotate('ls regression', xy=(15.3857, 220), xytext=(16, 220),
            arrowprops=dict(facecolor='black', shrink=0.01),
            )
plt.plot([12.4007,12.4007],[0,250], lw = 2, color = 'red')
plt.annotate('ridge regression', xy=(12.4007, 200), xytext=(14, 200),
            arrowprops=dict(facecolor='black', shrink=0.05),
            )
plt.title(r"Histogram for intercept")






## plot
plt.hist(beta2[:, 0], color = 'y')
plt.plot([0.660185,0.660185],[0,250], lw = 2)
plt.annotate('ridge regression', xy=(0.911105, 200), xytext=(1.1, 200),
            arrowprops=dict(facecolor='black', shrink=0.05),
            )
plt.annotate('ls regression', xy=(0.660185, 220), xytext=(0.8, 220),
            arrowprops=dict(facecolor='black', shrink=0.01),
            )
plt.plot([0.911105,0.911105],[0,250], lw = 2, color = 'red')
plt.title(r"Histogram for the coefficient of age")
plt.show()

plt.hist(beta2[:, 1], color = 'y')
plt.plot([2.32102,2.32102],[0,250], lw = 2)
plt.annotate('ls regression', xy=(2.32102, 220), xytext=(2.8, 220),
            arrowprops=dict(facecolor='black', shrink=0.01),
            )
plt.plot([2.50672,2.50672],[0,250], lw = 2, color = 'red')
plt.annotate('ridge regression', xy=(2.50672, 200), xytext=(3, 200),
            arrowprops=dict(facecolor='black', shrink=0.05),
            )
plt.title(r"Histogram for the coefficient of sex")
plt.show()

plt.hist(beta2[:, 2], color = 'y')
plt.plot([15.3857,15.3857],[0,250], lw = 2)
plt.annotate('ls regression', xy=(15.3857, 220), xytext=(16, 220),
            arrowprops=dict(facecolor='black', shrink=0.01),
            )
plt.plot([12.4007,12.4007],[0,250], lw = 2, color = 'red')
plt.annotate('ridge regression', xy=(12.4007, 200), xytext=(14, 200),
            arrowprops=dict(facecolor='black', shrink=0.05),
            )
plt.title(r"Histogram for intercept")
plt.show()




## plot
plt.hist(beta3[:, 0], color = 'y')
plt.plot([0.660185,0.660185],[0,250], lw = 2)
plt.annotate('ridge regression', xy=(0.911105, 200), xytext=(1.1, 200),
            arrowprops=dict(facecolor='black', shrink=0.05),
            )
plt.annotate('ls regression', xy=(0.660185, 220), xytext=(0.8, 220),
            arrowprops=dict(facecolor='black', shrink=0.01),
            )
plt.plot([0.911105,0.911105],[0,250], lw = 2, color = 'red')
plt.title(r"Histogram for the coefficient of age")
plt.show()

plt.hist(beta3[:, 1], color = 'y')
plt.plot([2.32102,2.32102],[0,250], lw = 2)
plt.annotate('ls regression', xy=(2.32102, 220), xytext=(2.8, 220),
            arrowprops=dict(facecolor='black', shrink=0.01),
            )
plt.plot([2.50672,2.50672],[0,250], lw = 2, color = 'red')
plt.annotate('ridge regression', xy=(2.50672, 200), xytext=(3, 200),
            arrowprops=dict(facecolor='black', shrink=0.05),
            )
plt.title(r"Histogram for the coefficient of sex")
plt.show()

plt.hist(beta3[:, 2], color = 'y')
plt.plot([15.3857,15.3857],[0,250], lw = 2)
plt.annotate('ls regression', xy=(15.3857, 220), xytext=(16, 220),
            arrowprops=dict(facecolor='black', shrink=0.01),
            )
plt.plot([12.4007,12.4007],[0,250], lw = 2, color = 'red')
plt.annotate('ridge regression', xy=(12.4007, 200), xytext=(14, 200),
            arrowprops=dict(facecolor='black', shrink=0.05),
            )
plt.title(r"Histogram for intercept")
plt.show()



















