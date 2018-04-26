#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 24 21:10:41 2018

@author: xinjie
"""


import os
os.chdir(r'/home/xinjie/Dropbox/courses/2018_Spring/STAT_MODEL/sds383d/code/ex5/')
import random 
random.seed(1)
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# for any dimensional data and number of clusters
def gibbs(x, n, K):
    nx, p = x.shape
    burnin = 1000
    w_samples = 1/K * np.ones((n + burnin, K))
    mu_samples = np.zeros((K, n + burnin, p))
    lambda_samples = np.ones((K, n + burnin))
    D = np.zeros((nx,K))
    O = np.zeros((nx,nx))
    for i in range(1, n + burnin):
        # first update D
        # matrix probabilities
        prob = np.array([np.array(list(map(lambda r: normpdf(r, mu_samples[k,i-1,:], 1 / lambda_samples[k, i-1]), x))) for k in range(K)]).T * w_samples[i-1,:]
        prob = (prob.T / np.sum(prob, axis = 1)).T
        for j in range(nx):
            D[j,:] = np.random.multinomial(1, prob[j,:], size = 1)
        O += np.dot(D, D.T)
        # update w
        count = np.sum(D, axis = 0)
        w_samples[i,:] = np.random.dirichlet(1 + count)
        # update lambda and mu
        for k in range(K):
            nk = count[k]
            index = D[:, k].astype(bool)
            lambda_samples[k, i] = 1 #np.random.gamma(1 + nk/2, 1 / (1 + (np.sum(np.power(x[index,:] - mu_samples[k, i-1, :],2)) )/2))
            mu_samples[k, i, :] = np.random.multivariate_normal(np.sum(x[index], axis = 0)/(1/(100*lambda_samples[k, i]) + nk), np.identity(p) * (1/(1/100 + nk * lambda_samples[k,i])))
    return w_samples[burnin:, ], mu_samples[:,burnin:, ], lambda_samples[:, burnin: ], D, O
def normpdf(x, mu, sigma2):
    return np.exp(-np.linalg.norm(x-mu)**2/(2*sigma2))/np.power(sigma2, len(mu)/2)
# data generation
K = 5
dim = 2
n = 1000
w = np.random.dirichlet([10,]*K)
mu = np.random.multivariate_normal([0,]*dim, 20*np.identity(dim), K)
D = np.random.multinomial(1, w, n)
x = np.array([np.random.multivariate_normal(mu[np.dot(D[i,:], np.arange(K)),:], np.identity(dim)) for i in range(n)] )  

w_samples, mu_samples, lambda_samples, D = gibbs(x, 1000, K)

LABEL_COLOR_MAP = {0 : 'r',
                   1 : 'k',
                   2 : 'b',
                   3 : 'g',
                   4 : 'c'
                   }
label = np.dot(D, np.arange(K))
label_color = [LABEL_COLOR_MAP[l] for l in label]
plt.scatter(x[:,0],x[:,1], c = label_color)
plt.title('Clustering mixture of Gaussian')
plt.savefig(r'7.png')
## deal with MNIST
df = pd.read_csv(r'mnist.csv')
x = df.values
K = 10
# PCA
from sklearn import preprocessing
from sklearn.decomposition import PCA
X_scaled = preprocessing.scale(x)
pca = PCA(n_components=50)
pca.fit(X_scaled)
x_pca = pca.transform(X_scaled)
# cluster
w_samples, mu_samples, lambda_samples, D, O = gibbs(x_pca, 2000, K = 10)
# precision
plt.plot(np.dot(D, np.arange(10)))
# mean of each cluster and corresponding image
plt.figure(figsize=(15,5))
plt.subplots_adjust(hspace=0.01)
for k in range(K):
    # average the original images
    #plt.imshow(np.mean(x[D[:, k].astype(bool),:], axis = 0).reshape(28,28))
    #plt.savefig(r'average_original'+str(k)+'.png')
    # average the pca
    plt.subplot(2,5,k+1)
    plt.imshow((pca.inverse_transform(np.mean(x_pca[D[:,k].astype(bool),:], axis = 0)) * np.std(x, axis = 0) + np.mean(x, axis = 0)).reshape(28,28))
    #plt.savefig(r'average_pca'+str(k)+'.png')
# co-occurrence matrix
#equal to sum np.dot(D, D.T)

import seaborn as sns; sns.set()
O[O>3000] = 0
O[O<0] = 0
ax = sns.heatmap(O)
plt.savefig(r'8occurrence.png')

## 17 with dirichlet 100
def gibbs(x, n, K):
    nx, p = x.shape
    burnin = 1000
    w_samples = 1/K * np.ones((n + burnin, K))
    mu_samples = np.zeros((K, n + burnin, p))
    lambda_samples = np.ones((K, n + burnin))
    D = np.zeros((nx,K))
    O = np.zeros((nx,nx))
    for i in range(1, n + burnin):
        # first update D
        # matrix probabilities
        prob = np.array([np.array(list(map(lambda r: normpdf(r, mu_samples[k,i-1,:], 1 / lambda_samples[k, i-1]), x))) for k in range(K)]).T * w_samples[i-1,:]
        prob = (prob.T / np.sum(prob, axis = 1)).T
        for j in range(nx):
            D[j,:] = np.random.multinomial(1, prob[j,:], size = 1)
        O += np.dot(D, D.T)
        # update w
        count = np.sum(D, axis = 0)
        w_samples[i,:] = np.random.dirichlet(0.01 + count)
        # update lambda and mu
        for k in range(K):
            nk = count[k]
            index = D[:, k].astype(bool)
            lambda_samples[k, i] = 1 #np.random.gamma(1 + nk/2, 1 / (1 + (np.sum(np.power(x[index,:] - mu_samples[k, i-1, :],2)) )/2))
            mu_samples[k, i, :] = np.random.multivariate_normal(np.sum(x[index], axis = 0)/(1/(100*lambda_samples[k, i]) + nk), np.identity(p) * (1/(1/100 + nk * lambda_samples[k,i])))
    return w_samples[burnin:, ], mu_samples[:,burnin:, ], lambda_samples[:, burnin: ], D, O


# cluster
K = 100
w_samples, mu_samples, lambda_samples, D, O = gibbs(x_pca, 2000, K)
# precision
plt.plot(np.dot(D, np.arange(K)))
# mean of each cluster and corresponding image
plt.figure(figsize=(15,15))
plt.subplots_adjust(hspace=0)
for k in range(K):
    # average the original images
    #plt.imshow(np.mean(x[D[:, k].astype(bool),:], axis = 0).reshape(28,28))
    #plt.savefig(r'average_original'+str(k)+'.png')
    # average the pca
    plt.subplot(10,10,k+1)
    plt.imshow((pca.inverse_transform(np.mean(x_pca[D[:,k].astype(bool),:], axis = 0)) * np.std(x, axis = 0) + np.mean(x, axis = 0)).reshape(28,28))
    #plt.savefig(r'average_pca'+str(k)+'.png')
# co-occurrence matrix
#equal to sum np.dot(D, D.T)

import seaborn as sns; sns.set()
O[O>3000] = 0
O[O<0] = 0
ax = sns.heatmap(O)
plt.savefig(r'17occurrence.png')


plt.hist(np.dot(D,np.arange(K)), bins = 100)










