import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import truncnorm
import time 
def sampler1(X, y, Lambda, K, a, b, mu, nsamples, burnin):
    n, p = data.shape
    p = p - 1
    #n_positive = np.sum(y)
    beta = np.zeros((nsamples + burnin, p))
    omega = np.zeros(nsamples + burnin)
    omega.shape = (nsamples + burnin, 1)
    z = np.random.normal(size = n)
    z.shape = [n, 1]
    z[y] = abs(z[y])
    z[~y] = - abs(z[~y])
    Cov_n = np.identity(p) - np.transpose(X)@np.linalg.inv((np.identity(n) + X@np.transpose(X)))@X
    for i in range(nsamples + burnin):
        # first sample omega 
        a_n = a + n/2
        b_n = b + (np.transpose(mu)@K@(mu) + np.transpose(z)@Lambda@z- np.transpose(K@mu+np.transpose(X)@Lambda@z)@Cov_n@(K@mu + np.transpose(X)@Lambda@z))/2 
        omega[i] = np.random.gamma(shape = a_n, scale = 1/b_n)      
        mu_n = Cov_n@(K@mu + np.transpose(X)@Lambda@z)
        beta[i, :] = np.random.multivariate_normal(mu_n[:, 0], Cov_n/omega[i])
        norm_mean = np.dot(X, beta[i,:])
        #norm_cov = 1/omega[i]*np.identity(n)
        r = truncnorm.rvs(-norm_mean * np.sqrt(omega[i]), np.ones(n)*np.inf) / np.sqrt(omega[i]) + norm_mean
        r.shape = [n, 1]
        z[y] = r[y]
        r = truncnorm.rvs(-np.ones(n)*np.inf, -norm_mean * np.sqrt(omega[i])) / np.sqrt(omega[i]) + norm_mean
        r.shape = [n, 1]
        z[~y] = r[~y]
    return beta, omega

# data preprocessing
df = pd.read_csv(r'/home/xinjie/Dropbox/courses/2018_Spring/STAT_MODEL/sds383d/R/pima.csv')
[nrow, ncol] = df.shape
data = df.values
X_0 = np.ones((nrow,1))
data = np.hstack((X_0, data))
y = data[:, ncol].astype(bool)
X = data[:, 0 : ncol]
# set priors
Lambda = np.identity(nrow)
K = np.identity(ncol)
nsamples = 10000
a = 1
b = 1
tau = 1
mu = np.zeros(ncol)
mu.shape = (ncol, 1)
burnin = 1000
# sampling
start_time = time.time()
beta1, omega1 = sampler1(X, y, Lambda, K, a, b, mu, nsamples, burnin)
elapsed_time = time.time() - start_time

# plots
plt.plot(np.arange(nsamples + 1000), omega1)
plt.plot(np.arange(nsamples + 1000), beta1[:,0])
plt.plot(np.arange(nsamples + 1000), beta1[:,1])
plt.plot(np.arange(nsamples + 1000), beta1[:,2])
plt.plot(np.arange(nsamples + 1000), beta1[:,3])
plt.plot(np.arange(nsamples + 1000), beta1[:,4])
plt.plot(np.arange(nsamples + 1000), beta1[:,5])
plt.plot(np.arange(nsamples + 1000), beta1[:,6])
plt.plot(np.arange(nsamples + 1000), beta1[:,7])
plt.plot(np.arange(nsamples + 1000), beta1[:,8])

beta_estimate = np.average(beta1[burnin:, :], axis = 0)
predict = np.dot(X, beta_estimate) > 0.5
# correct ration
accuracy = np.sum(predict == y)/nrow













