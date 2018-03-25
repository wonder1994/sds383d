import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import time
from scipy.optimize import minimize

df = pd.read_csv(r'/home/xinjie/Dropbox/courses/2018_Spring/STAT_MODEL/sds383d/R/titanic.csv')
df = df.drop(['Name','PClass','Sex'], axis = 1)
idx = df.iloc[:, 1] == 'Yes'
data = df.values
data[idx, 1] = 1
data[~idx, 1] = 0
data = data.astype(float)
data = data[~np.isnan(data).any(axis = 1),:]
x = data[:,0]
y = data[:,1]
def obj(beta, x, y):
    y = y.astype(bool)
    v1 = np.log(1 + np.exp(- x[y] * beta))
    v2 = np.log(1 + np.exp(x[y] * beta))
    return np.sum(v1)+np.sum(v2)-0.5*(beta)**2

res = minimize(obj, x0 = 1, args = (x,y), method = 'Nelder-Mead', tol = 1e-8)

beta = pd.Series(np.arange(-10,10,0.1))
obj_array = list(map(lambda b: obj(b, x, y), beta))
plt.plot(beta, obj_array)