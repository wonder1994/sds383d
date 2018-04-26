#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 23 18:21:52 2018

@author: xinjie
"""

import os
os.chdir(r'/home/xinjie/Dropbox/courses/2018_Spring/STAT_MODEL/sds383d/code/ex5/')
from sklearn import preprocessing
import random 
random.seed(1)
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
# data generation
K = 100
alpha0 = 1
alpha = alpha0*np.ones(K)
sample = np.zeros((5,10))
sample = sample.astype(int)
for i in range(5):
    pi = np.random.dirichlet(alpha)
    sample[i] = np.dot(np.random.multinomial(1, pi, 10), np.arange(K)).astype(int)
pi = np.random.dirichlet(alpha)
sample_long = np.dot(np.random.multinomial(1, pi, 10000), np.arange(K)).astype(int)
#plt.hist(sample_long)

freq = sorted([list(sample_long).count(k) for k in range(K)])
plt.plot(freq)
plt.title('frequency for $alpha$ =' + str(alpha0))
plt.savefig(r'alpha='+str(alpha0)+'.png')


df = pd.DataFrame(data = sample)
with open('mytable.tex','w') as tf:
    tf.write(df.to_latex())
    
