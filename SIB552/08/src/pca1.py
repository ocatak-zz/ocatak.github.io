# -*- coding: utf-8 -*-
"""
Created on Sun Apr 22 18:42:13 2018

@author: ozgurcatak
"""

import numpy as np, matplotlib.pyplot as plt
from sklearn.datasets import make_classification
import matplotlib

plt.style.use('ggplot')

np.random.seed(20) #make sure we're all working with the same numbers


X, y = make_classification(n_informative=1, n_clusters_per_class=1, n_samples=100)

print(X.shape)
print(y)
plt.scatter(X[:,0],X[:,1], c=[matplotlib.cm.spectral(float(i) /10) for i in y])
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.title('Raw Data')
plt.axis([-6,6,-30,30]);
plt.show()

plt.clf

import scipy.stats as stats

X = stats.mstats.zscore(X,axis=1)

plt.scatter(X[:,0],X[:,1], c=[matplotlib.cm.spectral(float(i) /10) for i in y])
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.title('Standardized Data')
plt.axis([-4,4,-4,4]);
plt.show()

plt.clf

C = np.dot(X,np.transpose(X))/(np.shape(X)[1]-1);
[V,PC] = np.linalg.eig(C)
print(PC.shape)

plt.scatter(X[:,0],X[:,1], c=[matplotlib.cm.spectral(float(i) /10) for i in y])
'''
plt.plot([0,PC[0,0]*V[0]],[0,PC[1,0]*V[0]],'o-')
plt.plot([0,PC[0,1]*V[1]],[0,PC[1,1]*V[1]],'o-')
'''

for i in range(10):
    plt.plot([0,PC[0,i]*V[i]],[0,PC[1,i]*V[i]],'o-')


plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.title('Standardized Data with Eigenvectors')
plt.axis([-4,4,-4,4]);
plt.show()

indices = np.argsort(-1*V)
V = V[indices]
PC = PC[indices,:]

X_rotated = np.dot(X.T,PC)

#plt.plot(X_rotated.T[:,0],X_rotated.T[:,1],'o')
plt.scatter(X_rotated.T[:,0],X_rotated.T[:,1], c=[matplotlib.cm.spectral(float(i) /10) for i in y])
plt.plot([0,PC[1,0]*V[0]],[0,0],'o-')
plt.plot([0,0],[0,PC[1,1]*V[1]],'o-')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.title('Data Projected into PC space')
plt.axis([-4,4,-4,4]);