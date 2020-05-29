import numpy as np
import pandas as pd
from sklearn.linear_model import SGDRegressor

# veri kumesini oku
verikumesi = pd.read_csv("ds2.txt",delimiter="\t")

X = verikumesi.iloc[:,:-1].values
y = verikumesi.iloc[:,X.shape[1]].values

# modeli tanimla
clf = SGDRegressor(penalty='none', verbose=1, max_iter=100000)
# modeli egit
clf.fit(X, y)

print(clf.intercept_, clf.coef_)
# Grad. Dc.: [ 0.14157558  1.91993045  2.99348001  0.98027489  1.94982636]
# Norm. Eq.: [ 0.12490622  2.06239085  2.99213354  0.98455834  2.02928992]