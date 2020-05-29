import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

# veri kumesini oku
verikumesi = pd.read_csv("ds_logreg.txt",delimiter="\t")

X = verikumesi.iloc[:,:-1].values
y = verikumesi.iloc[:,X.shape[1]].values

# modeli tanimla
clf = LogisticRegression(verbose=1)
# modeli egit
clf.fit(X, y)

print(clf.intercept_, clf.coef_)