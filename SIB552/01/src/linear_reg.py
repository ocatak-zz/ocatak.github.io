import pandas as pd
import matplotlib.pyplot as plt

# veri kumesini oku
verikumesi = pd.read_csv("ds1.txt",delimiter="\t")

X = verikumesi.iloc[:,:-1].values
y = verikumesi.iloc[:,1].values

# veri kumesini egitim ve test olarak parcala
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

# dogrusal regresyon modeli
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, y_train)

# tahmin
y_pred = regressor.predict(X_test)

# veri gorsellestirme
plt.scatter(X_train, y_train, color='red')
plt.plot(X_test, y_pred, color='blue')
plt.show()