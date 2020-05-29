import matplotlib.pyplot as plt
import numpy as np

from sklearn import neighbors
from sklearn.datasets import make_classification
from matplotlib.colors import ListedColormap
from scipy.spatial import Voronoi, voronoi_plot_2d

np.random.seed(10)
X, y = make_classification(n_features=2, n_redundant=0)

f, (ax1, ax2) = plt.subplots(1, 2, sharey=True)
cmap_light = ListedColormap(['#00AAAA', '#00FFAA', '#AAAAFF'])

'''
vor = Voronoi(X)
fig = voronoi_plot_2d(vor, show_vertices=True, show_points=True, line_colors='orange',line_width=0.5, line_alpha=0.6, point_size=5)
plt.show()
'''

## Decision boundary start
x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1

xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.01), np.arange(y_min, y_max, 0.01))

clf = neighbors.KNeighborsClassifier(15, weights='distance')
clf.fit(X, y)

Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
# Put the result into a color plot
Z = Z.reshape(xx.shape)
ax2.pcolormesh(xx, yy, Z, cmap=cmap_light)
## Decision boundary end

ax1.scatter(X[:, 0], X[:, 1], marker='o', c=y,s=25, edgecolor='k')
ax1.scatter(0.7, -0.2, marker='x',s=100, edgecolor='k')
ax2.scatter(0.7, -0.2, marker='x',s=100, edgecolor='k')

vor = Voronoi(X)
fig = voronoi_plot_2d(vor, ax=ax2, show_vertices=True, show_points=True, line_colors='orange',line_width=0.5, line_alpha=0.6, point_size=5)


plt.tight_layout()
plt.show()