import matplotlib.pyplot as plt
import numpy as np

from sklearn import neighbors
from sklearn.datasets import make_classification
from matplotlib.colors import ListedColormap
from scipy.spatial import Voronoi, voronoi_plot_2d

np.random.seed(10)
X, y = make_classification(n_features=2, n_redundant=0)

# f, axes = plt.subplots(2, 3, sharey=True, figsize=(20, 10))
f, axes = plt.subplots(2, 3, sharey=True, figsize=(20, 10))
cmap_light = ListedColormap(['#00AAAA', '#00FFAA', '#AAAAFF'])

'''
vor = Voronoi(X)
fig = voronoi_plot_2d(vor, show_vertices=True, show_points=True, line_colors='orange',line_width=0.5, line_alpha=0.6, point_size=5)
plt.show()
'''

## Decision boundary start
x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1

xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.05), np.arange(y_min, y_max, 0.05))


def plot_boundary(ax, k):
    clf = neighbors.KNeighborsClassifier(k, weights='distance')
    clf.fit(X, y)

    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    ax.pcolormesh(xx, yy, Z, cmap=cmap_light)
    print(y)
    ax.scatter(X[:, 0], X[:, 1], marker='o', c=y,s=25, edgecolor='k')
    ax.set_title("k = " + str(k))

plot_boundary(axes[0,0], 1)
plot_boundary(axes[0,1], 5)
plot_boundary(axes[0,2], 10)
plot_boundary(axes[1,0], 15)
plot_boundary(axes[1,1], 20)
plot_boundary(axes[1,2], 30)

plt.tight_layout()

plt.savefig("../img/knn_decision_boundary.eps", format="eps", dpi=100)
plt.show()