import matplotlib.pyplot as plt
import numpy as np

from sklearn import neighbors
from sklearn.datasets import make_classification
from matplotlib.colors import ListedColormap
import matplotlib.patches as patches

import matplotlib.cm as cm

np.random.seed(10)
X, y = make_classification(n_features=2, n_redundant=0)

clf = neighbors.KNeighborsClassifier(15, weights='distance')
clf.fit(X, y)

f, (ax1) = plt.subplots(1, 1, sharey=True)

ax1.scatter(X[:, 0], X[:, 1], c=y,s=25, edgecolor='k', cmap='Greys')
ax1.scatter(0.7, -0.2, marker='x',s=100, edgecolor='k')

ax1.add_patch(
    patches.Circle(
        (0.7, -0.2),0.5,
        linestyle='dashed',
        fill=False      # remove background
    )
)

ax1.add_patch(
    patches.Circle(
        (0.7, -0.2),0.7,
        linestyle='dashed',
        fill=False      # remove background
    )
)

#plt.text(0.7, 0.0, '?', fontsize=12)

plt.text(0.7, 0.15, "?", size=20, rotation=0.,
         ha="center", va="center",
         bbox=dict(boxstyle="round",
                   ec=(1., 0.5, 0.5),
                   fc=(1., 0.8, 0.8),
                   )
         )

plt.tight_layout()

plt.savefig("../img/knn.eps", format="eps")

plt.show()