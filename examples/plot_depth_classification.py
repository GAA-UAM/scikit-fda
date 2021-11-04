"""
Classification
==============

This example shows the use of the depth based classifications methods
applied to the Berkeley Growth Study data. An attempt to show the
differences and similarities between MaximumDepthClassifier,
DDClassifier, and DDGClassifier is made.
"""

# Author: Pedro Martín Rodríguez-Ponga Eyriès
# License: MIT

# sphinx_gallery_thumbnail_number = 3

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

from skfda import datasets
from skfda.exploratory.depth import ModifiedBandDepth
from skfda.exploratory.visualization import DDPlot
from skfda.ml.classification import (
    DDClassifier,
    DDGClassifier,
    MaximumDepthClassifier,
)
from skfda.preprocessing.dim_reduction.feature_extraction import DDGTransformer
from skfda.representation.grid import FDataGrid

##############################################################################
#
# The Berkeley Growth Study data contains the heights of 39 boys and 54
# girls from age 1 to 18 and the ages at which they were collected. Males
# are assigned the numeric value 0 while females are coded to a 1. In our
# comparison of the different methods, we will try to learn the sex of a
# person by using its growth curve.
X, y = datasets.fetch_growth(return_X_y=True, as_frame=True)
X = X.iloc[:, 0].values
categories = y.values.categories
y = y.values.codes

##############################################################################
#
# As in many ML algorithms, we split the dataset into train and test. In
# this graph, we can see the training dataset. These growth curves will
# be used to train the model. Hence, the predictions will be data-driven.
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.5,
    stratify=y,
    random_state=0,
)

# Plot samples grouped by sex
X_train.plot(group=y_train, group_names=categories)

##############################################################################
#
# Below are the growth graphs of those individuals that we would like to
# classify. Some of them will be male and some female.
X_test.plot()

##############################################################################
#
# As said above, we are trying to compare three different methods:
# MaximumDepthClassifier, DDClassifier, and DDGClassifier.
# Below are the classification predictions of these models as well as the
# score (obtained by comparing to the real known sex). For the three
# algorithms we will be using the depth
# :class:`~skfda.representation.depth.ModifiedBandDepth` for consistency.
# We will try polynomes of degrees one, two, and three for DDClassifier.
# DDClassifier will be used with
# :class:`~sklearn.neighbors.KNeighborsClassifier`.
clf = MaximumDepthClassifier(depth_method=ModifiedBandDepth())
clf.fit(X_train, y_train)
print(clf.predict(X_test))
print(clf.score(X_test, y_test))

clf1 = DDClassifier(degree=1, depth_method=ModifiedBandDepth())
clf1.fit(X_train, y_train)
print(clf1.predict(X_test))
print(clf1.score(X_test, y_test))

clf2 = DDClassifier(degree=2, depth_method=ModifiedBandDepth())
clf2.fit(X_train, y_train)
print(clf2.predict(X_test))
print(clf2.score(X_test, y_test))

clf3 = DDClassifier(degree=3, depth_method=ModifiedBandDepth())
clf3.fit(X_train, y_train)
print(clf3.predict(X_test))
print(clf3.score(X_test, y_test))


clf = DDGClassifier(
    KNeighborsClassifier(n_neighbors=5),
    depth_method=ModifiedBandDepth(),
)
clf.fit(X_train, y_train)
print(clf.predict(X_test))
clf.score(X_test, y_test)

##############################################################################
#
# Finally, we plot all these classifiers in a DDPlot. There is a
# one-to-one correspondence between growth curves and data points. The
# coordinates of the points in the graph correspond to the respective
# depth to the class of all boys and the class of all girls. Note that
# the dots are blue if the true sex is female and red otherwise. The
# other elements of the graph are the decision boundaries:
#
# | Boundary  | Classifier                           |
# | --------- | ------------------------------------ |
# | Gray line | MaximumDepthClassifier               |
# | P1        | DDClassifier with degree 1           |
# | P2        | DDClassifier with degree 2           |
# | P3        | DDClassifier with degree 3           |
# | Colors    | DDGClassifier with nearest neighbors |
ddg: DDGTransformer[FDataGrid] = DDGTransformer(
    depth_method=ModifiedBandDepth(),
)
X_train_trans = ddg.fit_transform(X_train, y_train)

# Code adapted from:
# https://stackoverflow.com/questions/45075638/graph-k-nn-decision-boundaries-in-matplotlib
clf = KNeighborsClassifier(n_neighbors=5)
clf.fit(X_train_trans, y_train)

h = 0.02  # step size in the mesh

# Create color maps
cmap_light = ListedColormap(['#FFAAAA', '#AAAAFF'])
cmap_bold = ListedColormap(['#FF0000', '#0000FF'])


# Plot the decision boundary. For that, we will assign a color to each
# point in the mesh [x_min, x_max]x[y_min, y_max].
x_min, x_max = X_train_trans[:, 0].min() - 1, X_train_trans[:, 0].max() + 1
y_min, y_max = X_train_trans[:, 1].min() - 1, X_train_trans[:, 1].max() + 1
xx, yy = np.meshgrid(
    np.arange(x_min, x_max, h),
    np.arange(y_min, y_max, h),
)
Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])

# Put the result into a color plot
Z = Z.reshape(xx.shape)

fig, ax = plt.subplots()

margin = 0.025
ts = np.linspace(- margin, 1 + margin, 100)
pol1 = ax.plot(
    ts,
    np.polyval(clf1.poly_, ts),
    'c',
    linewidth=1,
    label="Polynomial",
)
pol2 = ax.plot(
    ts,
    np.polyval(clf2.poly_, ts),
    'm',
    linewidth=1,
    label="Polynomial",
)
pol3 = ax.plot(
    ts,
    np.polyval(clf3.poly_, ts),
    'g',
    linewidth=1,
    label="Polynomial",
)
ax.pcolormesh(xx, yy, Z, cmap=cmap_light, shading='auto')

ax.legend([pol1, pol2, pol3], ['P1', 'P2', 'P3'])


index = y_train.astype(bool)
ddp = DDPlot(
    fdata=X_test,
    dist1=X_train[np.invert(index)],
    dist2=X_train[index],
    depth_method=ModifiedBandDepth(),
    axes=ax,
    c=y_test,
    cmap_bold=cmap_bold,
    x_label="Boy class depth",
    y_label="Girl class depth",
)
ddp.plot()
