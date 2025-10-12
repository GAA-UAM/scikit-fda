"""
Depth based classification
==========================

This example shows the use of the depth based classification methods
applied to the Berkeley Growth Study data. An attempt to show the
differences and similarities between
:class:`~skfda.ml.classification.MaximumDepthClassifier`,
:class:`~skfda.ml.classification.DDClassifier`,
and :class:`~skfda.ml.classification.DDGClassifier` is made.
"""

# Author: Pedro Martín Rodríguez-Ponga Eyriès
# License: MIT

# sphinx_gallery_thumbnail_number = 5

# %%
# The Berkeley Growth Study data contains the heights of 39 boys and 54 girls
# from age 1 to 18 and the ages at which they were collected. Males are
# assigned the numeric value 0 while females are assigned a 1. In our
# comparison of the different methods, we will try to learn the sex of a person
# by using its growth curve.

from skfda.datasets import fetch_growth

X_df, y_df = fetch_growth(return_X_y=True, as_frame=True)
X = X_df.iloc[:, 0].array
target = y_df.array
# sphinx_gallery_start_ignore
from pandas import Categorical

from skfda import FDataGrid

assert isinstance(X, FDataGrid)
assert isinstance(target, Categorical)
# sphinx_gallery_end_ignore
categories = target.categories
y = target.codes

# %%
# As in many ML algorithms, we split the dataset into train and test. In this
# graph, we can see the training dataset. These growth curves will be used to
# train the model. Hence, the predictions will be data-driven.

import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.5,
    stratify=y,
    random_state=0,
)

# Plot samples grouped by sex
X_train.plot(group=y_train, group_names=categories)
plt.show()

# %%
# Below are the growth graphs of those individuals that we would like to
# classify. Some of them will be male and some female.
X_test.plot()
plt.show()

# %%
# As said above, we are trying to compare three different methods:
# :class:`~skfda.ml.classification.MaximumDepthClassifier`,
# :class:`~skfda.ml.classification.DDClassifier`, and
# :class:`~skfda.ml.classification.DDGClassifier`. They all use a
# depth which in our example is
# :class:`~skfda.exploratory.depth.ModifiedBandDepth` for consistency. With
# this depth we can create a :class:`~skfda.exploratory.visualization.DDPlot`.
#
# In a :class:`~skfda.exploratory.visualization.DDPlot`, a growth curve is
# mapped to :math:`[0,1]\times[0,1]` where the first coordinate corresponds
# to the depth in the class of all boys and the second to that of all girls.
# Note that the dots will be blue if the true sex is female and red otherwise.

# %%
# Below we can see how a :class:`~skfda.exploratory.visualization.DDPlot` is
# used to classify with
# :class:`~skfda.ml.classification.MaximumDepthClassifier`. In this case it is
# quite straighforward, a person is classified to the class where it is
# deeper. This means that if a point is above the diagonal it is a girl and
# otherwise it is a boy.

import numpy as np
from matplotlib.colors import ListedColormap

from skfda.exploratory.depth import ModifiedBandDepth
from skfda.exploratory.visualization import DDPlot
from skfda.ml.classification import MaximumDepthClassifier

# sphinx_gallery_start_ignore
# isort: split
from numpy import integer
from numpy.typing import NDArray

NDArrayInt = NDArray[integer]

max_depth_classifier: MaximumDepthClassifier[FDataGrid, NDArrayInt]
# sphinx_gallery_end_ignore
max_depth_classifier = MaximumDepthClassifier(depth_method=ModifiedBandDepth())
max_depth_classifier.fit(X_train, y_train)
print(max_depth_classifier.predict(X_test))
print(f"The score is {max_depth_classifier.score(X_test, y_test):2.2%}")

fig, ax = plt.subplots()

cmap_bold = ListedColormap(["#FF0000", "#0000FF"])

index = y_train.astype(bool)
DDPlot(
    fdata=X_test,
    dist1=X_train[np.invert(index)],
    dist2=X_train[index],
    depth_method=ModifiedBandDepth(),
    axes=ax,
    c=y_test,
    cmap_bold=cmap_bold,
    x_label="Boy class depth",
    y_label="Girl class depth",
).plot()
plt.show()

# %%
# We can see that we have used the classification predictions to compute the
# score (obtained by comparing to the real known sex). This will also be done
# for the rest of the classifiers.

# %%
# Next we use :class:`~skfda.ml.classification.DDClassifier` with polynomes
# of degrees one, two, and three. Here, if a point in the
# :class:`~skfda.exploratory.visualization.DDPlot` is above the polynome,
# the classifier will predict that it is a girl and otherwise, a boy.

from skfda.ml.classification import DDClassifier

# sphinx_gallery_start_ignore
dd1_classifier: DDClassifier[FDataGrid, NDArrayInt]
# sphinx_gallery_end_ignore
dd1_classifier = DDClassifier(degree=1, depth_method=ModifiedBandDepth())
dd1_classifier.fit(X_train, y_train)
print(dd1_classifier.predict(X_test))
print(f"The score is {dd1_classifier.score(X_test, y_test):2.2%}")

# %%

# sphinx_gallery_start_ignore
dd2_classifier: DDClassifier[FDataGrid, NDArrayInt]
# sphinx_gallery_end_ignore
dd2_classifier = DDClassifier(degree=2, depth_method=ModifiedBandDepth())
dd2_classifier.fit(X_train, y_train)
print(dd2_classifier.predict(X_test))
print(f"The score is {dd2_classifier.score(X_test, y_test):2.2%}")

# %%

# sphinx_gallery_start_ignore
dd3_classifier: DDClassifier[FDataGrid, NDArrayInt]
# sphinx_gallery_end_ignore
dd3_classifier = DDClassifier(degree=3, depth_method=ModifiedBandDepth())
dd3_classifier.fit(X_train, y_train)
print(dd3_classifier.predict(X_test))
print(f"The score is {dd3_classifier.score(X_test, y_test):2.2%}")

# %%
from matplotlib.axes import Axes


def plot_boundaries(ax: Axes) -> None:
    """Plot the boundaries of the DD classifier with different degrees."""
    margin = 0.025
    ts = np.linspace(- margin, 1 + margin, 100)
    pol1 = ax.plot(
        ts,
        np.polyval(dd1_classifier.poly_, ts),
        "c",
        label="Polynomial",
    )[0]
    pol2 = ax.plot(
        ts,
        np.polyval(dd2_classifier.poly_, ts),
        "m",
        label="Polynomial",
    )[0]
    pol3 = ax.plot(
        ts,
        np.polyval(dd3_classifier.poly_, ts),
        "g",
        label="Polynomial",
    )[0]
    max_depth = ax.plot(
        [0, 1],
        color="gray",
    )[0]

    ax.legend([pol1, pol2, pol3, max_depth], ["P1", "P2", "P3", "MaxDepth"])

fig, ax = plt.subplots()
plot_boundaries(ax)

DDPlot(
    fdata=X_test,
    dist1=X_train[np.invert(index)],
    dist2=X_train[index],
    depth_method=ModifiedBandDepth(),
    axes=ax,
    c=y_test,
    cmap_bold=cmap_bold,
    x_label="Boy class depth",
    y_label="Girl class depth",
).plot()
plt.show()

# %%
# :class:`~skfda.ml.classification.DDClassifier` used with
# :class:`~sklearn.neighbors.KNeighborsClassifier`.

from sklearn.neighbors import KNeighborsClassifier

from skfda.ml.classification import DDGClassifier

# sphinx_gallery_start_ignore
ddg_classifier: DDClassifier[FDataGrid, NDArrayInt]
# sphinx_gallery_end_ignore
ddg_classifier = DDGClassifier(
    depth_method=ModifiedBandDepth(),
    multivariate_classifier=KNeighborsClassifier(n_neighbors=5),
)
ddg_classifier.fit(X_train, y_train)
print(ddg_classifier.predict(X_test))
print(f"The score is {ddg_classifier.score(X_test, y_test):2.2%}")


# %%
# The other elements of the graph are the decision boundaries:
#
# +--------------+--------------------------------------+
# | Boundary     | Classifier                           |
# +==============+======================================+
# | MaxDepth     | MaximumDepthClassifier               |
# +--------------+--------------------------------------+
# | P1           | DDClassifier with degree 1           |
# +--------------+--------------------------------------+
# | P2           | DDClassifier with degree 2           |
# +--------------+--------------------------------------+
# | P3           | DDClassifier with degree 3           |
# +--------------+--------------------------------------+
# | NearestClass | DDGClassifier with nearest neighbors |
# +--------------+--------------------------------------+

from skfda.preprocessing.feature_construction import PerClassTransformer

# sphinx_gallery_start_ignore
# isort: split
from numpy import floating

NDArrayFloat = NDArray[floating]

pct: PerClassTransformer[FDataGrid, NDArrayFloat]
# sphinx_gallery_end_ignore
pct = PerClassTransformer(ModifiedBandDepth(), array_output=True)
X_train_trans = pct.fit_transform(X_train, y_train)
X_train_trans = X_train_trans.reshape(len(categories), X_train.shape[0]).T

knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train_trans, y_train)

h = 0.01  # step size in the mesh

# Create color maps
cmap_light = ListedColormap(["#FFAAAA", "#AAAAFF"])

# Plot the decision boundary. For that, we will assign a color to each
# point in the mesh [x_min, x_max]x[y_min, y_max].
x_min, x_max = X_train_trans[:, 0].min() - 1, X_train_trans[:, 0].max() + 1
y_min, y_max = X_train_trans[:, 1].min() - 1, X_train_trans[:, 1].max() + 1
xx, yy = np.meshgrid(
    np.arange(x_min, x_max, h),
    np.arange(y_min, y_max, h),
)
Z = knn.predict(np.c_[xx.ravel(), yy.ravel()])

# Put the result into a color plot
Z = Z.reshape(xx.shape)

fig, ax = plt.subplots()
ax.pcolormesh(xx, yy, Z, cmap=cmap_light, shading="auto")

plot_boundaries(ax)
DDPlot(
    fdata=X_test,
    dist1=X_train[np.invert(index)],
    dist2=X_train[index],
    depth_method=ModifiedBandDepth(),
    axes=ax,
    c=y_test,
    cmap_bold=cmap_bold,
    x_label="Boy class depth",
    y_label="Girl class depth",
).plot()
plt.show()

# %%
# In the above graph, we can see the obtained classifiers from the train set.
# The dots are all part of the test set and have their real color so, for
# example, if they are blue it means that the true sex is female. One can see
# that none of the built classifiers is perfect.
#
# Next, we will use :class:`~skfda.ml.classification.DDGClassifier` together
# with a neural network: :class:`~sklearn.neural_network.MLPClassifier`.

from sklearn.neural_network import MLPClassifier

ddg_classifier = DDGClassifier(
    depth_method=ModifiedBandDepth(),
    multivariate_classifier=MLPClassifier(
        solver="lbfgs",
        alpha=1e-5,
        hidden_layer_sizes=(6, 2),
        random_state=1,
    ),
)
ddg_classifier.fit(X_train, y_train)
print(ddg_classifier.predict(X_test))
print(f"The score is {ddg_classifier.score(X_test, y_test):2.2%}")

# %%
knn = KNeighborsClassifier(n_neighbors=5)
mlp_classifier = MLPClassifier(
    solver="lbfgs",
    alpha=1e-5,
    hidden_layer_sizes=(6, 2),
    random_state=1,
)
knn.fit(X_train_trans, y_train)
mlp_classifier.fit(X_train_trans, y_train)

Z1 = knn.predict(np.c_[xx.ravel(), yy.ravel()])
Z2 = mlp_classifier.predict(np.c_[xx.ravel(), yy.ravel()])

Z1 = Z1.reshape(xx.shape)
Z2 = Z2.reshape(xx.shape)

fig, axes = plt.subplots(1, 2, sharex=True, sharey=True)

axes[0].pcolormesh(xx, yy, Z1, cmap=cmap_light, shading="auto")
axes[1].pcolormesh(xx, yy, Z2, cmap=cmap_light, shading="auto")

DDPlot(
    fdata=X_test,
    dist1=X_train[np.invert(index)],
    dist2=X_train[index],
    depth_method=ModifiedBandDepth(),
    axes=axes[0],
    c=y_test,
    cmap_bold=cmap_bold,
    x_label="Boy class depth",
    y_label="Girl class depth",
).plot()

DDPlot(
    fdata=X_test,
    dist1=X_train[np.invert(index)],
    dist2=X_train[index],
    depth_method=ModifiedBandDepth(),
    axes=axes[1],
    c=y_test,
    cmap_bold=cmap_bold,
    x_label="Boy class depth",
    y_label="Girl class depth",
).plot()

for ax in axes:
    ax.label_outer()

plt.show()

# %%
# We can compare the behavior of two
# :class:`~skfda.ml.classification.DDGClassifier` based classifiers. The
# one on the left corresponds to nearest neighbors and the one on the right to
# a neural network. Interestingly, the neural network almost coincides with
# :class:`~skfda.ml.classification.MaximumDepthClassifier`.
