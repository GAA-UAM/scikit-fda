"""
K-nearest neighbors classification
==================================

Shows the usage of the k-nearest neighbors classifier.
"""

# Author: Pablo Marcos Manchón
# License: MIT

# %%
#
# In this example we are going to show the usage of the K-nearest neighbors
# classifier in their functional version, which is a extension of the
# multivariate one, but using functional metrics.
#
# Firstly, we are going to fetch a functional dataset, such as the Berkeley
# Growth Study. This dataset contains the height of several boys and girls
# measured until the 18 years of age.
# We will try to predict sex from their growth curves.
#
# The following figure shows the growth curves grouped by sex.

import matplotlib.pyplot as plt

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
y = target.codes

# Plot samples grouped by sex
X.plot(group=target.codes, group_names=target.categories)
plt.show()

# %%
#
# The class labels are stored in an array. Zeros represent male
# samples while ones represent female samples.

print(y)

# %%
#
# We can split the dataset using the sklearn function
# :func:`~sklearn.model_selection.train_test_split`.
#
# The function will return two
# :class:`~skfda.representation.grid.FDataGrid`'s, ``X_train`` and ``X_test``
# with the corresponding partitions, and arrays with their class labels.

from sklearn.model_selection import GridSearchCV, train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.25,
    stratify=y,
    random_state=0,
)


# %%
#
# We will fit the classifier
# :class:`~skfda.ml.classification.KNeighborsClassifier`
# with the training partition. This classifier works exactly like the sklearn
# multivariate classifier
# :class:`~sklearn.neighbors.KNeighborsClassifier`, but it's input is
# a :class:`~skfda.representation.grid.FDataGrid` with
# functional observations instead of an array with multivariate data.

from skfda.ml.classification import KNeighborsClassifier

knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train, y_train)

# %%
#
# Once it is fitted, we can predict labels for the test samples.
#
# To predict the label of a test sample, the classifier will calculate the
# k-nearest neighbors and will assign the class shared by most of those k
# neighbors. In this case, we have set the number of neighbors to 5
# (:math:`k=5`).
# By default, it will use the
# :math:`\mathbb{L}^2` distance between functions, to determine the
# neighborhood of a sample. However, it can be used with
# any of the functional metrics described in
# :doc:`/modules/misc/metrics`.

pred = knn.predict(X_test)
print(pred)

# %%
#
# The :func:`~skfda.ml.classification.KNeighborsClassifier.score` method
# allows us to calculate the mean accuracy for the test data. In this case we
# obtained around 96% of accuracy.

score = knn.score(X_test, y_test)
print(score)

# %%
#
# We can also estimate the probability of membership to the predicted class
# using :func:`~skfda.ml.classification.KNeighborsClassifier.predict_proba`,
# which will return an array with the probabilities of the classes, in
# lexicographic order, for each test sample.

probs = knn.predict_proba(X_test[:5])  # Predict first 5 samples
print(probs)


# %%
#
# We can use the sklearn
# :class:`~sklearn.model_selection.GridSearchCV` to perform a
# grid search to select the best hyperparams, using cross-validation.
#
# In this case, we will vary the number of neighbors between 1 and 17.

# Only odd numbers, to prevent ties
param_grid = {"n_neighbors": range(1, 18, 2)}


knn = KNeighborsClassifier()

# Perform grid search with cross-validation
gscv = GridSearchCV(knn, param_grid, cv=5)
gscv.fit(X_train, y_train)


print("Best params:", gscv.best_params_)
print("Best cross-validation score:", gscv.best_score_)


# %%
#
# We have obtained the greatest mean accuracy using 11 neighbors. The
# following figure shows the score depending on the number of neighbors.

fig, ax = plt.subplots()
ax.bar(param_grid["n_neighbors"], gscv.cv_results_["mean_test_score"])
ax.set_xticks(param_grid["n_neighbors"])
ax.set_ylabel("Number of Neighbors")
ax.set_xlabel("Cross-validation score")
ax.set_ylim((0.9, 1))

plt.show()

# %%
#
# By default, after performing the cross validation, the classifier will
# be fitted to the whole training data provided in the call to
# :func:`~skfda.ml.classification.KNeighborsClassifier.fit`.
# Therefore, to check the accuracy of the classifier for the number of
# neighbors selected (11), we can simply call the
# :func:`~sklearn.model_selection.GridSearchCV.score` method.

score = gscv.score(X_test, y_test)
print(score)

# %%
#
# This classifier can be used with multivariate functional data, as surfaces
# or curves in :math:`\mathbb{R}^N`, if the metric supports it too.
