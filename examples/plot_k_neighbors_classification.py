"""
K-nearest neighbors classification
==================================

Shows the usage of the k-nearest neighbors classifier.
"""

# Author: Pablo Marcos Manchón
# License: MIT

import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import GridSearchCV, train_test_split

import skfda
from skfda.ml.classification import KNeighborsClassifier

##############################################################################
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

X, y = skfda.datasets.fetch_growth(return_X_y=True, as_frame=True)
X = X.iloc[:, 0].values
y = y.values

# Plot samples grouped by sex
X.plot(group=y.codes, group_names=y.categories)

y = y.codes


##############################################################################
#
# The class labels are stored in an array. Zeros represent male
# samples while ones represent female samples.

print(y)

##############################################################################
#
# We can split the dataset using the sklearn function
# :func:`~sklearn.model_selection.train_test_split`.
#
# The function will return two
# :class:`~skfda.representation.grid.FDataGrid`'s, ``X_train`` and ``X_test``
# with the corresponding partitions, and arrays with their class labels.

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.25,
    stratify=y,
    random_state=0,
)


##############################################################################
#
# We will fit the classifier
# :class:`~skfda.ml.classification.KNeighborsClassifier`
# with the training partition. This classifier works exactly like the sklearn
# multivariate classifier
# :class:`~sklearn.neighbors.KNeighborsClassifier`, but it's input is
# a :class:`~skfda.representation.grid.FDataGrid` with
# functional observations instead of an array with multivariate data.

knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train, y_train)

##############################################################################
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

##############################################################################
#
# The :func:`~skfda.ml.classification.KNeighborsClassifier.score` method
# allows us to calculate the mean accuracy for the test data. In this case we
# obtained around 96% of accuracy.

score = knn.score(X_test, y_test)
print(score)

##############################################################################
#
# We can also estimate the probability of membership to the predicted class
# using :func:`~skfda.ml.classification.KNeighborsClassifier.predict_proba`,
# which will return an array with the probabilities of the classes, in
# lexicographic order, for each test sample.

probs = knn.predict_proba(X_test[:5])  # Predict first 5 samples
print(probs)


##############################################################################
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


##############################################################################
#
# We have obtained the greatest mean accuracy using 11 neighbors. The
# following figure shows the score depending on the number of neighbors.

fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)
ax.bar(param_grid["n_neighbors"], gscv.cv_results_["mean_test_score"])
ax.set_xticks(param_grid["n_neighbors"])
ax.set_ylabel("Number of Neighbors")
ax.set_xlabel("Cross-validation score")
ax.set_ylim((0.9, 1))

##############################################################################
#
# By default, after performing the cross validation, the classifier will
# be fitted to the whole training data provided in the call to
# :func:`~skfda.ml.classification.KNeighborsClassifier.fit`.
# Therefore, to check the accuracy of the classifier for the number of
# neighbors selected (11), we can simply call the
# :func:`~sklearn.model_selection.GridSearchCV.score` method.

score = gscv.score(X_test, y_test)
print(score)

##############################################################################
#
# When the functional data has been sampled in an equispaced way, or
# approximately equispaced, it is possible to use the scikit-learn vector
# metrics with similar results.
#
# For example, in the case of the :math:`\mathbb{L}^2` distance,
# by approximating the integral as a Riemann sum,
# we can derive that the value of said integral is proportional to the
# Euclidean distance between vectors.
#
# .. math::
#   \|f - g \|_{\mathbb{L}^2} =  \left ( \int_a^b |f(x) - g(x)|^2 dx \right )
#   ^{\frac{1}{2}} \approx \left ( \sum_{n=0}^{N}\bigtriangleup h \,|f(x_n)
#    - g(x_n)|^2 \right ) ^ {\frac{1}{2}}\\
#   = \sqrt{\bigtriangleup h} \, d_{euclidean}(\vec{f}, \vec{g})
#
#
# Therefore, in this case, it is roughly equivalent to use this metric instead
# of the functional one, since multiplying by a constant does not affect the
# order of the neighbors.
#
# By setting the parameter ``sklearn_metric`` of the classifier to ``True``,
# a vectorial metric of sklearn can be provided. The list of supported
# metrics can be found in :class:`~sklearn.neighbors.DistanceMetric`
#
# We will fit the model with the sklearn distance and search for the best
# parameter. The results can vary slightly, due to the approximation of
# the integral, but the result should be similar.

knn = KNeighborsClassifier(metric="euclidean", multivariate_metric=True)
gscv2 = GridSearchCV(knn, param_grid, cv=5)
gscv2.fit(X_train, y_train)

print("Best params:", gscv2.best_params_)
print("Best score:", gscv2.best_score_)

##############################################################################
#
# Using sklearn metrics results in a speedup of three orders of magnitude.
# However, it is not always possible to have equispaced sample and not all
# functional metrics have the vector equivalent required to do this
# approximation.
#
# The mean score time depending on the metric is shown below.

print("Mean score time (milliseconds)")
print(
    "L2 distance:{time}(ms)".format(
        time=1000 * np.mean(gscv.cv_results_["mean_score_time"]),
    ),
)

print(
    "Euclidean distance:{time}(ms)".format(
        time=1000 * np.mean(gscv2.cv_results_["mean_score_time"]),
    ),
)

##############################################################################
#
# This classifier can be used with multivariate functional data, as surfaces
# or curves in :math:`\mathbb{R}^N`, if the metric supports it too.
