"""
Neighbors Scalar Regression
===========================

Shows the usage of the nearest neighbors regressor with scalar response.
"""

# Author: Pablo Marcos Manchón
# License: MIT

# sphinx_gallery_thumbnail_number = 3

import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import GridSearchCV, train_test_split

import skfda
from skfda.ml.regression import KNeighborsRegressor

##############################################################################
#
# In this example, we are going to show the usage of the nearest neighbors
# regressors with scalar response. There is available a K-nn version,
# :class:`KNeighborsRegressor
# <skfda.ml.regression.KNeighborsRegressor>`, and other one based in the
# radius, :class:`RadiusNeighborsRegressor
# <skfda.ml.regression.RadiusNeighborsRegressor>`.
#
# Firstly we will fetch a dataset to show the basic usage.
#
# The Canadian weather dataset contains the daily temperature and
# precipitation at 35 different locations in Canada averaged over 1960 to
# 1994.
#
# The following figure shows the different temperature and precipitation
# curves.

data = skfda.datasets.fetch_weather()
fd = data['data']


# Split dataset, temperatures and curves of precipitation
X, y_func = fd.coordinates

##############################################################################
# Temperatures

X.plot()

##############################################################################
# Precipitation

y_func.plot()

##############################################################################
#
# We will try to predict the total log precipitation, i.e,
# :math:`logPrecTot_i = \log \sum_{t=0}^{365} prec_i(t)` using the temperature
# curves.


# Sum directly from the data matrix
prec = y_func.data_matrix.sum(axis=1)[:, 0]
log_prec = np.log(prec)

print(log_prec)

##############################################################################
#
# As in the nearest neighbors classifier examples, we will split the dataset
# in two partitions, for training and test, using the sklearn function
# :func:`~sklearn.model_selection.train_test_split`.

X_train, X_test, y_train, y_test = train_test_split(
    X,
    log_prec,
    random_state=7,
)

##############################################################################
#
# Firstly we will try make a prediction with the default values of the
# estimator, using 5 neighbors and the :math:`\mathbb{L}^2` distance.
#
# We can fit the :class:`~skfda.ml.regression.KNeighborsRegressor` in the
# same way than the
# sklearn estimators. This estimator is an extension of the sklearn
# :class:`~sklearn.neighbors.KNeighborsRegressor`, but accepting a
# :class:`~skfda.representation.grid.FDataGrid` as input instead of an array
# with multivariate data.

knn = KNeighborsRegressor(weights='distance')
knn.fit(X_train, y_train)

##############################################################################
#
# We can predict values for the test partition using
# :meth:`~skfda.ml.regression.KNeighborsScalarRegressor.predict`.


pred = knn.predict(X_test)
print(pred)

##############################################################################
#
# The following figure compares the real precipitations with the predicted
# values.


fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)
ax.scatter(y_test, pred)
ax.plot(y_test, y_test)
ax.set_xlabel("Total log precipitation")
ax.set_ylabel("Prediction")


##############################################################################
#
# We can quantify how much variability it is explained by the model with
# the coefficient of determination :math:`R^2` of the prediction,
# using :meth:`~skfda.ml.regression.KNeighborsScalarRegressor.score` for that.
#
# The coefficient :math:`R^2` is defined as :math:`(1 - u/v)`, where :math:`u`
# is the residual sum of squares :math:`\sum_i (y_i - y_{pred_i})^ 2`
# and :math:`v` is the total sum of squares :math:`\sum_i (y_i - \bar y )^2`.


score = knn.score(X_test, y_test)
print(score)


##############################################################################
#
# In this case, we obtain a really good aproximation with this naive approach,
# although, due to the small number of samples, the results will depend on
# how the partition was done. In the above case, the explained variation is
# inflated for this reason.
#
# We will perform cross-validation to test more robustly our model.
#
# Also, we can make a grid search, using
# :class:`~sklearn.model_selection.GridSearchCV`, to determine the optimal
# number of neighbors and the best way to weight their votes.


param_grid = {
    'n_neighbors': range(1, 12, 2),
    'weights': ['uniform', 'distance'],
}


knn = KNeighborsRegressor()
gscv = GridSearchCV(
    knn,
    param_grid,
    cv=5,
)
gscv.fit(X, log_prec)

##############################################################################
#
# We obtain that 7 is the optimal number of neighbors.


print("Best params", gscv.best_params_)
print("Best score", gscv.best_score_)

##############################################################################
#
# More detailed information about the Canadian weather dataset can be obtained
# in the following references.
#
#  * Ramsay, James O., and Silverman, Bernard W. (2006). Functional Data
#    Analysis, 2nd ed. , Springer, New York.
#
#  *  Ramsay, James O., and Silverman, Bernard W. (2002). Applied Functional
#     Data Analysis, Springer, New York\n'
