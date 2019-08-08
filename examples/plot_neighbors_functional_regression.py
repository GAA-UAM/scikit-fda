"""
Neighbors Functional Regression
===============================

Shows the usage of the nearest neighbors regressor with functional response.
"""

# Author: Pablo Marcos Manchón
# License: MIT

# sphinx_gallery_thumbnail_number = 4

import skfda
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from skfda.ml.regression import KNeighborsFunctionalRegressor
from skfda.representation.basis import Fourier


################################################################################
#
# In this example we are going to show the usage of the nearest neighbors
# regressors with functional response. There is available a K-nn version,
# :class:`KNeighborsFunctionalRegressor
# <skfda.ml.regression.KNeighborsFunctionalRegressor>`, and other one based in
# the radius, :class:`RadiusNeighborsFunctionalRegressor
# <skfda.ml.regression.RadiusNeighborsFunctionalRegressor>`.
#
#
# As in the scalar response example, we will fetch the caniadian weather
# dataset, which contains the daily temperature and
# precipitation at 35 different locations in Canada averaged over 1960 to 1994.
# The following figure shows the different temperature and precipitation
# curves.
#

data = skfda.datasets.fetch_weather()
fd = data['data']


# Split dataset, temperatures and curves of precipitation
X, y = fd.coordinates

plt.figure()
X.plot()

plt.figure()
y.plot()

################################################################################
#
# We will try to predict the precipitation curves. First of all we are going to
# make a smoothing of the precipitation curves using a basis representation,
# employing for it a fourier basis with 5 elements.
#


y = y.to_basis(Fourier(nbasis=5))

plt.figure()
y.plot()


################################################################################
#
# We will split the dataset in two partitions, for training and test,
# using the sklearn function :func:`sklearn.model_selection.train_test_split`.
#

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.1,
                                                    random_state=28)

################################################################################
#
# We will try make a prediction using 5 neighbors and the :math:`\mathbb{L}^2`
# distance. In this case, to calculate
# the response we will use a mean of the response, weighted by their distance
# to the test sample.
#


knn = KNeighborsFunctionalRegressor(n_neighbors=5, weights='distance')
knn.fit(X_train, y_train)

################################################################################
#
# We can predict values for the test partition using :meth:`predict`. The
# following figure shows the real precipitation curves, in dashed line, and
# the predicted ones.
#

y_pred = knn.predict(X_test)

# Plot prediction
plt.figure()
fig, ax = y_pred.plot()
ax[0].set_prop_cycle(None) # Reset colors
y_test.plot(linestyle='--')


################################################################################
#
# We can quantify how much variability it is explained by the model
# using the :meth:`score` method, which computes the value
#
# .. math::
#    1 - \frac{\sum_{i=1}^{n}\int (y_i(t) - \hat{y}_i(t))^2dt}
#    {\sum_{i=1}^{n} \int (y_i(t)- \frac{1}{n}\sum_{i=1}^{n}y_i(t))^2dt}
#
# where :math:`y_i` are the real responses and :math:`\hat{y}_i` the
# predicted ones.

score = knn.score(X_test, y_test)
print(score)

################################################################################
#
# More detailed information about the canadian weather dataset can be obtained
# in the following references.
#
#  * Ramsay, James O., and Silverman, Bernard W. (2006). Functional Data
#    Analysis, 2nd ed. , Springer, New York.
#
#  *  Ramsay, James O., and Silverman, Bernard W. (2002). Applied Functional
#     Data Analysis, Springer, New York\n'
#

plt.show()
