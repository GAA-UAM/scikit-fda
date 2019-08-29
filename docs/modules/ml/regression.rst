.. _regression-module:

Regression
==========

Module with classes to perform regression of functional data.

Linear regression
-----------------

Todo: Add documentation of linear regression models.

.. autosummary::
   :toctree: autosummary

   skfda.ml.regression.LinearScalarRegression

Nearest Neighbors
-----------------

This module contains `nearest neighbors
<https://en.wikipedia.org/wiki/K-nearest_neighbors_algorithm>`_ estimators to
perform regression. In the examples `Neighbors Scalar Regression
<../../../auto_examples/plot_neighbors_scalar_regression.html>`_ and
`Neighbors Functional Regression
<../../../auto_examples/plot_neighbors_functional_regression.html>`_
it is explained the basic usage of these estimators.

.. autosummary::
   :toctree: autosummary

   skfda.ml.regression.KNeighborsScalarRegressor
   skfda.ml.regression.RadiusNeighborsScalarRegressor
   skfda.ml.regression.KNeighborsFunctionalRegressor
   skfda.ml.regression.RadiusNeighborsFunctionalRegressor
