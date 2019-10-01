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
perform regression. In the examples
:ref:`sphx_glr_auto_examples_plot_neighbors_scalar_regression.py` and
:ref:`sphx_glr_auto_examples_plot_neighbors_functional_regression.py`
it is explained the basic usage of these estimators.

.. autosummary::
   :toctree: autosummary

   skfda.ml.regression.KNeighborsRegressor
   skfda.ml.regression.RadiusNeighborsRegressor
