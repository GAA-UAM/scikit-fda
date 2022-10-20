.. _classification-module:

Classification
==============

Module with classes to perform classification of functional data.

Nearest Neighbors
-----------------

This module contains `nearest neighbors
<https://en.wikipedia.org/wiki/K-nearest_neighbors_algorithm>`_ estimators to
perform classification. In the examples
:ref:`sphx_glr_auto_examples_plot_k_neighbors_classification.py`  and
:ref:`sphx_glr_auto_examples_plot_radius_neighbors_classification.py`
it is explained the basic usage of these estimators.

.. autosummary::
   :toctree: autosummary

   skfda.ml.classification.KNeighborsClassifier
   skfda.ml.classification.RadiusNeighborsClassifier

Nearest Centroid
----------------

This module contains `nearest centroid
<https://en.wikipedia.org/wiki/Nearest_centroid_classifier>`_ estimators to
perform classification.

.. autosummary::
   :toctree: autosummary

   skfda.ml.classification.NearestCentroid
   skfda.ml.classification.DTMClassifier


Depth
-----

This module contains depth based estimators to perform classification.

.. autosummary::
   :toctree: autosummary

   skfda.ml.classification.DDClassifier
   skfda.ml.classification.DDGClassifier
   skfda.ml.classification.MaximumDepthClassifier
   
Logistic regression
-----------------------
Classifier based on logistic regression.

.. autosummary::
   :toctree: autosummary

   skfda.ml.classification.LogisticRegression

Functional quadratic discriminant analysis
------------------------------------------
Classifier based on the quadratic discriminant analysis.

.. autosummary::
   :toctree: autosummary

   skfda.ml.classification.QuadraticDiscriminantAnalysis
