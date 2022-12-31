Miscellaneous
=============

Miscellaneous functions and objects.

This module groups classes and functions useful to work with functional data
but which do not belong to other categories.

Mathematical operations
-----------------------

Some math operations between functional data objects are directly available
in this module.
The most important ones are the ones that efficiently compute the inner
product between functions:

.. autosummary::
   :toctree: autosummary

   skfda.misc.inner_product
   skfda.misc.inner_product_matrix
   
A concept related with the inner product is that of the cosine similarity
between functions:

.. autosummary::
   :toctree: autosummary

   skfda.misc.cosine_similarity
   skfda.misc.cosine_similarity_matrix
   
Submodules
----------

In addition the following modules provide useful functionality to work with
functional data:

- :doc:`misc/covariances`: Contains covariance functions to use with
  and :func:`~skfda.datasets.make_gaussian_process`
- :doc:`misc/metrics`: Contains functional data metrics, suitable to being
  used with several metric-based machine-learning tools.
- :doc:`misc/operators`: Contains operators, or functions over functions.
- :doc:`misc/regularization`: Contains regularization functions, usable in
  contexts such as
  :class:`linear regression <skfda.ml.regression.LinearRegression>`,
  :class:`FPCA <skfda.preprocessing.dim_reduction.feature_extraction.FPCA>`,
  or :class:`basis smoothing <skfda.preprocessing.smoothing.BasisSmoother>`.


.. toctree::
   :maxdepth: 4
   :caption: Modules:
   :hidden:

   misc/covariances
   misc/metrics
   misc/operators
   misc/regularization
   misc/hat_matrix
   misc/scoring
