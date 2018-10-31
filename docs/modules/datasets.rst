Datasets
========

Functions to obtain datasets.

Fetching real datasets
----------------------

The following functions are used to retrieve specific functional datasets:

.. autosummary::
   :toctree: autosummary

   fda.datasets.fetch_growth
   fda.datasets.fetch_phoneme
   fda.datasets.fetch_tecator

Those functions return a dictionary with at least a "data" field containing the
instance data, and a "target" field containing the class labels or regression values,
if any.

Making synthetic datasets
-------------------------

The following functions are used to make synthetic functional datasets:

.. autosummary::
   :toctree: autosummary

   fda.datasets.make_gaussian_process
   fda.datasets.make_sinusoidal_process
