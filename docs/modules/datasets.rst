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
   fda.datasets.fetch_medflies
   fda.datasets.fetch_weather
   fda.datasets.fetch_aemet

Those functions return a dictionary with at least a "data" field containing the
instance data, and a "target" field containing the class labels or regression values,
if any.

In addition datasets can be downloaded from CRAN and the UCR:

.. autosummary::
   :toctree: autosummary

   fda.datasets.fetch_cran
   fda.datasets.fetch_ucr

Datasets from CRAN are not in a standardized format. Datasets from the UCR are in the same
format as the specific datasets, but often have an explicit test set, accessible as "data_test"
and "target_test".

Making synthetic datasets
-------------------------

The following functions are used to make synthetic functional datasets:

.. autosummary::
   :toctree: autosummary

   fda.datasets.make_gaussian_process
   fda.datasets.make_sinusoidal_process
   fda.datasets.make_multimodal_samples
   fda.datasets.make_multimodal_landmarks
   fda.datasets.make_random_warping
