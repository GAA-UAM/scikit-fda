Datasets
========

Functions to obtain datasets.

Fetching real datasets
----------------------

The following functions are used to retrieve specific functional datasets:

.. autosummary::
   :toctree: autosummary

   skfda.datasets.fetch_aemet
   skfda.datasets.fetch_gait
   skfda.datasets.fetch_growth
   skfda.datasets.fetch_mco
   skfda.datasets.fetch_medflies
   skfda.datasets.fetch_nox
   skfda.datasets.fetch_octane
   skfda.datasets.fetch_phoneme
   skfda.datasets.fetch_tecator
   skfda.datasets.fetch_weather

Those functions return a dictionary with at least a "data" field containing the
instance data, and a "target" field containing the class labels or regression values,
if any.

In addition datasets can be downloaded from CRAN and the UCR:

.. autosummary::
   :toctree: autosummary

   skfda.datasets.fetch_cran
   skfda.datasets.fetch_ucr

Datasets from CRAN are not in a standardized format. Datasets from the UCR are in the same
format as the specific datasets, but often have an explicit test set, accessible as "data_test"
and "target_test".

Making synthetic datasets
-------------------------

The following functions are used to make synthetic functional datasets:

.. autosummary::
   :toctree: autosummary
	
   skfda.datasets.make_gaussian
   skfda.datasets.make_gaussian_process
   skfda.datasets.make_sinusoidal_process
   skfda.datasets.make_multimodal_samples
   skfda.datasets.make_multimodal_landmarks
   skfda.datasets.make_random_warping
