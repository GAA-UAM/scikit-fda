Depth Measures
==============

Functions to order functional data.

Each sample of the dataset is assigned a number between 0 and 1.
Larger values correspond to more centered samples and smaller ones to those samples more outward.

.. _depth-measures:

.. autosummary::
   :toctree: autosummary

   skfda.depth_measures.band_depth
   skfda.depth_measures.modified_band_depth
   skfda.depth_measures.fraiman_muniz_depth

The possibility of obtaining the ordering of each point of the sample (compared to the other samples)
is given if a parameter is specified in the functions.

All of them support multivariate functional data, with more than one dimension on the image and
on the domain.



