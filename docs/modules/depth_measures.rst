Depth Measures
==============

Functions to order functional data.

Each sample of the dataset is assigned a number between 0 and 1.
Larger values correspond to more centered samples and smaller ones to those samples more outward.

.. _depth-measures:

.. autosummary::
   :toctree: autosummary

   fda.depth_measures.band_depth
   fda.depth_measures.modified_band_depth
   fda.depth_measures.Fraiman_Muniz_depth

The possibility of obtaining the ordering of each point of the sample is given
if a parameter is specified.

All of them support multivariate functional data, with more than one dimension on the image and up to two
dimensions on the domain.



