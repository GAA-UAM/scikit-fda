Depth Measures
==============

Functions to order functional data.

Each sample of the dataset is assigned a number between 0 and 1.
Larger values correspond to more centered samples and smaller ones to those samples more outward.

.. _depth-measures:

.. autosummary::
   :toctree: autosummary

   skfda.exploratory.depth.band_depth
   skfda.exploratory.depth.modified_band_depth
   skfda.exploratory.depth.fraiman_muniz_depth

The possibility of obtaining the ordering of each point of the sample (compared to the other samples)
is given if a parameter is specified in the functions.

All of them support multivariate functional data, with more than one dimension on the image and
on the domain.

Outlyingness conversion to depth
--------------------------------

The concepts of depth and outlyingness are (inversely) related. A deeper datum is less likely an outlier. Conversely,
a datum with very low depth is possibly an outlier. In order to convert an outlying measure to a depth measure
the following convenience function is provided.

.. autosummary::
   :toctree: autosummary
   
   skfda.exploratory.depth.outlyingness_to_depth
   
Multivariate depths
-------------------

Some utilities, such as the :class:`~skfda.exploratory.visualization.MagnitudeShapePlot` require computing a non-functional
(multivariate) depth pointwise. Thus we also provide some multivariate depth functions.

.. autosummary::
   :toctree: autosummary
   
   skfda.exploratory.depth.multivariate.projection_depth



