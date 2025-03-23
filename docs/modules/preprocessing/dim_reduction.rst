Dimensionality Reduction
========================

When dealing with data samples with high dimensionality, we often need to
reduce the dimensions so we can better observe the data.

Variable selection
------------------
One approach to reduce the dimensionality of the data is to select a subset of
the original variables or features. This approach is called variable
selection. In FDA, this means evaluating the function at a small number of
points. These evaluations would be the selected features of the functional
datum.

The variable selection transformers implemented in scikit-fda are the
following:

.. autosummary::
   :toctree: autosummary

   skfda.preprocessing.dim_reduction.variable_selection.MaximaHunting
   skfda.preprocessing.dim_reduction.variable_selection.RecursiveMaximaHunting
   skfda.preprocessing.dim_reduction.variable_selection.RKHSVariableSelection
   skfda.preprocessing.dim_reduction.variable_selection.MinimumRedundancyMaximumRelevance

.. toctree::
   :hidden:
   :maxdepth: 4
   :caption: Modules:

   dim_reduction/recursive_maxima_hunting

Feature extraction
------------------
Other dimensionality reduction methods construct new features from
existing ones. For example, in functional principal component
analysis, we project the data samples into a smaller sample of
functions that preserve most of the original
variance. Similarly, in functional partial least squares, we project
the data samples into a smaller sample of functions that preserve most
of the covariance between the two data blocks.

.. autosummary::
   :toctree: autosummary
   
   skfda.preprocessing.dim_reduction.FPCA
   skfda.preprocessing.dim_reduction.FPLS

Difussion methods
-----------------
Diffusion methods, such as functional difussion maps, try to find a natural
less-dimensional manifold in which the data lives, trying to preserve the local
neigborhood of the observations in the reduced space.

.. autosummary::
   :toctree: autosummary

   skfda.preprocessing.dim_reduction.DiffusionMap