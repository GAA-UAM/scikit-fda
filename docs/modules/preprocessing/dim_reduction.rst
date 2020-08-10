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

   skfda.preprocessing.dim_reduction.variable_selection.RKHSVariableSelection

Projection
----------
Another way to reduce the dimension is through projection. For example, in
functional principal component analysis, we project the data samples
into a smaller sample of functions that preserve the maximum sample
variance.

.. toctree::
   :maxdepth: 4
   :caption: Modules:

   dim_reduction/fpca