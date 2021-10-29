Functional Principal Component Analysis (FPCA)
==============================================

This module provides tools to analyse functional data using FPCA. FPCA is
a common tool used to reduce dimensionality. It can be applied to a functional
data object in either a basis representation or a discretized representation.
The output of FPCA are the projections of the original sample functions into the
directions (principal components) in which most of the variance is conserved.
In multivariate PCA those directions are vectors. However, in FPCA we seek
functions that maximizes the sample variance operator, and then project our data
samples into those principal components. The number of principal components are
at most the number of original features.

For a detailed example please view :ref:`sphx_glr_auto_examples_plot_fpca.py`,
where the process is applied to several datasets in both discretized and basis
forms.

FPCA for functional data in both representations
----------------------------------------------------------------

.. autosummary::
   :toctree: autosummary

   skfda.preprocessing.dim_reduction.feature_extraction.FPCA
