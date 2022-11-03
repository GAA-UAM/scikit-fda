Summary statistics
==================

As in univariate and multivariate analysis, in :term:`FDA` summary statistics
can be used to summarize a set of :term:`functional observations`.

Location
--------

The following statistics are available in scikit-fda in order to obtain a
measure of the location or central tendency of :term:`functional data`.

.. autosummary::
   :toctree: autosummary

   skfda.exploratory.stats.mean
   skfda.exploratory.stats.gmean
   skfda.exploratory.stats.trim_mean
   skfda.exploratory.stats.depth_based_median
   skfda.exploratory.stats.geometric_median
   skfda.exploratory.stats.fisher_rao_karcher_mean
   
Dispersion
----------

For obtaining a measure of the dispersion of the data, the following
statistics can be used.

.. autosummary::
   :toctree: autosummary

   skfda.exploratory.stats.cov
   skfda.exploratory.stats.var

