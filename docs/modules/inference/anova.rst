ANOVA
==============
This package groups a collection of statistical models, useful for analyzing
equality of means for different subsets of a sample.

One-way functional ANOVA
------------------------
Functionality to perform One-way ANOVA analysis, to compare means among
different samples. One-way stands for one functional response variable and
one unique variable of input.

.. autosummary::
   :toctree: autosummary

   skfda.inference.anova.oneway_anova

Statistics
----------
Statistics that measure the internal and external variability between
groups, used in the models above.

.. autosummary::
   :toctree: autosummary

   skfda.inference.anova.v_sample_stat
   skfda.inference.anova.v_asymptotic_stat

