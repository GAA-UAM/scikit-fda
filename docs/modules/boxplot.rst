Boxplot
=======

Functions to construct the functional data boxplot. Only supported for
functional data with domain dimension 1 or 2 and as many dimensions on
the image as required.

If the dimension of the domain is 1, the following function must be used.

.. autosummary::
   :toctree: autosummary

   fda.boxplot.fdboxplot

If the dimension of the domain is 2, this one.

.. autosummary::
   :toctree: autosummary

   fda.boxplot.surface_boxplot

Both functions return a FDataBoxplotInfo object from which the plot can be retrieved.

.. autosummary::
   :toctree: autosummary

   fda.boxplot.FDataBoxplotInfo




