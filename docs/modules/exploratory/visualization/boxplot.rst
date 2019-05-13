Functional Data Boxplot
=======================

Classes to construct the functional data boxplot. Only supported for
functional data with domain dimension 1 or 2 and as many dimensions on
the image as required.

The base abstract class from which the others inherit is FDataBoxplot.

.. autosummary::
   :toctree: autosummary

   skfda.exploratory.visualization.boxplot.FDataBoxplot

If the dimension of the domain is 1, the following class must be used.
See `Boxplot Example <../auto_examples/plot_boxplot.html>`_ for detailed explanation.

.. autosummary::
   :toctree: autosummary

   skfda.exploratory.visualization.boxplot.Boxplot

If the dimension of the domain is 2, this one. See `Surface Boxplot Example
<../auto_examples/plot_surface_boxplot.html>`_ for detailed explanation.

.. autosummary::
   :toctree: autosummary

   skfda.exploratory.visualization.boxplot.SurfaceBoxplot






