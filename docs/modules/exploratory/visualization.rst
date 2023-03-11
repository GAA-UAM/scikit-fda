Visualization
=============

Visualization methods are one of the most important tools for exploratory analysis.
They can provide intuition over particular data that is very difficult to obtain otherwise.
As functional data is infinite dimensional, good visualization tools capable to summarize
and illustrate the main features of the data are of particular importance.
The visualization module provides a thorough collection of these tools.
Each of them highlights different characteristics of the data and thus they complement each other.

Basic representation
--------------------

Functional data with :term:`domain` dimension of 1 or 2 can be represented directly as function
graphs, which will be curves or surfaces respectively. Each :term:`codomain` dimension will be plotted
separately.
Additionally, for discretized data, the discretization points can be plotted as a scatter plot.
The following classes implement these plotting methods.

.. autosummary::
   :toctree: autosummary

   skfda.exploratory.visualization.representation.GraphPlot
   skfda.exploratory.visualization.representation.ScatterPlot
   
Note that the :func:`~skfda.representation.FData.plot` and
:func:`~skfda.representation.grid.FDataGrid.plot` methods simply instantiate and plot an object
of one of these classes.

Parametric plot
---------------

Parametric plots are used to plot one function versus another when they have the same :term:`domain`.
This is used for example in phase plane plots, showing the relation between two derivatives
of different order.
It is also useful to plot observations corresponding to curves in 2D, as it shows both dimensions
of the :term:`codomain` in the same plot.

.. autosummary::
   :toctree: autosummary

   skfda.exploratory.visualization.ParametricPlot

Functional Data Boxplot
-----------------------

The functional data boxplot is an extension of the univariate boxplot to the functional data domain.
As such, it is a very useful tool to detect outliers and check the magnitude of the variation of the data.
There are two variants of this plot, depending on the number of dimensions (1 or 2) of the :term:`domain`.

If the dimension of the :term:`domain` is 1, the following class must be used.
See the :ref:`sphx_glr_auto_examples_plot_boxplot.py` example for detailed explanation.

.. autosummary::
   :toctree: autosummary

   skfda.exploratory.visualization.Boxplot

If the dimension of the :term:`domain` is 2, this one. See the
:ref:`sphx_glr_auto_examples_plot_surface_boxplot.py`
example for detailed explanation.

.. autosummary::
   :toctree: autosummary

   skfda.exploratory.visualization.SurfaceBoxplot
   
Outliergram
-----------

The outliergram represents each functional observation as a point whose coordinates are its
:class:`modified band depth<skfda.exploratory.depth.ModifiedBandDepth>` and its
:func:`modified epigraph index<skfda.exploratory.stats.modified_epigraph_index>`.
These quantities are related, and in absence of crossings between observations the points
should lie on a parabola.
Thus, substantial deviations from that behavior characterize observations that are shape
outliers.

.. autosummary::
   :toctree: autosummary

   skfda.exploratory.visualization.Outliergram

Magnitude-Shape Plot
--------------------

The Magnitude-Shape plot tries to summarize the shape and magnitude of an observation as real
numbers, and plot them in a scatter plot.
In addition it computes an ellipse, which serves as a decision boundary for detecting outliers.

This is a very useful tool to detect shape and magnitude outliers and differentiate between them. 

.. autosummary::
   :toctree: autosummary

   skfda.exploratory.visualization.MagnitudeShapePlot

Clustering Plots
----------------
In order to show the results of the cluster algorithms in a visual way,
:mod:`this module <skfda.exploratory.visualization.clustering_plots>` is
implemented. It contains the following classes:

.. autosummary::
   :toctree: autosummary

   skfda.exploratory.visualization.clustering.ClusterPlot
   skfda.exploratory.visualization.clustering.ClusterMembershipLinesPlot
   skfda.exploratory.visualization.clustering.ClusterMembershipPlot

In the first one, the samples of the FDataGrid are divided by clusters which
are assigned different colors. The following functions, are only valid for the
class :class:`FuzzyKMeans <skfda.ml.clustering.base_kmeans.FuzzyKMeans>` to see
the results graphically in the form of a parallel coordinates plot or a barplot
respectively.

See `Clustering Example <../auto_examples/plot_clustering.html>`_ for detailed
explanation.

Functional Principal Component Analysis plots
---------------------------------------------
In order to show the modes of variation that the principal components represent,
the following class is implemented:

.. autosummary::
   :toctree: autosummary

   skfda.exploratory.visualization.FPCAPlot

See the example :ref:`sphx_glr_auto_examples_plot_fpca.py` for detailed
explanation.