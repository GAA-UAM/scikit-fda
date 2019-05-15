Clustering Plots
================
In order to show the results of the cluster algorithms in a visual way,
:mod:`this module <skfda.exploratory.visualization.clustering_plots>` is
implemented. It contains the following methods:

.. autosummary::
   :toctree: autosummary

   skfda.exploratory.visualization.clustering_plots.plot_clusters
   skfda.exploratory.visualization.clustering_plots.plot_cluster_lines
   skfda.exploratory.visualization.clustering_plots.plot_cluster_bars

In the first one, the samples of the FDataGrid are divided by clusters which
are assigned different colors. The following functions, are only valid for the
class :class:`FuzzyKMeans <skfda.ml.clustering.base_kmeans.FuzzyKMeans>` to see
the results graphically in the form of a parallel coordinates plot or a barplot
respectively.

See `Clustering Example <../auto_examples/plot_clustering.html>`_ for detailed
explanation.


