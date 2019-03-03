Clustering
==========

Functions to cluster functional data in a FDataGrid object.

This module contains functions to group observations in such a way that those in
the same group (called a cluster) are more similar (in some sense) to each other
than to those in other groups (clusters). In this context, the :func:`norm <fda.math.norm_lp>`
between functions is used to classify the observations.

The following functions implement the K-Means and the Fuzzy C-Means algorithms
respectively. They are run as many times as dimensions on the image the FDataGrid
object has.

.. autosummary::
   :toctree: autosummary

   fda.clustering.kmeans
   fda.clustering.fuzzy_kmeans

In order to show the results in a visual way, three different functions have been
implemented:

.. autosummary::
   :toctree: autosummary

   fda.clustering.plot_clustering
   fda.clustering.plot_fuzzy_kmeans_lines
   fda.clustering.plot_fuzzy_kmeans_bars

The first one plots the data of the FDataGrid divided by clusters which are assigned
different colors. The clustering method can be customized.
The last two functions show the results of the :func:`fuzzy_kmeans method
<fda.clustering.fuzzy_kmeans>` in the form of a parallel coordinates plot or a barplot
respectively. See `Clustering Example <../auto_examples/plot_clustering.html>`_
for detailed explanation.

