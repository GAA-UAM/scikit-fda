.. _clustering-module:

Clustering
==========

Module with classes to perform clustering of functional data.


K means algorithms
------------------

The following classes implement both, the K-Means and the Fuzzy K-Means
algorithms respectively. In order to show the results in a visual way,
the module :mod:`skfda.exploratory.visualization.clustering_plots
<skfda.exploratory.visualization.clustering_plots>` can be used.
See the example :ref:`sphx_glr_auto_examples_plot_clustering.py` for a
detailed explanation.

.. autosummary::
   :toctree: autosummary

   skfda.ml.clustering.KMeans
   skfda.ml.clustering.FuzzyKMeans


Nearest Neighbors
-----------------

The class :class:`NearestNeighbors <skfda.ml.clustering.NearestNeighbors>`
implements the nearest neighbors algorithm to perform unsupervised neighbor
searches.

.. autosummary::
   :toctree: autosummary

   skfda.ml.clustering.NearestNeighbors
