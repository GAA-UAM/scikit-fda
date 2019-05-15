BaseKMeans
==========

Functions to cluster functional data in a FDataGrid object.

This module contains functions to group observations in such a way that those in
the same group (called a cluster) are more similar (in some sense) to each other
than to those in other groups (clusters). In this context, the :func:`norm
<skfda.misc.metrics.norm_lp>` between functions is used to classify the observations.

The following classes implement both, the K-Means and the Fuzzy C-Means algorithms
respectively. They both inherit from the :class:`BaseKMeans class
<skfda.ml.clustering.base_kmeans.BaseKMeans>`. They are run as many times as
dimensions on the image the FDataGrid object has.

.. autosummary::
   :toctree: autosummary

   skfda.ml.clustering.base_kmeans.KMeans
   skfda.ml.clustering.base_kmeans.FuzzyKMeans

In order to show the results in a visual way, the :func:`plot method
<skfda.ml.clustering.base_kmeans.BaseKMeans.plot>` can be used. The samples of the FDataGrid
are divided by clusters which are assigned different colors.

Moreover, the class :class:`FuzzyKMeans <skfda.ml.clustering.base_kmeans.FuzzyKMeans>`
has other two different functions to see the results graphically in the form of a
parallel coordinates plot or a barplot respectively.

See `Clustering Example <../auto_examples/plot_clustering.html>`_ for detailed
explanation.
