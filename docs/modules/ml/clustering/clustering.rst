Clustering
==========

Functions to cluster functional data in a FDataGrid object.

This module contains functions to group observations in such a way that those in
the same group (called a cluster) are more similar (in some sense) to each other
than to those in other groups (clusters). In this context, the :func:`norm
<skfda.misc.metrics.norm_lp>` between functions is used to classify the observations.

The following classes implement both, the K-Means and the Fuzzy C-Means algorithms
respectively. They both inherit from the :class:`BaseKMeansData class
<skfda.ml.clustering.clustering.BaseKMeansData>`. They are run as many times as
dimensions on the image the FDataGrid object has.

.. autosummary::
   :toctree: autosummary

   skfda.ml.clustering.clustering.KMeans
   skfda.ml.clustering.clustering.FuzzyKMeans

In order to show the results in a visual way, the :func:`plot method
<skfda.ml.clustering.clustering.BaseKMeansData.plot>` can be used. The samples of the FDataGrid
are divided by clusters which are assigned different colors.

Moreover, the class :class:`FuzzyKMeans <skfda.ml.clustering.clustering.FuzzyKMeans>`
has other two different functions to see the results graphically in the form of a
parallel coordinates plot or a barplot respectively.

.. autosummary::
   :toctree: autosummary

   skfda.ml.clustering.clustering.FuzzyKMeans.plot_lines
   skfda.ml.clustering.clustering.FuzzyKMeans.plot_bars

See `Clustering Example <../auto_examples/plot_clustering.html>`_ for detailed
explanation.
