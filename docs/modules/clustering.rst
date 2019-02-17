Clustering
==========

Functions to cluster functional data in a FDataGrid object.

This module contains functions to group observations in such a way that those in the same group (called a cluster)
are more similar (in some sense) to each other than to those in other groups (clusters). In this context, the
:func:`norm <fda.math.norm_lp>` between functions is used to classify the observations.

The following functions implement the K-Means and the Fuzzy C-Means algorithms respectively. They are run as many
times as dimensions on the image the FDataGrid object has.

.. autosummary::
   :toctree: autosummary

   fda.clustering.clustering
   fda.clustering.fuzzy_clustering


