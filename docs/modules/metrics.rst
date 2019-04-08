Metrics
=======

This module contains multiple functional distances and norms.


Lp Spaces
------------

The following functions computes the norms and distances used in Lp spaces.

 .. autosummary::
   :toctree: autosummary

   fda.metrics.norm_lp
   fda.metrics.lp_distance



Elastic distances
-----------------

The following functions implements multiple distances used in the elastic
analysis and registration of functional data.

 .. autosummary::
   :toctree: autosummary

    fda.metrics.fisher_rao_distance
    fda.metrics.amplitude_distance
    fda.metrics.phase_distance
    fda.metrics.warping_distance


Utils
-----

.. autosummary::
   :toctree: autosummary

   fda.metrics.distance_from_norm
   fda.metrics.pairwise_distance
