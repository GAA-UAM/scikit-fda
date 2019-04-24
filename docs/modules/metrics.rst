Metrics
=======

This module contains multiple functional distances and norms.


Lp Spaces
---------

The following functions computes the norms and distances used in Lp spaces.

.. autosummary::
   :toctree: autosummary

   skfda.metrics.norm_lp
   skfda.metrics.lp_distance



Elastic distances
-----------------

The following functions implements multiple distances used in the elastic
analysis and registration of functional data.

.. autosummary::
   :toctree: autosummary

    skfda.metrics.fisher_rao_distance
    skfda.metrics.amplitude_distance
    skfda.metrics.phase_distance
    skfda.metrics.warping_distance


Utils
-----

.. autosummary::
   :toctree: autosummary

   skfda.metrics.distance_from_norm
   skfda.metrics.pairwise_distance
