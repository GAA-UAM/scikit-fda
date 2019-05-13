Metrics
=======

This module contains multiple functional distances and norms.


Lp Spaces
---------

The following functions computes the norms and distances used in Lp spaces.

.. autosummary::
   :toctree: autosummary

   skfda.math.metrics.norm_lp
   skfda.math.metrics.lp_distance



Elastic distances
-----------------

The following functions implements multiple distances used in the elastic
analysis and registration of functional data.

.. autosummary::
   :toctree: autosummary

    skfda.math.metrics.fisher_rao_distance
    skfda.math.metrics.amplitude_distance
    skfda.math.metrics.phase_distance
    skfda.math.metrics.warping_distance


Utils
-----

.. autosummary::
   :toctree: autosummary

   skfda.math.metrics.distance_from_norm
   skfda.math.metrics.pairwise_distance
