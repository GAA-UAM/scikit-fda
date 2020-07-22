Metrics
=======

This module contains multiple functional distances and norms.


Lp Spaces
---------

The following functions computes the norms and distances used in Lp spaces.

.. autosummary::
   :toctree: autosummary

   skfda.misc.metrics.lp_norm
   skfda.misc.metrics.lp_distance



Elastic distances
-----------------

The following functions implements multiple distances used in the elastic
analysis and registration of functional data.

.. autosummary::
   :toctree: autosummary

    skfda.misc.metrics.fisher_rao_distance
    skfda.misc.metrics.amplitude_distance
    skfda.misc.metrics.phase_distance
    skfda.misc.metrics.warping_distance


Utils
-----

.. autosummary::
   :toctree: autosummary

   skfda.misc.metrics.distance_from_norm
   skfda.misc.metrics.pairwise_distance
