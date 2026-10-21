Metrics
=======

This module contains multiple functional distances and norms.


Lp Spaces
---------

The following classes compute the norms and metrics used in Lp spaces. One
first has to create an instance for the class, specifying the desired value
for ``p``, and use this instance to evaluate the norm or distance over
:term:`functional data objects`.

.. autosummary::
   :toctree: autosummary

   skfda.misc.metrics.LpNorm
   skfda.misc.metrics.LpDistance

As the :math:`L_1`, :math:`L_2` and :math:`L_{\infty}` norms are very common
in :term:`FDA`, instances for these have been created, called respectively
``l1_norm``, ``l2_norm`` and ``linf_norm``. The same is true for metrics,
having ``l1_distance``, ``l2_distance`` and ``linf_distance`` already
created.

The following functions are wrappers for convenience, in case that one
only wants to evaluate the norm/metric for a value of ``p``. These functions
cannot be used in objects or methods that require a norm or metric, as the
value of ``p`` must be explicitly passed in each call.

.. autosummary::
   :toctree: autosummary

   skfda.misc.metrics.lp_norm
   skfda.misc.metrics.lp_distance

Weighted Lp Spaces
------------------

In some applications, it is useful to emphasize certain time points more than
others when measuring the distance between functional observations. This can
be achieved by introducing a weight function :math:`w(t)` inside the integral.
The weight function assigns varying importance to different regions of the
domain :math:`\mathcal{T}`, resulting in a weighted :math:`L^p` norm.

These classes generalize the standard Lp norm and distance by supporting such
weighting. This is useful in domains where relevance is time- or region-specific—
such as emphasizing peak hours in energy data or symptom windows in medical
signals.

.. autosummary::
   :toctree: autosummary

   skfda.misc.metrics.WeightedLpNorm
   skfda.misc.metrics.WeightedLpDistance

The following functional wrappers are provided for convenience, allowing direct
evaluation of the norm or distance without explicitly creating a class instance:

.. autosummary::
   :toctree: autosummary

   skfda.misc.metrics.weighted_lp_norm
   skfda.misc.metrics.weighted_lp_distance

These norms and distances are compatible with the same functionality that uses
standard Lp norms, including classification pipelines, clustering algorithms,
and pairwise distance computations.

Angular distance
----------------

The angular distance (using the normalized "angle" between functions given
by the inner product) is also available, and useful in some contexts.

.. autosummary::
   :toctree: autosummary

   skfda.misc.metrics.angular_distance


Elastic distances
-----------------

The following functions implements multiple distances used in the elastic
analysis and registration of functional data.

.. autosummary::
   :toctree: autosummary

    skfda.misc.metrics.fisher_rao_distance
    skfda.misc.metrics.fisher_rao_amplitude_distance
    skfda.misc.metrics.fisher_rao_phase_distance

Mahalanobis distance
--------------------

The following class implements a functional version of the Mahalanobis
distance:

.. autosummary::
   :toctree: autosummary

    skfda.misc.metrics.MahalanobisDistance



Metric induced by a norm
------------------------

If a norm has been defined, it is possible to construct a metric between two
elements simply subtracting one from the other and computing the norm of the
result. Such a metric is called the metric induced by the norm, and the
:math:`Lp` distance is an example of these. The following class can be used
to construct a metric from a norm in this way:

.. autosummary::
   :toctree: autosummary

   skfda.misc.metrics.NormInducedMetric


Pairwise metric
---------------

Some tasks require the computation of all possible distances between pairs
of objets. The following class can compute that efficiently:

.. autosummary::
   :toctree: autosummary

   skfda.misc.metrics.PairwiseMetric


Transformation metric
---------------------

Some metrics, such as those based in derivatives, can be expressed as a
transformation followed by another metric:

.. autosummary::
   :toctree: autosummary

   skfda.misc.metrics.TransformationMetric
