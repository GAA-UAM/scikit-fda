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


Product metric for mixed data
-----------------------------

In applications involving heterogeneous data—such as combinations of 
scalar values and functional observations—it is often useful to define
a metric that captures distances across all components in a consistent
way. The following class implements a weighted product metric that
supports combining multiple data types, each potentially requiring
a different metric.

This class generalizes the idea of computing a norm over a product
space, where each component may be associated with a different scale
or importance. It supports common functional data representations
(:class:`~skfda.representation.FDataGrid`, 
:class:`~skfda.representation.FDataBasis`), numeric arrays, and 
:class:`pandas.DataFrame` objects mixing them.

.. autosummary::
   :toctree: autosummary

   skfda.misc.metrics.PProductMetric
   skfda.misc.metrics.pproduct_metric
   skfda.misc.metrics.DefaultMetric

The metric is defined as a weighted :math:`L^p`` norm of component-wise
distances. Each component may be assigned its own metric and weight,
allowing fine-grained control over the overall distance calculation.
This is especially useful in machine learning tasks that involve mixed
input types, such as classification or clustering over functional and
scalar features.

Functional wrappers for ease of use are also provided, along with a
default metric that infers the appropriate behavior depending on the
input type.