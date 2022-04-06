Depth and outlyingness measures
===============================

Depth and outlyingness functions are related concepts proposed to order the
observations of a dataset, extend the concept of median and trimmed
statistics to multivariate and functional data and to detect outliers.

Depth
-----

.. _depth-measures:

Depth measures are functions that assign, to each possible observation, a value
measuring how deep is that observation inside a given distribution (usually the
distribution is approximated by a dataset).
This function has it maximum value towards a "center" of the distribution,
called the median of the depth.
This allows a extension of the concept of median to multivariate or functional
data.
These functions also provide a natural order of the data, which is required to
apply methods such as the boxplot or the trimmed mean.

The interface of a depth function is given by the following class:

.. autosummary::
   :toctree: autosummary

   skfda.exploratory.depth.Depth

The following classes implement depth functions for functional data:

.. autosummary::
   :toctree: autosummary

   skfda.exploratory.depth.IntegratedDepth
   skfda.exploratory.depth.BandDepth
   skfda.exploratory.depth.ModifiedBandDepth
   skfda.exploratory.depth.DistanceBasedDepth

Most of them support functional data with more than one dimension
on the :term:`domain` and on the :term:`codomain`.

Multivariate depths
^^^^^^^^^^^^^^^^^^^

Some utilities, such as the
:class:`~skfda.exploratory.visualization.MagnitudeShapePlot` require computing
a non-functional (multivariate) depth pointwise.
Moreover, some functional depths, such as the
:class:`integrated depth <skfda.exploratory.depth.IntegratedDepth>` are defined
using multivariate depths.
Thus we also provide some multivariate depth functions:

.. autosummary::
   :toctree: autosummary

   skfda.exploratory.depth.multivariate.ProjectionDepth
   skfda.exploratory.depth.multivariate.SimplicialDepth

Outlyingness
------------

The concepts of depth and outlyingness are (inversely) related.
A deeper datum is less likely an outlier.
Conversely, a datum with very low depth is possibly an outlier.
The following interface (which is very similar to the one used for depths) is
used to define an outlyingness measure:

.. autosummary::
   :toctree: autosummary

   skfda.exploratory.depth.Outlyingness

Multivariate outlyingness
^^^^^^^^^^^^^^^^^^^^^^^^^

We provide the classical Stahel-Donoho outlyingness measure for the univariate
data case:

.. autosummary::
   :toctree: autosummary

   skfda.exploratory.depth.multivariate.StahelDonohoOutlyingness

Conversion
----------

As depth and outlyingness are closely related, there are ways to convert one
into the other.
The following class define a depth based on an outlyingness measure.

.. autosummary::
   :toctree: autosummary

   skfda.exploratory.depth.OutlyingnessBasedDepth


