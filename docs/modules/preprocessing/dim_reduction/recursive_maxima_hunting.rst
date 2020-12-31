Recursive Maxima Hunting
========================

The recursive maxima hunting method is described in
:class:`~skfda.preprocessing.dim_reduction.variable_selection.RecursiveMaximaHunting`.

This method has several parts that can be customized and are described here.

Correction
----------

Recursive Maxima Hunting is an iterative variable selection method that
modifies the data functions in each iteration subtracting the information of
the selected points, in order to uncover points that become relevant once
other points are selected. Thus, the correction applied depends on how we
define the information of the selected points. This can be customized using
the ``correction`` parameter, passing a object with one of the following
interfaces:

.. autosummary::
    :toctree: autosummary

    skfda.preprocessing.dim_reduction.variable_selection.recursive_maxima_hunting.Correction
    skfda.preprocessing.dim_reduction.variable_selection.recursive_maxima_hunting.ConditionalMeanCorrection
    
Currently the available objects are:

.. autosummary::
    :toctree: autosummary

    skfda.preprocessing.dim_reduction.variable_selection.recursive_maxima_hunting.UniformCorrection
    skfda.preprocessing.dim_reduction.variable_selection.recursive_maxima_hunting.GaussianCorrection
    skfda.preprocessing.dim_reduction.variable_selection.recursive_maxima_hunting.GaussianConditionedCorrection
    skfda.preprocessing.dim_reduction.variable_selection.recursive_maxima_hunting.GaussianSampleCorrection
    
Redundancy
----------

Although redundant points should be eliminated by the correction, numerical
errors and inappropriate corrections may cause redundant points to become
maxima. Thus, redundant points are explicitly masked to exclude them for
future considerations.

Currently there is only one way to detect if a point is redundant:

.. autosummary::
    :toctree: autosummary

    skfda.preprocessing.dim_reduction.variable_selection.recursive_maxima_hunting.DependenceThresholdRedundancy
    
Stopping criterion
------------------

In order for the algorithm to stop, the remaining points should not be relevant
enough. There are several ways to check this condition:

.. autosummary::
    :toctree: autosummary

    skfda.preprocessing.dim_reduction.variable_selection.recursive_maxima_hunting.ScoreThresholdStop
    skfda.preprocessing.dim_reduction.variable_selection.recursive_maxima_hunting.AsymptoticIndependenceTestStop
