Smoothing
=========

Sometimes the functional observations are noisy. The noise can be reduced
by smoothing the data.

This module provide several classes, called smoothers, that perform a
smoothing transformation of the data. All of the smoothers follow the
API of an scikit-learn transformer object.

The degree of smoothing is controlled in all smoothers by an 
*smoothing parameter*, named ``smoothing_parameter``, that has different
meaning for each smoother.

Kernel smoothers
----------------

Kernel smoothing methods compute the smoothed value at a point by considering
the influence of each input point over it. For doing this, it considers a
kernel function placed at the desired point. The influence of each input point
will be related with the value of the kernel function at that input point.

The kernel smoother provided in this library is
also a *linear* smoother, meaning that it computes a smoothing matrix (or hat
matrix) that performs the smoothing as a linear transformation.

.. autosummary::
   :toctree: autosummary

   skfda.preprocessing.smoothing.KernelSmoother
   
Basis smoother
--------------

The basis smoother smooths the data by means of expressing it in a truncated basis
expansion. The data can be further smoothed penalizing its derivatives, using
a linear differential operator. This has the effect of reducing the curvature
of the function and/or its derivatives.

This smoother is also a linear smoother, although if the QR or Cholesky methods
are used, the matrix does not need to be explicitly computed.

.. autosummary::
   :toctree: autosummary

   skfda.preprocessing.smoothing.BasisSmoother

Validation
----------

It is necessary to measure how good is the smoothing to prevent
*undersmoothing* and *oversmoothing*. The following classes follow the
scikit-learn API for a scorer object, and measure how good is the smoothing.
In both of them, the target object ``y`` should also be the original data.
These scorers need that the smoother is linear, as they use internally the
hat matrix.

.. autosummary::
   :toctree: autosummary

   skfda.preprocessing.smoothing.validation.LinearSmootherLeaveOneOutScorer
   skfda.preprocessing.smoothing.validation.LinearSmootherGeneralizedCVScorer
   
The 
:class:`~skfda.preprocessing.smoothing.validation.LinearSmootherGeneralizedCVScorer` 
object accepts also an optional penalization_function, used instead of the 
default one. The available ones are:

.. autosummary::
   :toctree: autosummary

   skfda.preprocessing.smoothing.validation.akaike_information_criterion
   skfda.preprocessing.smoothing.validation.finite_prediction_error
   skfda.preprocessing.smoothing.validation.shibata
   skfda.preprocessing.smoothing.validation.rice
   
An utility class is also provided, which inherits from the sckit-learn class 
:class:`~sklearn.model_selection.GridSearchCV`
and performs a grid search using the scorers to find the best
``smoothing_parameter`` from a list.

.. autosummary::
   :toctree: autosummary

   skfda.preprocessing.smoothing.validation.SmoothingParameterSearch


References
----------

* Ramsay, J., Silverman, B. W. (2005). Functional Data Analysis. Springer.

* Wasserman, L. (2006). All of nonparametric statistics. Springer Science & Business Media.
