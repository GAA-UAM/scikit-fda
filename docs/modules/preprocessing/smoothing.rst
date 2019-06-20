Smoothing
=========

Sometimes the functional observations are noisy. The noise can be reduced
by smoothing the data.

Kernel smoothers
----------------

Kernel smoothing methods compute the smoothed value at a point by considering
the influence of each input point over it. For doing this, it considers a
kernel function placed at the desired point. The influence of each input point
will be related with the value of the kernel function at that input point.

There are several kernel smoothers provided in this library. All of them are
also *linear* smoothers, meaning that they compute a smoothing matrix (or hat
matrix) that performs the smoothing as a linear transformation.

All of the smoothers follow the API of an scikit-learn transformer object.

The degree of smoothing is controlled in all smoother by an 
*smoothing parameter*, named ``smoothing_parameter``, that has different
meaning for each smoother.

.. autosummary::
   :toctree: autosummary

   skfda.preprocessing.smoothing.kernel_smoothers.NadarayaWatsonSmoother
   skfda.preprocessing.smoothing.kernel_smoothers.LocalLinearRegressionSmoother
   skfda.preprocessing.smoothing.kernel_smoothers.KNeighborsSmoother

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
   
The `LinearSmootherGeneralizedCVScorer` object accepts also an optional
penalization_function, used instead of the default one. The available ones
are:

.. autosummary::
   :toctree: autosummary

   skfda.preprocessing.smoothing.validation.akaike_information_criterion
   skfda.preprocessing.smoothing.validation.finite_prediction_error
   skfda.preprocessing.smoothing.validation.shibata
   skfda.preprocessing.smoothing.validation.rice
   
An utility method is also provided, which calls the sckit-learn `GridSearchCV`
object with the scorers to find the best smoothing parameters from a list.

.. autosummary::
   :toctree: autosummary

   skfda.preprocessing.smoothing.validation.optimize_smoothing_parameter


References
----------

* Ramsay, J., Silverman, B. W. (2005). Functional Data Analysis. Springer.

* Wasserman, L. (2006). All of nonparametric statistics. Springer Science & Business Media.
