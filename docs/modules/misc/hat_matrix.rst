Hat Matrix
==========

A hat matrix is an object used in kernel smoothing (:class:`~skfda.preprocessing.smoothing.KernelSmoother`) and
kernel regression (:class:`~skfda.ml.regression.KernelRegression`) algorithms.

Those algorithms estimate the desired values as a weighted mean of train data. The different Hat matrix types define how
these weights are calculated.

See the links below for more information.


.. autosummary::
    :toctree: autosummary

     skfda.misc.hat_matrix.HatMatrix
     skfda.misc.hat_matrix.NadarayaWatsonHatMatrix
     skfda.misc.hat_matrix.LocalLinearRegressionHatMatrix
     skfda.misc.hat_matrix.KNeighborsHatMatrix
