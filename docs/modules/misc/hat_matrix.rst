Hat Matrix
==========

Hat matrix is used in kernel smoothing (:class:`~skfda.preprocessing.smoothing.KernelSmoother`) and
kernel regression (:class:`~skfda.ml.regression.KernelRegression`) algorithms. See the links below for more information of how the matrix are calculated.


.. autosummary::
    :toctree: autosummary

     skfda.misc.hat_matrix.NadarayaWatsonHatMatrix
     skfda.misc.hat_matrix.LocalLinearRegressionHatMatrix
     skfda.misc.hat_matrix.KNeighborsHatMatrix
