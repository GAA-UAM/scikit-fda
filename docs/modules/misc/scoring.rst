Scoring methods for regression with functional response.
========================================================

The functions in this module are a generalization for functional data of
the regression metrics of the sklearn library
(https://scikit-learn.org/stable/modules/model_evaluation.html#regression-metrics).
Only scores that support multioutput are included.


.. autosummary::
    :toctree: autosummary

     skfda.misc.scoring.explained_variance_score
     skfda.misc.scoring.mean_absolute_error
     skfda.misc.scoring.mean_absolute_percentage_error
     skfda.misc.scoring.mean_squared_error
     skfda.misc.scoring.mean_squared_log_error
     skfda.misc.scoring.r2_score
     skfda.misc.scoring.root_mean_squared_error
     skfda.misc.scoring.root_mean_squared_log_error

.. warning::

    The `squared` parameter in `mean_squared_error` and `mean_squared_log_error`
    is deprecated and will be removed in a future version. Use
    `root_mean_squared_error` and `root_mean_squared_log_error` instead.
