Scoring methods for regression with functional response.
========================================================

The functions in this module are a generalization for functional data of
the regression metrics of the sklearn library
(https://scikit-learn.org/stable/modules/model_evaluation.html#regression-metrics).
Only scores that support multioutput are included.


.. autosummary::
    :toctree: autosummary

     skfda.misc.score_functions.explained_variance_score
     skfda.misc.score_functions.mean_absolute_error
     skfda.misc.score_functions.mean_absolute_percentage_error
     skfda.misc.score_functions.mean_squared_error
     skfda.misc.score_functions.mean_squared_log_error
     skfda.misc.score_functions.r2_score
