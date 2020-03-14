import numpy as np
from skfda.representation.grid import FDataGrid
from sklearn.model_selection import GridSearchCV, LeaveOneOut


def inner_product_regularized(first,
                              second,
                              derivative_degree,
                              regularization_parameter):
    return first.inner_product(second) + \
           regularization_parameter * \
           first.derivative(derivative_degree). \
               inner_product(second.derivative(derivative_degree))


class FPCARegularizationCVScorer:
    r""" This calculates the regularization score which is basically the norm
    of the orthogonal component to the projection of the data onto the
    components
    Args:
        estimator (Estimator): Linear smoothing estimator.
        X (FDataGrid): Functional data to smooth.
        y (FDataGrid): Functional data target. Should be the same as X.

    Returns:
        float: Cross validation score, with negative sign, as it is a
        penalization.

    """

    def __call__(self, estimator, X, y=None):
        projection_coefficients = inner_product_regularized(X,
                                                            estimator.components,
                                                            estimator.regularization_derivative_degree,
                                                            estimator.regularization_parameter)[
            0]

        for i in range(len(projection_coefficients)):
            estimator.components.coefficients[i] *= projection_coefficients[i]
        data_copy = X.copy(coefficients=np.copy(np.squeeze(X.coefficients)))

        result = 0

        for i in range(estimator.components.n_samples):
            data_copy.coefficients -= estimator.components.coefficients[i]
            result += data_copy.inner_product(data_copy)
            #result += inner_product_regularized(data_copy, data_copy,
            #                                    estimator.regularization_derivative_degree,
            #                                    estimator.regularization_parameter)

        return -result


class RegularizationParameterSearch(GridSearchCV):
    """Chooses the best smoothing parameter and performs smoothing.


    Args:
        estimator (smoother estimator): scikit-learn compatible smoother.
        param_values (iterable): iterable containing the values to test
            for *smoothing_parameter*.
        scoring (scoring method): scoring method used to measure the
            performance of the smoothing. If ``None`` (the default) the
            ``score`` method of the estimator is used.
        n_jobs (int or None, optional (default=None)):
            Number of jobs to run in parallel.
            ``None`` means 1 unless in a :obj:`joblib.parallel_backend`
            context. ``-1`` means using all processors. See
            :term:`scikit-learn Glossary <sklearn:n_jobs>` for more details.

        pre_dispatch (int, or string, optional):
            Controls the number of jobs that get dispatched during parallel
            execution. Reducing this number can be useful to avoid an
            explosion of memory consumption when more jobs get dispatched
            than CPUs can process. This parameter can be:

            - None, in which case all the jobs are immediately
              created and spawned. Use this for lightweight and
              fast-running jobs, to avoid delays due to on-demand
              spawning of the jobs

            - An int, giving the exact number of total jobs that are
              spawned

            - A string, giving an expression as a function of n_jobs,
              as in '2*n_jobs'
        verbose (integer):
            Controls the verbosity: the higher, the more messages.

        error_score ('raise' or numeric):
            Value to assign to the score if an error occurs in estimator
            fitting. If set to 'raise', the error is raised. If a numeric
            value is given, FitFailedWarning is raised. This parameter does
            not affect the refit step, which will always raise the error.
            Default is np.nan.
    """

    def __init__(self, estimator, param_values, *, scoring=None, n_jobs=None,
                 verbose=0):
        super().__init__(estimator=estimator, scoring=scoring,
                         param_grid={'regularization_parameter': param_values},
                         n_jobs=n_jobs,
                         refit=True, cv=LeaveOneOut(),
                         verbose=verbose)
        self.components_basis = estimator.components_basis

    def fit(self, X, y=None, groups=None, **fit_params):

        X -= X.mean()

        if not self.components_basis:
            self.components_basis = X.basis.copy()

        # the maximum number of components only depends on the target basis
        max_components = self.components_basis.n_basis

        # and it cannot be bigger than the number of samples-1, as we are using
        # leave one out cross validation
        if max_components > X.n_samples:
            raise AttributeError("The target basis must have less n_basis"
                                 "than the number of samples - 1")

        self.estimator.n_components = max_components

        return super().fit(X, y, groups=groups, **fit_params)

