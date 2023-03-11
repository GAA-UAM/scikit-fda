"""Defines methods for the validation of the smoothing."""
from __future__ import annotations

from typing import Any, Callable, Iterable, Tuple

import numpy as np
import sklearn
from sklearn.model_selection import GridSearchCV

from ...representation import FDataGrid
from ...typing._numpy import NDArrayFloat, NDArrayInt
from ._linear import _LinearSmoother


def _get_input_estimation_and_matrix(
    estimator: _LinearSmoother,
    X: FDataGrid,
) -> Tuple[FDataGrid, NDArrayFloat]:
    """Return the smoothed data evaluated at the input points & the matrix."""
    if estimator.output_points is not None:
        estimator = sklearn.base.clone(estimator)
        estimator.output_points = None
        estimator.fit(X)
    y_est = estimator.transform(X)

    hat_matrix = estimator.hat_matrix_

    return y_est, hat_matrix


Scorer = Callable[[_LinearSmoother, FDataGrid, FDataGrid], float]


class LinearSmootherLeaveOneOutScorer:
    r"""Leave-one-out cross validation scoring method for linear smoothers.

    It calculates the cross validation score for every sample in a FDataGrid
    object given a linear smoother with a smoothing matrix
    :math:`\mathbf{S}(h)` calculated with a parameter :math:`h`:

    .. math::
        CV_{loo}(h)=\frac{1}{M} \sum_m \left(x(t_m) - \hat{x}(t_m; h)^{(-m)}
        \right)^2,

    where :math:`\hat{x}(t_m; h)^{(-m)}` is the estimated :math:`x(t_m)` when
    the point :math:`t_m` is excluded in the smoothing. This
    would require to recalculate the smoothing matrix n times. Fortunately
    the above formula can be expressed in a way where the smoothing matrix
    does not need to be calculated again.

    .. math::
        CV_{loo}(h)=\frac{1}{M} \sum_{m=1}^{M}
        \left(\frac{x(t_m) - \hat{x}(t_m; h)}{1 - S_{mm}(h)}\right)^2,

    Args:
        estimator (Estimator): Linear smoothing estimator.
        X (FDataGrid): Functional data to smooth.
        y (FDataGrid): Functional data target. Should be the same as X.

    Returns:
        float: Cross validation score, with negative sign, as it is a
        penalization.

    """

    def __call__(
        self,
        estimator: _LinearSmoother,
        X: FDataGrid,
        y: FDataGrid,
    ) -> float:
        """Calculate Leave-One-Out score for linear smoothers."""
        y_est, hat_matrix = _get_input_estimation_and_matrix(estimator, X)

        return -float(
            np.mean(
                (
                    (y.data_matrix[..., 0] - y_est.data_matrix[..., 0])
                    / (1 - hat_matrix.diagonal())
                ) ** 2,
            ),
        )


class LinearSmootherGeneralizedCVScorer:
    r"""Generalized cross validation scoring method for linear smoothers.

    It calculates the general cross validation score for every sample in a
    FDataGrid object given a smoothing matrix :math:`\mathbf{S}(h)`
    calculated with a parameter :math:`h`:

    .. math::
        GCV(h)=\Xi(\mathbf{S}(h))\frac{1}{M} \sum_{m=1}^{M}
        \left(x(t_m) - \hat{x}(t_m; h)\right)^2,

    Where :math:`\hat{x}(t_m; h)` is the adjusted :math:`x(t_m)` and
    :math:`\Xi` is a penalization function. By default the penalization
    function is:

    .. math::
        \Xi(\mathbf{S}(h)) = \frac{1}{(1 - \text{tr}(\mathbf{S}(h))/M)^2}.

    but others such as the Akaike's information criterion can be considered.

    Args:
        estimator (Estimator): Linear smoothing estimator.
        X (FDataGrid): Functional data to smooth.
        y (FDataGrid): Functional data target. Should be the same as X.

    Returns:
        float: Cross validation score, with negative sign, as it is a
        penalization.

    """

    def __init__(
        self,
        penalization_function: Callable[[NDArrayFloat], float] | None = None,
    ):
        self.penalization_function = penalization_function

    def __call__(
        self,
        estimator: _LinearSmoother,
        X: FDataGrid,
        y: FDataGrid,
    ) -> float:
        """Calculate Leave-One-Out score with penalization function."""
        y_est, hat_matrix = _get_input_estimation_and_matrix(estimator, X)

        if self.penalization_function is None:
            self.penalization_function = _default_penalization_function

        return -float(
            np.mean(
                (
                    (y.data_matrix[..., 0] - y_est.data_matrix[..., 0])
                    / (1 - hat_matrix.diagonal())
                ) ** 2,
            ) * self.penalization_function(hat_matrix),
        )


class SmoothingParameterSearch(
    GridSearchCV,  # type: ignore[misc]
):
    """Chooses the best smoothing parameter and performs smoothing.

    Performs the smoothing of a FDataGrid object choosing the best
    parameter of a given list using a cross validation scoring method.

    Note:
        This is similar to fitting a scikit-learn GridSearchCV over the
        data, using the cv_method as a scorer.

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

    Examples:
        Creates a FDataGrid object of the function :math:`y=x^2` and peforms
        smoothing by means of the k-nearest neighbours method.

        >>> import skfda
        >>> from skfda.preprocessing.smoothing import KernelSmoother
        >>> from skfda.misc.hat_matrix import KNeighborsHatMatrix
        >>> x = np.linspace(-2, 2, 5)
        >>> fd = skfda.FDataGrid(x ** 2, x)
        >>> grid = SmoothingParameterSearch(
        ...         KernelSmoother(
        ...             kernel_estimator=KNeighborsHatMatrix()),
        ...         [2,3],
        ...         param_name='kernel_estimator__n_neighbors')
        >>> _ = grid.fit(fd)
        >>> np.array(grid.cv_results_['mean_test_score']).round(2)
        array([-11.67, -12.37])
        >>> round(grid.best_score_, 2)
        -11.67
        >>> grid.best_params_['kernel_estimator__n_neighbors']
        2
        >>> grid.best_estimator_.hat_matrix().round(2)
        array([[ 0.5 , 0.5 , 0.  , 0.  , 0.  ],
               [ 0.33, 0.33, 0.33, 0.  , 0.  ],
               [ 0.  , 0.33, 0.33, 0.33, 0.  ],
               [ 0.  , 0.  , 0.33, 0.33, 0.33],
               [ 0.  , 0.  , 0.  , 0.5 , 0.5 ]])
        >>> grid.transform(fd).round(2)
        FDataGrid(
            array([[[ 2.5 ],
                    [ 1.67],
                    [ 0.67],
                    [ 1.67],
                    [ 2.5 ]]]),
            grid_points=(array([-2., -1.,  0.,  1.,  2.]),),
            domain_range=((-2.0, 2.0),),
            ...)

        Other validation methods can be used such as cross-validation or
        general cross validation using other penalization functions.

        >>> grid = SmoothingParameterSearch(
        ...         KernelSmoother(
        ...             kernel_estimator=KNeighborsHatMatrix()),
        ...         [2,3],
        ...         param_name='kernel_estimator__n_neighbors',
        ...         scoring=LinearSmootherLeaveOneOutScorer())
        >>> _ = grid.fit(fd)
        >>> np.array(grid.cv_results_['mean_test_score']).round(2)
        array([-4.2, -5.5])
        >>> grid = SmoothingParameterSearch(
        ...         KernelSmoother(
        ...             kernel_estimator=KNeighborsHatMatrix()),
        ...         [2,3],
        ...         param_name='kernel_estimator__n_neighbors',
        ...         scoring=LinearSmootherGeneralizedCVScorer(
        ...                         akaike_information_criterion))
        >>> _ = grid.fit(fd)
        >>> np.array(grid.cv_results_['mean_test_score']).round(2)
        array([ -9.35, -10.71])
        >>> grid = SmoothingParameterSearch(
        ...         KernelSmoother(
        ...             kernel_estimator=KNeighborsHatMatrix()),
        ...         [2,3],
        ...         param_name='kernel_estimator__n_neighbors',
        ...         scoring=LinearSmootherGeneralizedCVScorer(
        ...                         finite_prediction_error))
        >>> _ = grid.fit(fd)
        >>> np.array(grid.cv_results_['mean_test_score']).round(2)
        array([ -9.8, -11. ])
        >>> grid = SmoothingParameterSearch(
        ...         KernelSmoother(
        ...             kernel_estimator=KNeighborsHatMatrix()),
        ...         [2,3],
        ...         param_name='kernel_estimator__n_neighbors',
        ...         scoring=LinearSmootherGeneralizedCVScorer(shibata))
        >>> _ = grid.fit(fd)
        >>> np.array(grid.cv_results_['mean_test_score']).round(2)
        array([-7.56, -9.17])
        >>> grid = SmoothingParameterSearch(
        ...         KernelSmoother(
        ...             kernel_estimator=KNeighborsHatMatrix()),
        ...         [2,3],
        ...         param_name='kernel_estimator__n_neighbors',
        ...         scoring=LinearSmootherGeneralizedCVScorer(rice))
        >>> _ = grid.fit(fd)
        >>> np.array(grid.cv_results_['mean_test_score']).round(2)
        array([-21. , -16.5])

        Different output points can also be used. In that case the value used
        as a target is still the smoothed value at the input points:

        >>> output_points = np.linspace(-2, 2, 9)
        >>> grid = SmoothingParameterSearch(
        ...         KernelSmoother(
        ...             kernel_estimator=KNeighborsHatMatrix(),
        ...             output_points=output_points),
        ...         [2,3],
        ...         param_name='kernel_estimator__n_neighbors')
        >>> _ = grid.fit(fd)
        >>> np.array(grid.cv_results_['mean_test_score']).round(2)
        array([-11.67, -12.37])
        >>> grid.transform(fd).data_matrix.round(2)
        array([[[ 2.5 ],
                [ 2.5 ],
                [ 1.67],
                [ 0.5 ],
                [ 0.67],
                [ 0.5 ],
                [ 1.67],
                [ 2.5 ],
                [ 2.5 ]]])
    """

    def __init__(
        self,
        estimator: _LinearSmoother,
        param_values: Iterable[float],
        *,
        param_name: str = 'smoothing_parameter',
        scoring: Scorer | None = None,
        n_jobs: int | None = None,
        verbose: int = 0,
        pre_dispatch: int | str | None = '2*n_jobs',
        error_score: str | float = np.nan,
    ):
        super().__init__(
            estimator=estimator,
            scoring=scoring,
            param_grid={param_name: param_values},
            n_jobs=n_jobs,
            refit=True,
            cv=[(slice(None), slice(None))],
            verbose=verbose,
            pre_dispatch=pre_dispatch,
            error_score=error_score,
            return_train_score=False,
        )
        self.param_values = param_values

    def fit(  # noqa: D102
        self,
        X: FDataGrid,
        y: FDataGrid | None = None,
        groups: NDArrayInt | None = None,
        **fit_params: Any,
    ) -> SmoothingParameterSearch:
        if y is None:
            y = X

        super().fit(X, y=y, groups=groups, **fit_params)
        return self


def _default_penalization_function(hat_matrix: NDArrayFloat) -> float:
    return float(1 - hat_matrix.diagonal().mean()) ** -2


def akaike_information_criterion(hat_matrix: NDArrayFloat) -> float:
    r"""Akaike's information criterion for cross validation
    :footcite:`febrero-bande+oviedo_2012_fda.usc`.

    .. math::
        \Xi(\nu,n) = \exp\left(2 * \frac{tr(\hat{H}^\nu)}{n}\right)

    Args:
        hat_matrix: Smoothing matrix whose penalization
            score is desired.

    Returns:
        Penalization given by the Akaike's information criterion.

    .. footbibliography::

    """
    return float(np.exp(2 * hat_matrix.diagonal().mean()))


def finite_prediction_error(hat_matrix: NDArrayFloat) -> float:
    r"""Finite prediction error for cross validation
    :footcite:`febrero-bande+oviedo_2012_fda.usc`.

    .. math::
        \Xi(\nu,n) = \frac{1 + \frac{tr(\hat{H}^\nu)}{n}}{1 -
        \frac{tr(\hat{H}^\nu)}{n}}

    Args:
        hat_matrix: Smoothing matrix whose penalization
            score is desired.

    Returns:
         Penalization given by the finite prediction error.

    .. footbibliography::

    """
    return float(
        (1 + hat_matrix.diagonal().mean())
        / (1 - hat_matrix.diagonal().mean())
    )


def shibata(hat_matrix: NDArrayFloat) -> float:
    r"""Shibata's model selector for cross validation
    :footcite:`febrero-bande+oviedo_2012_fda.usc`.

    .. math::
        \Xi(\nu,n) = 1 + 2 * \frac{tr(\hat{H}^\nu)}{n}

    Args:
        hat_matrix: Smoothing matrix whose penalization
            score is desired.

    Returns:
        Penalization given by the Shibata's model selector.

    .. footbibliography::

    """
    return float(1 + 2 * hat_matrix.diagonal().mean())


def rice(hat_matrix: NDArrayFloat) -> float:
    r"""Rice's bandwidth selector for cross validation
    :footcite:`febrero-bande+oviedo_2012_fda.usc`.

    .. math::
        \Xi(\nu,n) = \left(1 - 2 * \frac{tr(\hat{H}^\nu)}{n}\right)^{-1}

    Args:
        hat_matrix: Smoothing matrix whose penalization
            score is desired.

    Returns:
         Penalization given by the Rice's bandwidth selector.

    .. footbibliography::

    """
    return float(1 - 2 * hat_matrix.diagonal().mean()) ** -1
