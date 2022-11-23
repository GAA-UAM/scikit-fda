from __future__ import annotations

from typing import Callable, Tuple

import numpy as np
from sklearn.linear_model import LogisticRegression as mvLogisticRegression
from sklearn.utils.validation import check_is_fitted
from typing_extensions import Literal

from ..._utils import _classifier_get_classes
from ..._utils._sklearn_adapter import BaseEstimator, ClassifierMixin
from ...representation import FDataGrid
from ...typing._numpy import NDArrayAny, NDArrayInt

Solver = Literal["newton-cg", "lbfgs", "liblinear", "sag", "saga"]


class LogisticRegression(
    BaseEstimator,
    ClassifierMixin[FDataGrid, NDArrayAny],
):
    r"""Logistic Regression classifier for functional data.

    This class implements the sequential “greedy” algorithm
    for functional logistic regression proposed in
    :footcite:ts:`bueno++_2021_functional`.

    .. warning::
        For now, only binary classification for functional
        data with one dimensional domains is supported.

    Args:
        n_features_to_select:
            Number of points (and coefficients) to be selected by
            the algorithm.
        penalty:
            Penalty to use in the multivariate logistic regresion
            optimization problem. For more info check the parameter
            "penalty" in
            :external:class:`sklearn.linear_model.LogisticRegression`.
        C:
            Inverse of the regularization parameter in the multivariate
            logistic regresion optimization problem. For more info
            check the parameter "C" in
            :external:class:`sklearn.linear_model.LogisticRegression`.
        solver:
            Algorithm to use in the multivariate logistic regresion
            optimization problem. For more info check the parameter
            "solver" in
            :external:class:`sklearn.linear_model.LogisticRegression`.
        max_iter:
            Maximum number of iterations taken for the solver to converge.

    Attributes:
        classes\_: A list containing the name of the classes
        points\_: A list containing the selected points.
        coef\_: A list containing the coefficient for each selected point.
        intercept\_: Independent term.

    Examples:
        >>> from numpy import array
        >>> from skfda.datasets import make_gaussian_process
        >>> from skfda.ml.classification import LogisticRegression
        >>> fd1 = make_gaussian_process(
        ...         n_samples=50,
        ...         n_features=100,
        ...         noise=0.7,
        ...         random_state=0,
        ... )
        >>> fd2 = make_gaussian_process(
        ...         n_samples=50,
        ...         n_features = 100,
        ...         mean = array([1]*100),
        ...         noise = 0.7,
        ...         random_state=0
        ... )
        >>> fd = fd1.concatenate(fd2)
        >>> y = 50*[0] + 50*[1]
        >>> lr = LogisticRegression()
        >>> _ = lr.fit(fd[::2], y[::2])
        >>> lr.coef_.round(2)
        array([[ 18.91,  19.69,  19.9 ,   6.09,  12.49]])
        >>> lr.points_.round(2)
        array([ 0.11,  0.06,  0.07,  0.02,  0.03])
        >>> lr.score(fd[1::2],y[1::2])
        0.92

        References:
            .. footbibliography::

    """

    def __init__(
        self,
        max_features: int = 5,
        penalty: Literal["l1", "l2", "elasticnet", None] = None,
        C: float = 1,
        solver: Solver = 'lbfgs',
        max_iter: int = 100,
    ) -> None:

        self.max_features = max_features
        self.penalty = penalty
        self.C = C
        self.solver = solver
        self.max_iter = max_iter

    def fit(  # noqa: D102, WPS210
        self,
        X: FDataGrid,
        y: NDArrayAny,
    ) -> LogisticRegression:

        X, classes, y_ind = self._argcheck_X_y(X, y)

        self.classes_ = classes

        n_samples = len(y)
        n_features = len(X.grid_points[0])

        selected_indexes = np.empty(self.max_features, dtype=np.int_)
        selected_values = np.empty((n_samples, self.max_features))

        likelihood_curves_data = np.empty(
            (self.max_features, n_features),
        )

        penalty = 'none' if self.penalty is None else self.penalty

        # multivariate logistic regression
        mvlr = mvLogisticRegression(
            penalty=penalty,
            C=self.C,
            solver=self.solver,
            max_iter=self.max_iter,
        )

        max_features = min(self.max_features, len(X.grid_points[0]))

        last_max_likelihood = -np.inf

        for n_selected in range(max_features):
            for t in range(n_features):

                selected_values[:, n_selected] = X.data_matrix[:, t, 0]
                mvlr.fit(selected_values[:, :n_selected + 1], y_ind)

                # log-likelihood function at t
                with np.errstate(divide='ignore'):
                    log_probs = mvlr.predict_log_proba(
                        selected_values[:, :n_selected + 1],
                    )

                log_probs = np.concatenate(
                    (log_probs[y_ind == 0, 0], log_probs[y_ind == 1, 1]),
                )
                likelihood_curves_data[n_selected, t] = np.mean(log_probs)

            tmax = np.argmax(likelihood_curves_data[n_selected])
            max_likelihood = likelihood_curves_data[n_selected, tmax]
            if max_likelihood == last_max_likelihood:
                # This does not improve
                selected_indexes = selected_indexes[:n_selected]
                selected_values = selected_values[:, :n_selected]
                likelihood_curves_data = likelihood_curves_data[
                    :n_selected,
                    n_features - 1,
                ]
                break

            last_max_likelihood = max_likelihood

            selected_indexes[n_selected] = tmax
            selected_values[:, n_selected] = X.data_matrix[:, tmax, 0]

        # fit for the complete set of points
        mvlr.fit(selected_values, y)

        self.coef_ = mvlr.coef_
        self.intercept_ = mvlr.intercept_
        self._likelihood_curves = FDataGrid(
            likelihood_curves_data,
            grid_points=X.grid_points,
        )
        self._mvlr = mvlr

        self._selected_indexes = selected_indexes
        self.points_ = X.grid_points[0][selected_indexes]

        return self

    def predict(self, X: FDataGrid) -> NDArrayInt:  # noqa: D102
        check_is_fitted(self)
        return self._wrapper(self._mvlr.predict, X)

    def predict_log_proba(self, X: FDataGrid) -> NDArrayInt:  # noqa: D102
        check_is_fitted(self)
        return self._wrapper(self._mvlr.predict_log_proba, X)

    def predict_proba(self, X: FDataGrid) -> NDArrayInt:  # noqa: D102
        check_is_fitted(self)
        return self._wrapper(self._mvlr.predict_proba, X)

    def _argcheck_X(  # noqa: N802
        self,
        X: FDataGrid,
    ) -> FDataGrid:

        if X.dim_domain > 1:
            raise ValueError(
                f'The dimension of the domain has to be one'
                f'; got {X.dim_domain} dimensions',
            )

        return X

    def _argcheck_X_y(  # noqa: N802
        self,
        X: FDataGrid,
        y: NDArrayAny,
    ) -> Tuple[FDataGrid, NDArrayAny, NDArrayAny]:

        X = self._argcheck_X(X)

        classes, y_ind = _classifier_get_classes(y)

        if classes.size > 2:
            raise ValueError(
                f'The number of classes has to be two'
                f'; got {classes.size} classes',
            )

        if (len(y) != len(X)):
            raise ValueError(
                "The number of samples on independent variables"
                " and classes should be the same",
            )

        return (X, classes, y_ind)

    def _wrapper(
        self,
        method: Callable[[NDArrayAny], NDArrayAny],
        X: FDataGrid,
    ) -> NDArrayAny:
        """Wrap multivariate logistic regression method.

        This function transforms functional data in order to pass
        them to a multivariate logistic regression method.

        .. warning::
            This function can't be called before fit.
        """
        X = self._argcheck_X(X)

        x_mv = X.data_matrix[:, self._selected_indexes, 0]

        return method(x_mv)
