from __future__ import annotations

from typing import Callable, Tuple

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.linear_model import LogisticRegression as mvLogisticRegression
from sklearn.utils.validation import check_is_fitted

from ..._utils import _classifier_get_classes
from ...representation import FDataGrid
from ...representation._typing import NDArrayAny, NDArrayInt


class LogisticRegression(
    BaseEstimator,  # type: ignore
    ClassifierMixin,  # type: ignore
):
    r"""Logistic Regression classifier for functional data.

    This class implements the sequential “greedy” algorithm
    for functional logistic regression proposed in
    :footcite:ts:`bueno++_2021_functional`.

    .. warning::
        For now, only binary classification for functional
        data with one dimensional domains is supported.

    Args:
        p:
            number of points (and coefficients) to be selected by
            the algorithm.

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
        array([[ 1.28,  1.17,  1.27,  1.27,  0.96]])
        >>> lr.points_.round(2)
        array([ 0.11,  0.06,  0.07,  0.03,  0.  ])
        >>> lr.score(fd[1::2],y[1::2])
        0.94

        References:
            .. footbibliography::

    """

    def __init__(
        self,
        p: int = 5,
    ) -> None:

        self.p = p

    def fit(  # noqa: D102
        self,
        X: FDataGrid,
        y: NDArrayAny,
    ) -> LogisticRegression:

        X, classes, y_ind = self._argcheck_X_y(X, y)

        self.classes_ = classes

        n_samples = len(y)
        n_features = len(X.grid_points[0])

        selected_indexes = np.zeros(self.p, dtype=np.intc)

        # multivariate logistic regression
        mvlr = mvLogisticRegression(penalty='l2')

        x_mv = np.zeros((n_samples, self.p))
        LL = np.zeros(n_features)
        for q in range(self.p):
            for t in range(n_features):

                x_mv[:, q] = X.data_matrix[:, t, 0]
                mvlr.fit(x_mv[:, :q + 1], y_ind)

                # log-likelihood function at t
                log_probs = mvlr.predict_log_proba(x_mv[:, :q + 1])
                log_probs = np.concatenate(
                    (log_probs[y_ind == 0, 0], log_probs[y_ind == 1, 1]),
                )
                LL[t] = np.mean(log_probs)

            tmax = np.argmax(LL)
            selected_indexes[q] = tmax
            x_mv[:, q] = X.data_matrix[:, tmax, 0]

        # fit for the complete set of points
        mvlr.fit(x_mv, y_ind)

        self.coef_ = mvlr.coef_
        self.intercept_ = mvlr.intercept_
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
