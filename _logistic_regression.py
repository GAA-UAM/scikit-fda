from __future__ import annotations

from typing import Callable, Tuple

from numpy import append, array, ndarray, zeros
from numpy.core.fromnumeric import argmax, mean
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.linear_model import LogisticRegression as mvLogisticRegression
from sklearn.utils.validation import check_is_fitted

from ..._utils import _classifier_get_classes
from ...representation import FData, FDataGrid


class LogisticRegression(
    BaseEstimator,  # type: ignore
    ClassifierMixin,  # type: ignore
):
    r"""Logistic Regression classifier for functional data.

    This class implements the sequential “greedy” algorithm
    for functional logistic regression proposed in
    https://arxiv.org/abs/1812.00721.

    .. warning::
        For now, only functional data whith one dimensional domains
        are supported.

    Args:
        p (int): number of points (and coefficients) to be selected by
        the algoritm.

    Attributes:
        points\_: A list containing the selected points.
        coef\_: A list containing the coefficient for each selected point.
        intercept\_: Independent term.

    Examples:
        >>> from numpy import array
        >>> from skfda.datasets import make_gaussian_process
        >>> from skfda.ml.classification import LogisticRegression

        >>> fd1 = make_gaussian_process(n_samples = 50, n_features = 100,
        ...                             noise = 0.5, random_state = 0)
        >>> fd2 = make_gaussian_process(n_samples=50, n_features = 100,
        ...                             mean = array([1]*100), noise = 0.5,
        ...                             random_state=0)

        >>> fd = fd1.concatenate(fd2)
        >>> y = 50*[0] + 50*[1]

        >>> lr = LogisticRegression(p=2)
        >>> _ = lr.fit(fd[::2], y[::2])
        >>> lr.coef_.round(2)
        array([[ 2.41,  1.68]])
        >>> lr.points_.round(2)
        array([ 0.11,  0.  ])
        >>> lr.score(fd[1::2],y[1::2])
        0.92

    """

    def __init__(
        self,
        p: int = 5,
    ) -> None:

        self.p = p

    def fit(  # noqa: D102
        self,
        X: FData,
        y: ndarray,
    ) -> LogisticRegression:

        X, classes, y_ind = self._argcheck_X_y(X, y)

        self.classes_ = classes

        n_samples = len(y)
        n_features = len(X.grid_points[0])

        ts = zeros((self.p, ))  # set of indexes of the selected points

        mvlr = mvLogisticRegression()  # multivariate logistic regression
        ts_values = [[] for _ in range(n_samples)]

        LL = zeros((n_features, ))
        for q in range(self.p):
            for t in range(n_features):

                x_mv = self._multivariate_append(
                    ts_values,
                    X.data_matrix[:, t, 0],
                )
                mvlr.fit(x_mv, y_ind)

                # log-likelihood function at t
                log_probs = mvlr.predict_log_proba(x_mv)
                log_probs = array(
                    [log_probs[i, y[i]] for i in range(n_samples)],
                )
                LL[t] = mean(log_probs)

            tmax = argmax(LL)
            ts[q] = tmax
            ts_values = self._multivariate_append(
                ts_values,
                X.data_matrix[:, tmax, 0],
            )

        # fit for the complete set of points
        mvlr.fit(ts_values, y_ind)
        self.coef_ = mvlr.coef_
        self.intercept_ = mvlr.intercept_
        self._mvlr = mvlr

        self._ts = ts
        self.points_ = array(
            [X.grid_points[0][int(t)] for t in ts],  # noqa: WPS441
        )

        return self

    def predict(self, X: FData) -> ndarray:  # noqa: D102
        check_is_fitted(self)
        return self._wrapper(self._mvlr.predict, X)

    def predict_log_proba(self, X: FData) -> ndarray:  # noqa: D102
        check_is_fitted(self)
        return self._wrapper(self._mvlr.predict_log_proba, X)

    def predict_proba(self, X: FData) -> ndarray:  # noqa: D102
        check_is_fitted(self)
        return self._wrapper(self._mvlr.predict_proba, X)

    def _argcheck_X(
        self,
        X: FData,
    ) -> FDataGrid:

        X = X.to_grid()

        dim = len(X.grid_points)
        if dim > 1:
            raise ValueError(
                f'The dimension of the domain has to be one'
                f'; got {dim} dimensions',
            )

        return X

    def _argcheck_X_y(
        self,
        X: FData,
        y: ndarray,
    ) -> Tuple[FDataGrid, ndarray, ndarray]:

        self._argcheck_X(X)

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

    def _to_multivariate(
        self,
        ts: ndarray,
        X: FData,
    ) -> ndarray:
        """Transform the data for multivariate logistic regression."""
        X = self._argcheck_X(X)

        return array([X.data_matrix[:, int(t), 0] for t in ts]).T

    def _multivariate_append(
        self,
        a: ndarray,
        b: ndarray,
    ) -> ndarray:
        """Append two arrays in a particular manner.

        Args:
            a: ndarray of shape (n, m).
            b: ndarray of shape (n,).

        Returns:
            Array of shape (n, m + 1)
        """
        return append(a, b.reshape(-1, 1), axis=1)

    def _wrapper(
        self,
        method: Callable[[ndarray], ndarray],
        X: FData,
    ):
        """Wrap multivariate logistic regression method.

        This function transforms functional data in order to pass
        them to a multivariate logistic regression method.

        .. warning::
            This function can't be called before fit.
        """

        X = self._argcheck_X(X)

        ts_values = self._to_multivariate(self._ts, X)

        return method(ts_values)
