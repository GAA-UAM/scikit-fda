from __future__ import annotations

from typing import Optional

import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.utils.validation import check_is_fitted

from skfda.misc.hat_matrix import HatMatrix, NadarayaWatsonHatMatrix
from skfda.misc.metrics import PairwiseMetric, l2_distance
from skfda.misc.metrics._typing import Metric
from skfda.representation._functional_data import FData


class KernelRegression(
    BaseEstimator,
    RegressorMixin,
):
    r"""Kernel regression with scalar response.

    Let :math:`fd_1 = (f_1, f_2, ..., f_n)` be the functional data set and
    :math:`y = (y_1, y_2, ..., y_n)` be the scalar response corresponding
    to each function in :math:`fd_1`. Then, the estimation for the
    functions in :math:`fd_2 = (g_1, g_2, ..., g_n)` can be calculated as

    .. math::
        \hat{y} = \hat{H}y

    Where :math:`\hat{H}` is a matrix described in
    :class:`~skfda.misc.HatMatrix`.

    Args:
        kernel_estimator: Method used to calculate the hat matrix
            (default = :class:`~skfda.misc.NadarayaWatsonHatMatrix`).
        metric: Metric used to calculate the distances
            (default = :func:`L2 distance <skfda.misc.metrics.distance_l2>`).

    Examples:
        >>> from skfda import FDataGrid
        >>> from skfda.misc.hat_matrix import NadarayaWatsonHatMatrix
        >>> from skfda.misc.hat_matrix import KNeighborsHatMatrix

        >>> grid_points = np.linspace(0, 1, num=11)
        >>> data1 = np.array([i + grid_points for i in range(1, 9, 2)])
        >>> data2 = np.array([i + grid_points for i in range(2, 7, 2)])

        >>> fd_1 = FDataGrid(grid_points=grid_points, data_matrix=data1)
        >>> y = np.array([1, 3, 5, 7])
        >>> fd_2 = FDataGrid(grid_points=grid_points, data_matrix=data2)

        >>> kernel_estimator = NadarayaWatsonHatMatrix(bandwidth=1)
        >>> estimator = KernelRegression(kernel_estimator=kernel_estimator)
        >>> _ = estimator.fit(fd_1, y)
        >>> estimator.predict(fd_2)
        array([ 2.02723928,  4.        ,  5.97276072])

        >>> kernel_estimator = KNeighborsHatMatrix(bandwidth=2)
        >>> estimator = KernelRegression(kernel_estimator=kernel_estimator)
        >>> _ = estimator.fit(fd_1, y)
        >>> estimator.predict(fd_2)
        array([ 2.,  4.,  6.])

    """

    def __init__(
        self,
        *,
        kernel_estimator: Optional[HatMatrix] = None,
        metric: Metric[FData] = l2_distance,
    ):

        self.kernel_estimator = kernel_estimator
        self.metric = metric

    def fit(  # noqa: D102
        self,
        X: FData,
        y: np.ndarray,
        weight: Optional[np.ndarray] = None,
    ) -> KernelRegression:

        self.X_train_ = X
        self.y_train_ = y
        self.weights_ = weight

        if self.kernel_estimator is None:
            self.kernel_estimator = NadarayaWatsonHatMatrix()

        return self

    def predict(  # noqa: D102
        self,
        X: FData,
    ) -> np.ndarray:

        check_is_fitted(self)
        delta_x = PairwiseMetric(self.metric)(X, self.X_train_)

        return self.kernel_estimator(
            delta_x=delta_x,
            X_train=self.X_train_,
            y_train=self.y_train_,
            X=X,
            weights=self.weights_,
        )
