from __future__ import annotations

from typing import TypeVar, Union

from sklearn.utils.validation import check_is_fitted

from ..._utils._sklearn_adapter import BaseEstimator, RegressorMixin
from ...misc.hat_matrix import HatMatrix, NadarayaWatsonHatMatrix
from ...misc.metrics import PairwiseMetric, l2_distance
from ...representation._functional_data import FData
from ...typing._metric import Metric
from ...typing._numpy import NDArrayFloat

Input = TypeVar("Input", bound=Union[NDArrayFloat, FData], contravariant=True)
Prediction = TypeVar("Prediction", bound=Union[NDArrayFloat, FData])


class KernelRegression(
    BaseEstimator,
    RegressorMixin[Input, Prediction],
):
    r"""Kernel regression with scalar response.

    Let :math:`fd_1 = (X_1, X_2, ..., X_n)` be the functional data set and
    :math:`y = (y_1, y_2, ..., y_n)` be the scalar response corresponding
    to each function in :math:`fd_1`. Then, the estimation for the
    functions in :math:`fd_2 = (X'_1, X'_2, ..., X'_n)` can be
    calculated as

    .. math::
        \hat{y} = \hat{H}y

    Where :math:`\hat{H}` is a matrix described in
    :class:`~skfda.misc.hat_matrix.HatMatrix`.

    Args:
        kernel_estimator: Method used to calculate the hat matrix
            (default =
            :class:`~skfda.misc.hat_matrix.NadarayaWatsonHatMatrix`).
        metric: Metric used to calculate the distances
            (default = :func:`L2 distance <skfda.misc.metrics.distance_l2>`).

    Examples:
        >>> import numpy as np
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

        >>> kernel_estimator = KNeighborsHatMatrix(n_neighbors=2)
        >>> estimator = KernelRegression(kernel_estimator=kernel_estimator)
        >>> _ = estimator.fit(fd_1, y)
        >>> estimator.predict(fd_2)
        array([ 2.,  4.,  6.])

    """

    def __init__(
        self,
        *,
        kernel_estimator: HatMatrix | None = None,
        metric: Metric[Input] = l2_distance,
    ):

        self.kernel_estimator = kernel_estimator
        self.metric = metric

    def fit(  # noqa: D102
        self,
        X: Input,
        y: Prediction,
        weight: NDArrayFloat | None = None,
    ) -> KernelRegression[Input, Prediction]:

        self.X_train_ = X
        self.y_train_ = y
        self.weights_ = weight

        self._kernel_estimator: HatMatrix
        if self.kernel_estimator is None:
            self._kernel_estimator = NadarayaWatsonHatMatrix()
        else:
            self._kernel_estimator = self.kernel_estimator

        return self

    def predict(  # noqa: D102
        self,
        X: Input,
    ) -> Prediction:

        check_is_fitted(self)
        delta_x = PairwiseMetric(self.metric)(X, self.X_train_)

        return self._kernel_estimator(
            delta_x=delta_x,
            X_train=self.X_train_,
            y_train=self.y_train_,
            X=X,
            weights=self.weights_,
        )
