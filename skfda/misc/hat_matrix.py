# -*- coding: utf-8 -*-

"""Hat Matrix.

This module include implementation to create Nadaraya-Watson,
Local Linear Regression and K-Nearest Neighbours hat matrices used in
kernel smoothing and kernel regression.

"""
from __future__ import annotations

import abc
import math
from typing import Callable, TypeVar, Union, overload

import numpy as np

from .._utils._sklearn_adapter import BaseEstimator
from ..representation._functional_data import FData
from ..representation.basis import FDataBasis
from ..typing._base import GridPointsLike
from ..typing._numpy import NDArrayFloat
from . import kernels

Input = TypeVar("Input", bound=Union[FData, GridPointsLike])
Prediction = TypeVar("Prediction", bound=Union[NDArrayFloat, FData])


class HatMatrix(
    BaseEstimator,
):
    """
    Hat Matrix.

    Base class for different hat matrices.

    See also:
        :class:`~skfda.misc.hat_matrix.NadarayaWatsonHatMatrix`
        :class:`~skfda.misc.hat_matrix.LocalLinearRegressionHatMatrix`
        :class:`~skfda.misc.hat_matrix.KNeighborsHatMatrix`
    """

    def __init__(
        self,
        *,
        kernel: Callable[[NDArrayFloat], NDArrayFloat] = kernels.normal,
    ):
        self.kernel = kernel

    @overload
    def __call__(
        self,
        *,
        delta_x: NDArrayFloat,
        X_train: Input | None = None,
        X: Input | None = None,
        y_train: None = None,
        weights: NDArrayFloat | None = None,
        _cv: bool = False,
    ) -> NDArrayFloat:
        pass

    @overload
    def __call__(
        self,
        *,
        delta_x: NDArrayFloat,
        X_train: Input | None = None,
        X: Input | None = None,
        y_train: Prediction | None = None,
        weights: NDArrayFloat | None = None,
        _cv: bool = False,
    ) -> Prediction:
        pass

    def __call__(
        self,
        *,
        delta_x: NDArrayFloat,
        X_train: Input | None = None,
        X: Input | None = None,
        y_train: NDArrayFloat | FData | None = None,
        weights: NDArrayFloat | None = None,
        _cv: bool = False,
    ) -> NDArrayFloat | FData:
        r"""
        Calculate the hat matrix or the prediction.

        If y_train is given, the estimation for X is calculated, otherwise,
        hat matrix, :math:`\hat{H]`, is returned. The prediction, :math:`y`,
        can be calculated as :math:`y = \hat{H} \dot y_train`.

        Args:
            delta_x: Matrix of distances between points or
                functions.
            X_train: Training data.
            X: Test samples.
            y_train: Target values for X_train.
            weights: Weights to be applied to the
                resulting matrix columns.

        Returns:
            The prediction if y_train is given, hat matrix otherwise.

        """
        # Obtain the non-normalized matrix
        matrix = self._hat_matrix_function_not_normalized(
            delta_x=delta_x,
        )

        # Adjust weights
        if weights is not None:
            matrix *= weights

        # Set diagonal to zero if requested (for testing purposes only)
        if _cv:
            np.fill_diagonal(matrix, 0)

        # Renormalize weights
        rs = np.sum(matrix, axis=1)
        rs[rs == 0] = 1

        matrix = (matrix.T / rs).T

        if y_train is None:
            return matrix

        return matrix @ y_train

    @abc.abstractmethod
    def _hat_matrix_function_not_normalized(
        self,
        *,
        delta_x: NDArrayFloat,
    ) -> NDArrayFloat:
        pass


class NadarayaWatsonHatMatrix(HatMatrix):
    r"""Nadaraya-Watson method.

    Creates the matrix :math:`\hat{H}`, used in the kernel smoothing and kernel
    regression algorithms, as explained below

    .. math::
        \hat{H}_{i,j} = \frac{K\left(\frac{d(x_j-x_i')}{h}\right)}{\sum_{k=1}^{
        n}K\left(\frac{d(x_k-x_i')}{h}\right)}

    For smoothing, :math:`\{x_1, ..., x_n\}` are the points with known value
    and :math:`\{x_1', ..., x_m'\}` are the points for which it is desired to
    estimate the smoothed value. The distance :math:`d` is the absolute value
    function :footcite:`wasserman_2006_nonparametric_nw`.

    For regression, :math:`\{x_1, ..., x_n\}` is the functional data and
    :math:`\{x_1', ..., x_m'\}` are the functions for which it is desired to
    estimate the scalar value. Here, :math:`d` is some functional distance
    :footcite:`ferraty+vieu_2006_nonparametric_nw`.

    In both cases :math:`K(\cdot)` is a kernel function and :math:`h` is the
    bandwidth.

    Args:
        bandwidth: Window width of the kernel
            (also called h or bandwidth).
        kernel: Kernel function. By default a normal
            kernel.

    References:
        .. footbibliography::

    """

    def __init__(
        self,
        *,
        bandwidth: float | None = None,
        kernel: Callable[[NDArrayFloat], NDArrayFloat] = kernels.normal,
    ):
        super().__init__(kernel=kernel)
        self.bandwidth = bandwidth

    def _hat_matrix_function_not_normalized(
        self,
        *,
        delta_x: NDArrayFloat,
    ) -> NDArrayFloat:

        bandwidth = (
            np.percentile(np.abs(delta_x), 15)
            if self.bandwidth is None
            else self.bandwidth
        )

        return self.kernel(delta_x / bandwidth)


class LocalLinearRegressionHatMatrix(HatMatrix):
    r"""Local linear regression method.

    Creates the matrix :math:`\hat{H}`, used in the kernel smoothing and kernel
    regression algorithms, as explained below.

    For **kernel smoothing** algorithm to estimate the smoothed value for
    :math:`t_i'` the following error must be minimised

    .. math::
        AWSE(a, b) = \sum_{j=1}^n \left[ \left(y_j -
        \left(a +  b (t_j - t'_i) \right) \right)^2
        K \left( \frac {|t_j - t'_i|}{h} \right) \right ]

    which gives the following expression for each cell

    .. math::
        \hat{H}_{i,j} = \frac{b_j(t_i')}{\sum_{k=1}^{n}b_k(t_i')}

    .. math::
        b_j(t_i') = K\left(\frac{t_j - t_i'}{h}\right) S_{n,2}(t_i') -
        (t_j - t_i')S_{n,1}(t_i')

    .. math::
        S_{n,k}(t_i') = \sum_{j=1}^{n}K\left(\frac{t_j-t_i'}{h}\right)
        (t_j-t_i')^k

    where :math:`\{t_1, t_2, ..., t_n\}` are points with known value and
    :math:`\{t_1', t_2', ..., t_m'\}` are the points for which it is
    desired to estimate the smoothed value
    :footcite:`wasserman_2006_nonparametric_llr`.

    For **kernel regression** algorithm:

    Given functional data, :math:`\{X_1, X_2, ..., X_n\}` where each function
    is expressed in a orthonormal basis with :math:`J` elements and scalar
    response :math:`Y = \{y_1, y_2, ..., y_n\}`.

    It is desired to estimate the values
    :math:`\hat{Y} = \{\hat{y}_1, \hat{y}_2, ..., \hat{y}_m\}`
    for the data :math:`\{X'_1, X'_2, ..., X'_m\}`
    (expressed in the same basis).

    For each :math:`X'_k` the estimation :math:`\hat{y}_k` is obtained by
    taking the value :math:`a_k` from the vector
    :math:`(a_k, b_{1k}, ..., b_{Jk})` which minimizes the following expression

    .. math::
        AWSE(a_k, b_{1k}, ..., b_{Jk}) = \sum_{i=1}^n \left(y_i -
        \left(a_k + \sum_{j=1}^J b_{jk} c_{ik}^j \right) \right)^2
        K \left( \frac {d(X_i - X'_k)}{h} \right)

    Where :math:`c_{ik}^j` is the :math:`j`-th coefficient in a truncated basis
    expansion of :math:`X_i - X'_k = \sum_{j=1}^J c_{ik}^j` and :math:`d` some
    functional distance :footcite:`baillo+grane_2008_llr`

    For both cases, :math:`K(\cdot)` is a kernel function and :math:`h` the
    bandwidth.

    Args:
        bandwidth: Window width of the kernel
            (also called h).
        kernel: Kernel function. By default a normal
            kernel.

    References:
        .. footbibliography::

    """

    def __init__(
        self,
        *,
        bandwidth: float | None = None,
        kernel: Callable[[NDArrayFloat], NDArrayFloat] = kernels.normal,
    ):
        super().__init__(kernel=kernel)
        self.bandwidth = bandwidth

    @overload
    def __call__(
        self,
        *,
        delta_x: NDArrayFloat,
        X_train: FData | GridPointsLike | None = None,
        X: FData | GridPointsLike | None = None,
        y_train: None = None,
        weights: NDArrayFloat | None = None,
        _cv: bool = False,
    ) -> NDArrayFloat:
        pass

    @overload
    def __call__(
        self,
        *,
        delta_x: NDArrayFloat,
        X_train: FData | GridPointsLike | None = None,
        X: FData | GridPointsLike | None = None,
        y_train: Prediction | None = None,
        weights: NDArrayFloat | None = None,
        _cv: bool = False,
    ) -> Prediction:
        pass

    def __call__(  # noqa: D102
        self,
        *,
        delta_x: NDArrayFloat,
        X_train: FData | GridPointsLike | None = None,
        X: FData | GridPointsLike | None = None,
        y_train: NDArrayFloat | FData | None = None,
        weights: NDArrayFloat | None = None,
        _cv: bool = False,
    ) -> NDArrayFloat | FData:

        bandwidth = (
            np.percentile(np.abs(delta_x), 15)
            if self.bandwidth is None
            else self.bandwidth
        )

        # Regression
        if isinstance(X_train, FData):
            assert isinstance(X, FData)

            if not (
                isinstance(X_train, FDataBasis)
                and isinstance(X, FDataBasis)
            ):
                raise ValueError("Only FDataBasis is supported for now.")

            if y_train is None:
                y_train = np.identity(X_train.n_samples)

            m1 = X_train.coefficients
            m2 = X.coefficients

            # Subtract previous matrices obtaining a 3D matrix
            # The i-th element contains the matrix X_train - X[i]
            C = m1 - m2[:, np.newaxis]

            inner_product_matrix = X_train.basis.inner_product_matrix()

            # Calculate new coefficients taking into account cross-products
            # if the basis is orthonormal, C would not change
            C = C @ inner_product_matrix  # noqa: WPS350

            # Adding a column of ones in the first position of all matrices
            dims = (C.shape[0], C.shape[1], 1)
            C = np.concatenate((np.ones(dims), C), axis=-1)

            return self._solve_least_squares(
                delta_x=delta_x,
                coefs=C,
                y_train=y_train,
                bandwidth=bandwidth,
            )

        # Smoothing
        else:

            return super().__call__(  # type: ignore[misc, type-var] # noqa: WPS503
                delta_x=delta_x,
                X_train=X_train,
                X=X,
                y_train=y_train,  # type: ignore[arg-type]
                weights=weights,
                _cv=_cv,
            )

    def _solve_least_squares(
        self,
        delta_x: NDArrayFloat,
        coefs: NDArrayFloat,
        y_train: NDArrayFloat,
        *,
        bandwidth: float,
    ) -> NDArrayFloat:

        W = np.sqrt(self.kernel(delta_x / bandwidth))

        # A x = b
        # Where x = (a, b_1, ..., b_J).
        A = (coefs.T * W.T).T
        b = np.einsum('ij, j... -> ij...', W, y_train)

        # For Ax = b calculates x that minimize the square error
        # From https://stackoverflow.com/questions/42534237/broadcasted-lstsq-least-squares  # noqa: E501
        u, s, vT = np.linalg.svd(A, full_matrices=False)

        uTb = np.einsum('ijk, ij...->ik...', u, b)
        uTbs = (uTb.T / s.T).T
        x = np.einsum('ijk,ij...->ik...', vT, uTbs)

        return x[:, 0]  # type: ignore[no-any-return]

    def _hat_matrix_function_not_normalized(
        self,
        *,
        delta_x: NDArrayFloat,
    ) -> NDArrayFloat:

        if self.bandwidth is None:
            percentage = 15
            self.bandwidth = np.percentile(np.abs(delta_x), percentage)

        k = self.kernel(delta_x / self.bandwidth)

        s1 = np.sum(k * delta_x, axis=1, keepdims=True)  # S_n_1
        s2 = np.sum(k * delta_x ** 2, axis=1, keepdims=True)  # S_n_2
        b = (k * (s2 - delta_x * s1))  # b_i(x_j)

        return b  # type: ignore[no-any-return] # noqa: WPS331


class KNeighborsHatMatrix(HatMatrix):
    r"""K-nearest neighbour kernel method.

    Creates the matrix :math:`\hat{H}`, used in the kernel smoothing and kernel
    regression algorithms, as explained below.

    .. math::
        \hat{H}_{i,j} = \frac{K\left(\frac{d(x_j-x_i')}{h_i}\right)}
        {\sum_{k=1}^{n}K\left(\frac{d(x_k-x_i')}{h_{i}}\right)}

    For smoothing, :math:`\{x_1, ..., x_n\}` are the points with known value
    and :math:`\{x_1', ..., x_m'\}` are the points for which it is desired to
    estimate the smoothed value. The distance :math:`d` is the absolute value.

    For regression, :math:`\{x_1, ..., x_n\}` are the functional data and
    :math:`\{x_1', ..., x_m'\}` are the functions for which it is desired to
    estimate the scalar value. Here, :math:`d` is some functional distance.

    In both cases, :math:`K(\cdot)` is a kernel function and
    :math:`h_{i}` is calculated as the distance from :math:`x_i'` to its
    ``n_neighbors``-th nearest neighbor in :math:`\{x_1, ..., x_n\}`
    :footcite:`ferraty+vieu_2006_nonparametric_knn`.

    Used with the uniform kernel, it takes the average of the closest
    ``n_neighbors`` points to a given point.

    Args:
        n_neighbors: Number of nearest neighbours. By
            default it takes the 5% closest elements.
        kernel: Kernel function. By default a uniform
            kernel to perform a 'usual' k nearest neighbours estimation.

    References:
        .. footbibliography::

    """

    def __init__(
        self,
        *,
        n_neighbors: int | None = None,
        kernel: Callable[[NDArrayFloat], NDArrayFloat] = kernels.uniform,
    ):
        self.n_neighbors = n_neighbors
        self.kernel = kernel

    def _hat_matrix_function_not_normalized(
        self,
        *,
        delta_x: NDArrayFloat,
    ) -> NDArrayFloat:

        input_points_len = delta_x.shape[1]

        n_neighbors = (
            math.floor(
                np.percentile(
                    range(1, input_points_len),
                    5,
                ),
            )
            if self.n_neighbors is None
            else self.n_neighbors
        )

        if n_neighbors <= 0:
            raise ValueError('h must be greater than 0')

        # Tolerance to avoid points landing outside the kernel window due to
        # computation error
        tol = np.finfo(np.float64).eps

        # For each row in the distances matrix, it calculates the furthest
        # point within the k nearest neighbours
        vec = np.sort(
            np.abs(delta_x),
            axis=1,
        )[:, n_neighbors - 1] + tol

        return self.kernel((delta_x.T / vec).T)
