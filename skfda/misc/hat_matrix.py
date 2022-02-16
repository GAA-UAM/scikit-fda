# -*- coding: utf-8 -*-

"""Hat Matrix.

This module include implementation to create Nadaraya-Watson,
Local Linear Regression and K-Nearest Neighbours hat matrices used in
kernel smoothing and kernel regression.

"""

import abc
from typing import Callable, Optional, Union

import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin

from skfda.representation._functional_data import FData
from skfda.representation.basis import FDataBasis

from ..representation._typing import GridPoints, GridPointsLike
from . import kernels


class HatMatrix(
    BaseEstimator,
    RegressorMixin,
):
    """
    Kernel estimators.

    This module includes three types of kernel estimators that are used in
    KernelSmoother and KernelRegression classes.
    """

    def __init__(
        self,
        *,
        bandwidth: Optional[float] = None,
        kernel: Callable[[np.ndarray], np.ndarray] = kernels.normal,
    ):
        self.bandwidth = bandwidth
        self.kernel = kernel

    def __call__(
        self,
        *,
        delta_x: np.ndarray,
        X_train: Optional[Union[FData, GridPointsLike]] = None,
        X: Optional[Union[FData, GridPointsLike]] = None,
        y_train: Optional[np.ndarray] = None,
        weights: Optional[np.ndarray] = None,
        _cv: bool = False,
    ) -> np.ndarray:
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
        delta_x: np.ndarray,
    ) -> np.ndarray:
        pass


class NadarayaWatsonHatMatrix(HatMatrix):
    r"""Nadaraya-Watson method.

    Creates the matrix :math:`\hat{H}`, used in the kernel smoothing and kernel
    regression algorithms, as explained below

    .. math::
        \hat{H}_{i,j} = \frac{K\left(\frac{d(e_j-e_i')}{h}\right)}{\sum_{k=1}^{
        n}K\left(\frac{d(e_k-e_i')}{h}\right)}

    For smoothing, :math:`e_i` are the points of discretisation
    and :math:`e'_i` are the points for which it is desired to estimate the
    smoothed value. The distance :math:`d` is the absolute value
    function :footcite:`wasserman_2006_nonparametric_nw`.

    For regression, :math:`e_i` is the functional data and :math:`e_i'`
    are the functions for which it is desired to estimate the scalar value.
    Here, :math:`d` is some functional distance
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

    def _hat_matrix_function_not_normalized(
        self,
        *,
        delta_x: np.ndarray,
    ) -> np.ndarray:

        if self.bandwidth is None:
            percentage = 15
            self.bandwidth = np.percentile(np.abs(delta_x), percentage)

        return self.kernel(delta_x / self.bandwidth)


class LocalLinearRegressionHatMatrix(HatMatrix):
    r"""Local linear regression method.

    Creates the matrix :math:`\hat{H}`, used in the kernel smoothing and kernel
    regression algorithms, as explained below.

    For **kernel smoothing** algorithm to estimate the smoothed value for
    :math:`t_j` the following error must be minimised

    .. math::
        AWSE(a, b) = \sum_{i=1}^n \left[ \left(y_i -
        \left(a +  b (t_i - t'_j) \right) \right)^2
        K \left( \frac {|t_i - t'_j|}{h} \right) \right ]

    which gives the following expression for each cell

    .. math::
        \hat{H}_{i,j} = \frac{b_j(t_i')}{\sum_{k=1}^{n}b_k(t_i')}

    .. math::
        b_j(e') = K\left(\frac{t_j - t'}{h}\right) S_{n,2}(t') -
        (t_j - t')S_{n,1}(t')

    .. math::
        S_{n,k}(t') = \sum_{j=1}^{n}K\left(\frac{t_j-t'}{h}\right)(t_j-t')^k

    where :math:`t = (t_1, t_2, ..., t_n)` are points of discretisation and
    :math:`t' = (t_1', t_2', ..., t_m')` are the points for which it is desired
    to estimate the smoothed value
    :footcite:`wasserman_2006_nonparametric_llr`.

    For **kernel regression** algorithm:

    Given functional data, :math:`(X_1, X_2, ..., X_n)` where each function
    is expressed in a orthonormal basis with :math:`J` elements and scalar
    response :math:`Y = (y_1, y_2, ..., y_n)`.

    It is desired to estimate the values
    :math:`\hat{Y} = (\hat{y}_1, \hat{y}_2, ..., \hat{y}_m)`
    for the data :math:`(X'_1, X'_2, ..., X'_m)` (expressed in the same basis).

    For each :math:`X'_k` the estimation :math:`\hat{y}_k` is obtained by
    taking the value :math:`a_k` from the vector
    :math:`(a_k, b_{1k}, ..., b_{Jk})` which minimizes the following expression

    .. math::
        AWSE(a_k, b_{1k}, ..., b_{Jk}) = \sum_{i=1}^n \left(y_i -
        \left(a + \sum_{j=1}^J b_{jk} c_{ijk} \right) \right)^2
        K \left( \frac {d(X_i - X'_k)}{h} \right)

    Where :math:`c_{ij}^k` is the :math:`j`-th coefficient in a truncated basis
    expansion of :math:`X_i - X'_k = \sum_{j=1}^J c_{ij}^k` and :math:`d` some
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

    def __call__(  # noqa: D102
        self,
        *,
        delta_x: np.ndarray,
        X_train: Optional[Union[FDataBasis, GridPoints]] = None,
        X: Optional[Union[FDataBasis, GridPoints]] = None,
        y_train: Optional[np.ndarray] = None,
        weights: Optional[np.ndarray] = None,
        _cv: bool = False,
    ) -> np.ndarray:

        if self.bandwidth is None:
            percentage = 15
            self.bandwidth = np.percentile(np.abs(delta_x), percentage)

        # Regression
        if isinstance(X_train, FDataBasis):

            if y_train is None:
                y_train = np.identity(X_train.n_samples)

            m1 = X_train.coefficients
            m2 = X.coefficients

            return self._solve_least_squares(
                delta_x=delta_x,
                m1=m1,
                m2=m2,
                y_train=y_train,
            )

        # Smoothing
        else:

            return super().__call__(  # noqa: WPS503
                delta_x=delta_x,
                X_train=X_train,
                X=X,
                y_train=y_train,
                weights=weights,
                _cv=_cv,
            )

    def _solve_least_squares(
        self,
        delta_x: np.ndarray,
        m1: np.ndarray,
        m2: np.ndarray,
        y_train: np.ndarray,
    ) -> np.ndarray:

        W = np.sqrt(self.kernel(delta_x / self.bandwidth))

        # Adding a column of ones to m1
        m1 = np.concatenate(
            (
                np.ones(m1.shape[0])[:, np.newaxis],
                m1,
            ),
            axis=1,
        )

        # Adding a column of zeros to m2
        m2 = np.concatenate(
            (
                np.zeros(m2.shape[0])[:, np.newaxis],
                m2,
            ),
            axis=1,
        )

        # Subtract previous matrices obtaining a 3D matrix
        # The i-th element contains the matrix X_train - X[i]
        C = m1 - m2[:, np.newaxis]

        # A x = b
        # Where x = (a, b_1, ..., b_J)
        A = (C.T * W.T).T
        b = np.einsum('ij, j... -> ij...', W, y_train)

        # For Ax = b calculates x that minimize the square error
        # From https://stackoverflow.com/questions/42534237/broadcasted-lstsq-least-squares  # noqa: E501
        u, s, vT = np.linalg.svd(A, full_matrices=False)

        uTb = np.einsum('ijk, ij...->ik...', u, b)
        uTbs = (uTb.T / s.T).T
        x = np.einsum('ijk,ij...->ik...', vT, uTbs)

        return x[:, 0]

    def _hat_matrix_function_not_normalized(
        self,
        *,
        delta_x: np.ndarray,
    ) -> np.ndarray:

        if self.bandwidth is None:
            percentage = 15
            self.bandwidth = np.percentile(np.abs(delta_x), percentage)

        k = self.kernel(delta_x / self.bandwidth)

        s1 = np.sum(k * delta_x, axis=1, keepdims=True)  # S_n_1
        s2 = np.sum(k * delta_x ** 2, axis=1, keepdims=True)  # S_n_2
        b = (k * (s2 - delta_x * s1))  # b_i(x_j)

        return b  # noqa: WPS331


class KNeighborsHatMatrix(HatMatrix):
    r"""K-nearest neighbour kernel method.

    Creates the matrix :math:`\hat{H}`, used in the kernel smoothing and kernel
    regression algorithms, as explained below.

    .. math::
        \hat{H}_{i,j} = \frac{K\left(\frac{d(e_j-e_i')}{h}\right)}{\sum_{k=1}^{
        n}K\left(\frac{d(e_k-e_i')}{h_{ik}}\right)}

    For smoothing, :math:`e_i` are the points of discretisation
    and :math:`e'_i` are the points for which it is desired to estimate the
    smoothed value. The distance :math:`d` is the absolute value.

    For regression, :math:`e_i` are the functional data and :math:`e_i'`
    are the functions for which it is desired to estimate the scalar value.
    Here, :math:`d` is some functional distance.

    In both cases, :math:`K(\cdot)` is a kernel function and
    :math:`h_{ik}` is calculated as the distance from :math:`e_i'` to it's
    :math:`k`-th nearest neighbour in :math:`\{e_1, ..., e_n\}`
    :footcite:`ferraty+vieu_2006_nonparametric_knn`.

    Used with the uniform kernel, it takes the average of the closest k
    points to a given point.

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
        n_neighbors: Optional[int] = None,
        kernel: Callable[[np.ndarray], np.ndarray] = kernels.uniform,
    ):
        self.n_neighbors = n_neighbors
        self.kernel = kernel

    def _hat_matrix_function_not_normalized(
        self,
        *,
        delta_x: np.ndarray,
    ) -> np.ndarray:

        input_points_len = delta_x.shape[1]

        if self.n_neighbors is None:
            self.n_neighbors = np.floor(
                np.percentile(
                    range(1, input_points_len),
                    5,
                ),
            )
        elif self.n_neighbors <= 0:
            raise ValueError('h must be greater than 0')

        # Tolerance to avoid points landing outside the kernel window due to
        # computation error
        tol = np.finfo(np.float64).eps

        # For each row in the distances matrix, it calculates the furthest
        # point within the k nearest neighbours
        vec = np.percentile(
            np.abs(delta_x),
            self.n_neighbors / input_points_len * 100,
            axis=1,
            interpolation='lower',
        ) + tol

        return self.kernel((delta_x.T / vec).T)
