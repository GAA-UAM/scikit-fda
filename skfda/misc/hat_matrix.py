# -*- coding: utf-8 -*-

"""Hat Matrix.

This module include implementation to create Nadaraya-Watson,
Local Linear Regression and K-Nearest Neighbours hat matrices used in
kernel smoothing and kernel regression.

"""

import abc
from typing import Callable, Optional

import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin

from skfda.representation._functional_data import FData
from skfda.representation.basis import FDataBasis

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
        kernel: Optional[Callable] = None,
    ):
        self.bandwidth = bandwidth
        self.kernel = kernel

    def __call__(
        self,
        delta_x: np.ndarray,
        weights: Optional[np.ndarray] = None,
        _cv: bool = False,
    ) -> np.ndarray:
        """
        Hat matrix.

        Calculate and return matrix for smoothing (for all methods) and
        regression (for Nadaraya-Watson and K-Nearest Neighbours)

        Args:
            delta_x (np.ndarray): Matrix of distances between points or
                functions
            weights (np.ndarray, optional): Weights to be applied to the
                resulting matrix columns

        Returns:
            hat matrix (np.ndarray)

        """
        # Obtain the non-normalized matrix
        matrix = self._hat_matrix_function_not_normalized(delta_x=delta_x)

        # Adjust weights
        if weights is not None:
            matrix *= weights

        # Set diagonal to zero if requested (for testing purposes only)
        if _cv:
            np.fill_diagonal(matrix, 0)

        # Renormalize weights
        rs = np.sum(matrix, axis=1)
        rs[rs == 0] = 1
        return (matrix.T / rs).T

    def prediction(
        self,
        *,
        delta_x: np.ndarray,
        y_train: np.ndarray,
        X_train: Optional[FData] = None,
        X: Optional[FData] = None,
        weights: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """
        Prediction.

        Return the resulting :math:`y` array, by default calculates the hat
        matrix and  multiplies it by y_train (except for regression using
        Local Linear Regression method).

        Args:
            delta_x (np.ndarray): Matrix of distances between points or
                functions
            y_train (np.ndarray): Scalar response from for functional data
            X_train (FData , optional): Functional data. Only used in
                regression with Local Linear Regression method
            X (FData, optional): Functional data. Only used in regression
                with Local Linear Regression method
            weights (np.ndarray, optional): Weights to be applied to the
                resulting matrix columns

        Returns:
            prediction (np.ndarray)

        """
        return np.dot(self.__call__(delta_x=delta_x, weights=weights), y_train)

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
    regression algorithms, as explained below.

    For **kernel smoothing** algorithm the matrix :math:`\hat{H}` has the
    following expression for each cell
    :footcite:`wasserman_2006_nonparametric_nw`:

    .. math::
        \hat{H}_{i,j} = \frac{K\left(\frac{t_j-t_i'}{h}\right)}{\sum_{k=1}^{
        n}K\left(\frac{t_k-t_i'}{h}\right)}

    where :math:`t = (t_1, t_2, ..., t_n)` are points of discretisation and
    :math:`t' = (t_1', t_2', ..., t_m')` are the points for which it is desired
    to estimate the smoothed value.

    The result can be obtained as

    .. math::
        \hat{X} = \hat{H}X

    where :math:`X = (x_1, x_2, ..., x_n)` is the vector of observations at the
    points :math:`t` and
    :math:`\hat{X} = (\hat{x}_1, \hat{x}_2, ..., \hat{x}_m)` are the estimated
    values for the points :math:`t'`.

    For **kernel regression** algorithm
    :footcite:`ferraty+vieu_2006_nonparametric_nw`:,

    .. math::
        \hat{H}_{i,j} = \frac{K\left(\frac{d(f_j-f_i')}{h}\right)}{\sum_{k=1}^{
        n}K\left(\frac{d(f_k-f_i')}{h}\right)}

    where :math:`d(\cdot, \cdot)` is some functional distance
    (see :class:`~skfda.misc.metrics.LpDistance`),
    :math:`(f_1, f_2, ..., f_n)` is the functional data and the functions
    :math:`(f_1', f_2', ..., f_m')` are the functions for which it is desired
    to estimate the scalar value.

    The result can be obtained as

    .. math::
        \hat{Y} = \hat{H}Y

    where :math:`Y = (y_1, y_2, ..., y_n)` is the vector of scalar values
    corresponding to the dataset and
    :math:`\hat{Y} = (\hat{y}_1, \hat{y}_2, ..., \hat{y}_m)` are the estimated
    values

    In both cases, :math:`K(\cdot)` is a kernel function and :math:`h` the
    kernel window width or smoothing parameter.

    Args:
        bandwidth (float, optional): Window width of the kernel
            (also called h or bandwidth).
        kernel (function, optional): Kernel function. By default a normal
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
            self.bandwidth = np.percentile(delta_x, percentage)

        if self.kernel is None:
            self.kernel = kernels.normal

        return self.kernel(delta_x / self.bandwidth)


class LocalLinearRegressionHatMatrix(HatMatrix):
    r"""Local linear regression method.

    Creates the matrix :math:`\hat{H}`, used in the kernel smoothing and kernel
    regression algorithms, as explained below.

    For **kernel smoothing** algorithm the matrix :math:`\hat{H}` has the
    following expression for each cell
    :footcite:`wasserman_2006_nonparametric_llr`:

    .. math::
        \hat{H}_{i,j} = \frac{b_j(t_i')}{\sum_{k=1}^{n}b_k(t_i')}

    .. math::
        b_j(e') = K\left(\frac{t_j - t'}{h}\right) S_{n,2}(t') -
        (t_j - t')S_{n,1}(t')

    .. math::
        S_{n,k}(t') = \sum_{j=1}^{n}K\left(\frac{t_j-t'}{h}\right)(t_j-t')^k

    where :math:`t = (t_1, t_2, ..., t_n)` are points of discretisation and
    :math:`t' = (t_1', t_2', ..., t_m')` are the points for which it is desired
    to estimate the smoothed value.

    The result can be obtained as

    .. math::
        \hat{X} = \hat{H}X

    where :math:`X = (x_1, x_2, ..., x_n)` is the vector of observations at the
    points :math:`t` and
    :math:`\hat{X} = (\hat{x}_1, \hat{x}_2, ..., \hat{x}_m)` are the estimated
    values for the points :math:`t'`.

    For **kernel regression** algorithm
    :footcite:`baillo+grane+2008+llr`:

    Given functional data, :math:`(f_1, f_2, ..., f_n)` where each function
    is expressed in a orthonormal basis with :math:`J` elements and scalar
    response :math:`Y = (y_1, y_2, ..., y_n)`.

    It is desired to estimate the values
    :math:`\hat{Y} = (\hat{y}_1, \hat{y}'_2, ..., \hat{y}'_m)`
    for the data :math:`(f'_1, f'_2, ..., f'_m)` (expressed in the same basis).

    For each :math:`f'_k` the estimation :math:`\hat{y}_k` is obtained by
    taking the value :math:`a^k` from the vector
    :math:`(a^k, b_1^k, ..., b_J^k)` which minimizes the following expression

    .. math::
        AWSE(a^k, b_1^k, ..., b_J^k) = \sum_{i=1}^n \left(y_i -
        \left(a + \sum_{j=1}^J b_j^k c_{ij}^k \right) \right)^2
        K \left( \frac {d(f_i - f'_k)}{h} \right) 1/h

    Where:

    -   :math:`K(\cdot)` is a kernel function, :math:`h`
        the kernel window width or bandwidth and :math:`d` some
        functional distance
    -   :math:`c_{ij}^k` is the :math:`j`-th coefficient in a truncated basis
        expansion of :math:`f_i - f'_k = \sum_{j=1}^J c_{ij}^k`

    Args:
        bandwidth (float, optional): Window width of the kernel
            (also called h or bandwidth).
        kernel (function, optional): Kernel function. By default a normal
            kernel.

    References:
        .. footbibliography::

    """

    def prediction(
        self,
        *,
        delta_x: np.ndarray,
        X_train: FDataBasis,
        y_train: np.ndarray,
        X: FDataBasis,
        weights: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """
        Prediction.

        Return the resulting :math:`y` array

        Args:
            delta_x: np.ndarray
            X_train: np.ndarray
            y_train: np.ndarray
            X: np.ndarray
            weights: np.ndarray

        Returns:
            np.ndarray
        """
        if self.bandwidth is None:
            percentage = 15
            self.bandwidth = np.percentile(delta_x, percentage)

        if self.kernel is None:
            self.kernel = kernels.normal

        W = np.sqrt(self.kernel(delta_x / self.bandwidth))

        # Creating the matrices of coefficients
        m1 = X_train.coefficients
        m2 = X.coefficients

        # Adding a column of ones to X_train coefficients
        m1 = np.concatenate(
            (
                np.ones(X_train.n_samples)[:, np.newaxis],
                m1,
            ),
            axis=1,
        )

        # Adding a column of zeros to X coefficients
        m2 = np.concatenate((np.zeros(X.n_samples)[:, np.newaxis], m2), axis=1)

        # Subtract previous matrices obtaining a 3D matrix
        # The i-th element contains the matrix X_train - X[i]
        C = m1 - m2[:, np.newaxis]

        # A x = b
        # Where x = (a, b_1, ..., b_J)
        A = (C.T * W.T).T
        b = W * y_train

        # From https://stackoverflow.com/questions/42534237/broadcasted-lstsq-least-squares  # noqa: E501
        u, s, vT = np.linalg.svd(A, full_matrices=False)

        uTb = np.einsum('ijk,ij->ik', u, b)
        x = np.einsum('ijk,ij->ik', vT, uTb / s)

        return x[:, 0]

    def _hat_matrix_function_not_normalized(
        self,
        *,
        delta_x: np.ndarray,
    ) -> np.ndarray:

        if self.bandwidth is None:
            percentage = 15
            self.bandwidth = np.percentile(delta_x, percentage)

        if self.kernel is None:
            self.kernel = kernels.normal

        k = self.kernel(delta_x / self.bandwidth)

        s1 = np.sum(k * delta_x, axis=1, keepdims=True)  # S_n_1
        s2 = np.sum(k * delta_x ** 2, axis=1, keepdims=True)  # S_n_2
        b = (k * (s2 - delta_x * s1))  # b_i(x_j)

        return b  # noqa: WPS331


class KNeighborsHatMatrix(HatMatrix):
    r"""K-nearest neighbour kernel method.

    Creates the matrix :math:`\hat{H}`, used in the kernel smoothing and kernel
    regression algorithms, as explained below.

    For **kernel smoothing** algorithm the matrix :math:`\hat{H}` has the
    following expression for each cell

    .. math::
        \hat{H}_{i,j} = \frac{K\left(\frac{t_j-t_i'}{h}\right)}{\sum_{k=1}^{
        n}K\left(\frac{t_k-t_i'}{h}\right)}

    where :math:`t = (t_1, t_2, ..., t_n)` are points of discretisation and
    :math:`t' = (t_1', t_2', ..., t_m')` are the points for which it is desired
    to estimate the smoothed value.

    The result can be obtained as

    .. math::
        \hat{X} = \hat{H}X

    where :math:`X = (x_1, x_2, ..., x_n)` is the vector of observations at the
    points :math:`t` and
    :math:`\hat{X} = (\hat{x}_1, \hat{x}_2, ..., \hat{x}_m)` are the estimated
    values for the points :math:`t'`.


    For **kernel regression** algorithm
    :footcite:`ferraty+vieu_2006_nonparametric_knn`,

    .. math::
        \hat{H}_{i,j} = \frac{K\left(\frac{d(f_j-f_i')}{h_{ik}}\right)}
        {\sum_{k=1}^{n}K\left(\frac{d(f_k-f_i')}{h_{ik}}\right)}

    where :math:`d(\cdot, \cdot)` is some functional distance
    (see :class:`~skfda.misc.metrics.LpDistance`),
    :math:`F = (f_1, f_2, ..., f_n)` is the functional data and the functions
    :math:`F' = (f_1', f_2', ..., f_m')` are the functions for which it is
    wanted to estimate the scalar value.

    The result can be obtained as

    .. math::
        \hat{Y} = \hat{H}Y

    where :math:`Y = (y_1, y_2, ..., y_n)` is the vector of scalar values
    corresponding to the dataset and
    :math:`\hat{Y} = (\hat{y}_1, \hat{y}_2, ..., \hat{y}_m)` are the estimated
    values

    In both cases, :math:`K(\cdot)` is a kernel function, for smoothing,
    :math:`h_{ik}` is calculated as the distance from :math:`t_i'` to it's 𝑘-th
    nearest neighbor in :math:`t`, and for regression, :math:`h_{ik}` is
    calculated as the distance from :math:`f_i'` to it's 𝑘-th  nearest neighbor
    in :math:`F`.

    Usually used with the uniform kernel, it takes the average of the closest k
    points to a given point.

    Args:
        bandwidth (int, optional): Number of nearest neighbours. By
            default it takes the 5% closest points.
        kernel (function, optional): Kernel function. By default a uniform
            kernel to perform a 'usual' k nearest neighbours estimation.

    References:
        .. footbibliography::

    """

    def _hat_matrix_function_not_normalized(
        self,
        *,
        delta_x: np.ndarray,
    ) -> np.ndarray:

        if self.kernel is None:
            self.kernel = kernels.uniform

        input_points_len = delta_x.shape[1]

        if self.bandwidth is None:
            self.bandwidth = np.floor(
                np.percentile(
                    range(1, input_points_len),
                    5,
                ),
            )
        elif self.bandwidth <= 0:
            raise ValueError('h must be greater than 0')

        # Tolerance to avoid points landing outside the kernel window due to
        # computation error
        tol = 1.0e-15

        # For each row in the distances matrix, it calculates the furthest
        # point within the k nearest neighbours
        vec = np.percentile(
            np.abs(delta_x),
            self.bandwidth / input_points_len * 100,
            axis=1,
            interpolation='lower',
        ) + tol

        return self.kernel((delta_x.T / vec).T)
