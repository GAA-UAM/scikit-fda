# -*- coding: utf-8 -*-
"""Kernel smoother functions.

This module includes the most commonly used kernel smoother methods for FDA.
 So far only non parametric methods are implemented because we are only
 relying on a discrete representation of functional data.

Todo:
    * Closed-form for KNN

"""
import numpy as np

from ...misc import kernels
from sklearn.base import BaseEstimator, TransformerMixin
from skfda.representation.grid import FDataGrid
import abc

__author__ = "Miguel Carbajo Berrocal"
__email__ = "miguel.carbajo@estudiante.uam.es"


def _check_r_to_r(f):
    if f.ndim_domain != 1 or f.ndim_codomain != 1:
        raise NotImplementedError("Only accepts functions from R to R")


class _LinearKernelSmoother(abc.ABC, BaseEstimator, TransformerMixin):

    def __init__(self, *, smoothing_parameter=None,
                 kernel=kernels.normal, weights=None):
        self.smoothing_parameter = smoothing_parameter
        self.kernel = kernel
        self.weights = weights
        self._cv = False  # For testing purposes only

    def _hat_matrix_function(self, *, input_points, smoothing_parameter,
                             kernel, weights, _cv=False):

        # Time deltas
        delta_x = np.abs(np.subtract.outer(input_points, input_points))

        # Obtain the non-normalized matrix
        matrix = self._hat_matrix_function_not_normalized(
            delta_x=delta_x,
            smoothing_parameter=smoothing_parameter,
            kernel=kernel)

        # Adjust weights
        if weights is not None:
            matrix = matrix * weights

        # Set diagonal to cero if requested (for testing purposes only)
        if _cv:
            np.fill_diagonal(matrix, 0)

        # Renormalize weights
        rs = np.sum(matrix, 1)
        rs[rs == 0] = 1
        return (matrix.T / rs).T

    @abc.abstractmethod
    def _hat_matrix_function_not_normalized(self, *, delta_x,
                                            smoothing_parameter, kernel):
        pass

    def _more_tags(self):
        return {
            'X_types': []
            }

    def fit(self, X: FDataGrid, y=None):

        _check_r_to_r(X)

        self.input_points_ = X.sample_points[0]

        self.hat_matrix_ = self._hat_matrix_function(
                input_points=self.input_points_,
                smoothing_parameter=self.smoothing_parameter,
                kernel=self.kernel,
                weights=self.weights,
                _cv=self._cv
            )

        return self

    def transform(self, X: FDataGrid, y=None):

        assert all(self.input_points_ == X.sample_points[0])

        return X.copy(data_matrix=self.hat_matrix_ @ X.data_matrix)


class NadarayaWatsonSmoother(_LinearKernelSmoother):
    r"""Nadaraya-Watson smoothing method.

    Uses an smoothing matrix :math:`\hat{H}` for the discretisation
    points in argvals by the Nadaraya-Watson estimator. The smoothed
    values :math:`\hat{Y}` can be calculated as :math:`\hat{
    Y} = \hat{H}Y` where :math:`Y` is the vector of observations at the
    points of discretisation :math:`(x_1, x_2, ..., x_n)`.

    .. math::
        \hat{H}_{i,j} = \frac{K\left(\frac{x_i-x_j}{h}\right)}{\sum_{k=1}^{
        n}K\left(
        \frac{x_1-x_k}{h}\right)}

    where :math:`K(\cdot)` is a kernel function and :math:`h` the kernel
    window width or smoothing parameter.

    Args:
        argvals (ndarray): Vector of discretisation points.
        smoothing_parameter (float, optional): Window width of the kernel
            (also called h or bandwidth).
        kernel (function, optional): kernel function. By default a normal
            kernel.
        weights (ndarray, optional): Case weights matrix (in order to modify
            the importance of each point).

    Examples:
        >>> smoother = NadarayaWatsonSmoother(smoothing_parameter=3.5)
        >>> _ = smoother.fit(FDataGrid(sample_points=[1,2,4,5,7],
        ...                            data_matrix=[[0,0,0,0,0]]))
        >>> smoother.hat_matrix_.round(3)
        array([[ 0.294, 0.282, 0.204, 0.153, 0.068],
               [ 0.249, 0.259, 0.22 , 0.179, 0.093],
               [ 0.165, 0.202, 0.238, 0.229, 0.165],
               [ 0.129, 0.172, 0.239, 0.249, 0.211],
               [ 0.073, 0.115, 0.221, 0.271, 0.319]])
        >>> smoother = NadarayaWatsonSmoother(smoothing_parameter=2)
        >>> _ = smoother.fit(FDataGrid(sample_points=[1,2,4,5,7],
        ...                            data_matrix=[[0,0,0,0,0]]))
        >>> smoother.hat_matrix_.round(3)
        array([[ 0.425, 0.375, 0.138, 0.058, 0.005],
               [ 0.309, 0.35 , 0.212, 0.114, 0.015],
               [ 0.103, 0.193, 0.319, 0.281, 0.103],
               [ 0.046, 0.11 , 0.299, 0.339, 0.206],
               [ 0.006, 0.022, 0.163, 0.305, 0.503]])

    """
    def _hat_matrix_function_not_normalized(self, *, delta_x,
                                            smoothing_parameter,
                                            kernel):
        if smoothing_parameter is None:
            smoothing_parameter = np.percentile(delta_x, 15)

        k = kernel(delta_x / smoothing_parameter)
        return k


class LocalLinearRegressionSmoother(_LinearKernelSmoother):
    r"""Local linear regression smoothing method.

    Uses an smoothing matrix :math:`\hat{H}` for the discretisation
    points in argvals by the local linear regression estimator. The smoothed
    values :math:`\hat{Y}` can be calculated as :math:`\hat{
    Y} = \hat{H}Y` where :math:`Y` is the vector of observations at the points
    of discretisation :math:`(x_1, x_2, ..., x_n)`.

    .. math::
        \hat{H}_{i,j} = \frac{b_i(x_j)}{\sum_{k=1}^{n}b_k(x_j)}

    .. math::
        b_i(x) = K\left(\frac{x_i - x}{h}\right) S_{n,2}(x) - (x_i - x)S_{n,
        1}(x)

    .. math::
        S_{n,k} = \sum_{i=1}^{n}K\left(\frac{x_i-x}{h}\right)(x_i-x)^k

    where :math:`K(\cdot)` is a kernel function and :math:`h` the kernel
    window width.

    Args:
        argvals (ndarray): Vector of discretisation points.
        smoothing_parameter (float, optional): Window width of the kernel
            (also called h or bandwidth).
        kernel (function, optional): kernel function. By default a normal
            kernel.
        weights (ndarray, optional): Case weights matrix (in order to modify
            the importance of each point).

    Examples:
        >>> smoother = LocalLinearRegressionSmoother(smoothing_parameter=3.5)
        >>> _ = smoother.fit(FDataGrid(sample_points=[1,2,4,5,7],
        ...                            data_matrix=[[0,0,0,0,0]]))
        >>> smoother.hat_matrix_.round(3)
        array([[ 0.614,  0.429,  0.077, -0.03 , -0.09 ],
               [ 0.381,  0.595,  0.168, -0.   , -0.143],
               [-0.104,  0.112,  0.697,  0.398, -0.104],
               [-0.147, -0.036,  0.392,  0.639,  0.152],
               [-0.095, -0.079,  0.117,  0.308,  0.75 ]])
        >>> smoother = LocalLinearRegressionSmoother(smoothing_parameter=2)
        >>> _ = smoother.fit(FDataGrid(sample_points=[1,2,4,5,7],
        ...                            data_matrix=[[0,0,0,0,0]]))
        >>> smoother.hat_matrix_.round(3)
        array([[ 0.714,  0.386, -0.037, -0.053, -0.01 ],
               [ 0.352,  0.724,  0.045, -0.081, -0.04 ],
               [-0.078,  0.052,  0.74 ,  0.364, -0.078],
               [-0.07 , -0.067,  0.36 ,  0.716,  0.061],
               [-0.012, -0.032, -0.025,  0.154,  0.915]])

    """

    def _hat_matrix_function_not_normalized(self, *, delta_x,
                                            smoothing_parameter, kernel):
        k = kernel(delta_x / smoothing_parameter)

        s1 = np.sum(k * delta_x, 1)  # S_n_1
        s2 = np.sum(k * delta_x ** 2, 1)  # S_n_2
        b = (k * (s2 - delta_x * s1)).T  # b_i(x_j)
        return b


class KNeighborsSmoother(_LinearKernelSmoother):
    """K-nearest neighbour kernel smoother.

    Uses an smoothing matrix S for the discretisation points in argvals by
    the k nearest neighbours estimator.

    Usually used with the uniform kernel, it takes the average of the closest k
    points to a given point.

    Args:
        argvals (ndarray): Vector of discretisation points.
        smoothing_parameter (int, optional): Number of nearest neighbours. By
            default it takes the 5% closest points.
        kernel (function, optional): kernel function. By default a uniform
            kernel to perform a 'usual' k nearest neighbours estimation.
        weights (ndarray, optional): Case weights matrix (in order to modify
            the importance of each point).

    Examples:
        >>> smoother = KNeighborsSmoother(smoothing_parameter=2)
        >>> _ = smoother.fit(FDataGrid(sample_points=[1,2,4,5,7],
        ...                            data_matrix=[[0,0,0,0,0]]))
        >>> smoother.hat_matrix_.round(3)
        array([[ 0.5, 0.5, 0. , 0. , 0. ],
               [ 0.5, 0.5, 0. , 0. , 0. ],
               [ 0. , 0. , 0.5, 0.5, 0. ],
               [ 0. , 0. , 0.5, 0.5, 0. ],
               [ 0. , 0. , 0. , 0.5, 0.5]])

        In case there are two points at the same distance it will take both.

        >>> smoother = KNeighborsSmoother(smoothing_parameter=2)
        >>> _ = smoother.fit(FDataGrid(sample_points=[1,2,3,5,7],
        ...                            data_matrix=[[0,0,0,0,0]]))
        >>> smoother.hat_matrix_.round(3)
        array([[ 0.5  , 0.5  , 0.   , 0.   , 0.   ],
               [ 0.333, 0.333, 0.333, 0.   , 0.   ],
               [ 0.   , 0.5  , 0.5  , 0.   , 0.   ],
               [ 0.   , 0.   , 0.333, 0.333, 0.333],
               [ 0.   , 0.   , 0.   , 0.5  , 0.5  ]])

    """
    def __init__(self, *, smoothing_parameter=None,
                 kernel=kernels.uniform, weights=None):
        super().__init__(
            smoothing_parameter=smoothing_parameter,
            kernel=kernel,
            weights=weights
        )

    def _hat_matrix_function_not_normalized(self, *, delta_x,
                                            smoothing_parameter, kernel):

        input_points_len = delta_x.shape[1]

        if smoothing_parameter is None:
            smoothing_parameter = np.floor(np.percentile(
                range(1, input_points_len), 5))
        elif smoothing_parameter <= 0:
            raise ValueError('h must be greater than 0')

        # Tolerance to avoid points landing outside the kernel window due to
        # computation error
        tol = 1.0e-19

        # For each row in the distances matrix, it calculates the furthest
        # point within the k nearest neighbours
        vec = np.percentile(delta_x, smoothing_parameter
                            / input_points_len * 100,
                            axis=0, interpolation='lower') + tol

        rr = kernel((delta_x.T / vec).T)
        # Applies the kernel to the result of dividing each row by the result
        # of the previous operation, all the discretisation points
        # corresponding to the knn are below 1 and the rest above 1 so the
        # kernel returns values distinct to 0 only for the knn.

        return rr
