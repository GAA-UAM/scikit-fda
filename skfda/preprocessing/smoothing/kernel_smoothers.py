# -*- coding: utf-8 -*-
"""Kernel smoother functions.

This module includes the most commonly used kernel smoother methods for FDA.
 So far only non parametric methods are implemented because we are only
 relaying on a discrete representation of functional data.

Todo:
    * Closed-form for KNN

"""
import math

import numpy

from ...misc import kernels


__author__ = "Miguel Carbajo Berrocal"
__email__ = "miguel.carbajo@estudiante.uam.es"


def nw(argvals, h=None, kernel=kernels.normal, w=None, cv=False):
    r"""Nadaraya-Watson smoothing method.

    Provides an smoothing matrix :math:`\hat{H}` for the discretisation
    points in argvals by the Nadaraya-Watson estimator. The smoothed
    values :math:`\hat{Y}` can be calculated as :math:`\hat{
    Y} = \hat{H}Y` where :math:`Y` is the vector of observations at the
    points of discretisation :math:`(x_1, x_2, ..., x_n)`.

    .. math::
        \hat{H}_{i,j} = \frac{K\left(\frac{x_i-x_j}{h}\right)}{\sum_{k=1}^{
        n}K\left(
        \frac{x_1-x_k}{h}\right)}

    where :math:`K(\cdot)` is a kernel function and :math:`h` the kernel
    window width.

    Args:
        argvals (ndarray): Vector of discretisation points.
        h (float, optional): Window width of the kernel.
        kernel (function, optional): kernel function. By default a normal
            kernel.
        w (ndarray, optional): Case weights matrix.
        cv (bool, optional): Flag for cross-validation methods.
            Defaults to False.

    Examples:
        >>> nw(numpy.array([1,2,4,5,7]), 3.5).round(3)
        array([[ 0.294, 0.282, 0.204, 0.153, 0.068],
               [ 0.249, 0.259, 0.22 , 0.179, 0.093],
               [ 0.165, 0.202, 0.238, 0.229, 0.165],
               [ 0.129, 0.172, 0.239, 0.249, 0.211],
               [ 0.073, 0.115, 0.221, 0.271, 0.319]])
        >>> nw(numpy.array([1,2,4,5,7]), 2).round(3)
        array([[ 0.425, 0.375, 0.138, 0.058, 0.005],
               [ 0.309, 0.35 , 0.212, 0.114, 0.015],
               [ 0.103, 0.193, 0.319, 0.281, 0.103],
               [ 0.046, 0.11 , 0.299, 0.339, 0.206],
               [ 0.006, 0.022, 0.163, 0.305, 0.503]])

    Returns:
        ndarray: Smoothing matrix :math:`\hat{H}`.

    """
    delta_x = numpy.abs(numpy.subtract.outer(argvals, argvals))
    if h is None:
        h = numpy.percentile(delta_x, 15)
    if cv:
        numpy.fill_diagonal(delta_x, math.inf)
    delta_x = delta_x / h
    k = kernel(delta_x)
    if w is not None:
        k = k * w
    rs = numpy.sum(k, 1)
    rs[rs == 0] = 1
    return (k.T / rs).T


def local_linear_regression(argvals, h, kernel=kernels.normal, w=None,
                            cv=False):
    r"""Local linear regression smoothing method.

    Provides an smoothing matrix :math:`\hat{H}` for the discretisation
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
        h (float, optional): Window width of the kernel.
        kernel (function, optional): kernel function. By default a normal
            kernel.
        w (ndarray, optional): Case weights matrix.
        cv (bool, optional): Flag for cross-validation methods.
            Defaults to False.

    Examples:
        >>> local_linear_regression(numpy.array([1,2,4,5,7]), 3.5).round(3)
        array([[ 0.614,  0.429,  0.077, -0.03 , -0.09 ],
               [ 0.381,  0.595,  0.168, -0.   , -0.143],
               [-0.104,  0.112,  0.697,  0.398, -0.104],
               [-0.147, -0.036,  0.392,  0.639,  0.152],
               [-0.095, -0.079,  0.117,  0.308,  0.75 ]])
        >>> local_linear_regression(numpy.array([1,2,4,5,7]), 2).round(3)
        array([[ 0.714,  0.386, -0.037, -0.053, -0.01 ],
               [ 0.352,  0.724,  0.045, -0.081, -0.04 ],
               [-0.078,  0.052,  0.74 ,  0.364, -0.078],
               [-0.07 , -0.067,  0.36 ,  0.716,  0.061],
               [-0.012, -0.032, -0.025,  0.154,  0.915]])


    Returns:
        ndarray: Smoothing matrix :math:`\hat{H}`.

    """
    delta_x = numpy.abs(numpy.subtract.outer(argvals, argvals))  # x_i - x_j
    if cv:
        numpy.fill_diagonal(delta_x, math.inf)
    k = kernel(delta_x / h)  # K(x_i - x/ h)
    s1 = numpy.sum(k * delta_x, 1)  # S_n_1
    s2 = numpy.sum(k * delta_x ** 2, 1)  # S_n_2
    b = (k * (s2 - delta_x * s1)).T  # b_i(x_j)
    if cv:
        numpy.fill_diagonal(b, 0)
    if w is not None:
        b = b * w
    rs = numpy.sum(b, 1)  # sum_{k=1}^{n}b_k(x_j)
    return (b.T / rs).T  # \\hat{H}


def knn(argvals, k=None, kernel=kernels.uniform, w=None, cv=False):
    """K-nearest neighbour kernel smoother.

    Provides an smoothing matrix S for the discretisation points in argvals by
    the k nearest neighbours estimator.

    Usually used with the uniform kernel, it takes the average of the closest k
    points to a given point.



    Args:
        argvals (ndarray): Vector of discretisation points.
        k (int, optional): Number of nearest neighbours. By default it takes
            the 5% closest points.
        kernel (function, optional): kernel function. By default a uniform
            kernel to perform a 'usual' k nearest neighbours estimation.
        w (ndarray, optional): Case weights matrix.
        cv (bool, optional): Flag for cross-validation methods.
            Defaults to False.

    Returns:
        ndarray: Smoothing matrix.

    Examples:
        >>> knn(numpy.array([1,2,4,5,7]), 2)
        array([[ 0.5, 0.5, 0. , 0. , 0. ],
               [ 0.5, 0.5, 0. , 0. , 0. ],
               [ 0. , 0. , 0.5, 0.5, 0. ],
               [ 0. , 0. , 0.5, 0.5, 0. ],
               [ 0. , 0. , 0. , 0.5, 0.5]])

        In case there are two points at the same distance it will take both.

        >>> knn(numpy.array([1,2,3,5,7]), 2).round(3)
        array([[ 0.5  , 0.5  , 0.   , 0.   , 0.   ],
               [ 0.333, 0.333, 0.333, 0.   , 0.   ],
               [ 0.   , 0.5  , 0.5  , 0.   , 0.   ],
               [ 0.   , 0.   , 0.333, 0.333, 0.333],
               [ 0.   , 0.   , 0.   , 0.5  , 0.5  ]])


    """
    # Distances matrix of points in argvals
    delta_x = numpy.abs(numpy.subtract.outer(argvals, argvals))

    if k is None:
        k = numpy.floor(numpy.percentile(range(1, len(argvals)), 5))
    elif k <= 0:
        raise ValueError('h must be greater than 0')
    if cv:
        numpy.fill_diagonal(delta_x, math.inf)

    # Tolerance to avoid points landing outside the kernel window due to
    # computation error
    tol = 1 * 10 ** -19

    # For each row in the distances matrix, it calculates the furthest point
    # within the k nearest neighbours
    vec = numpy.percentile(delta_x, k / len(argvals) * 100, axis=0,
                           interpolation='lower') + tol

    rr = kernel((delta_x.T / vec).T)
    # Applies the kernel to the result of dividing each row by the result
    # of the previous operation, all the discretisation points corresponding
    # to the knn are below 1 and the rest above 1 so the kernel returns values
    # distinct to 0 only for the knn.

    if w is not None:
        rr = (rr.T * w).T

    # normalise every row
    rs = numpy.sum(rr, 1)
    return (rr.T / rs).T
