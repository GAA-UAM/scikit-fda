# -*- coding: utf-8 -*-

"""Kernel Smoother.

This module contains the class for kernel smoothing.

"""
from typing import Optional

import numpy as np

from ..._utils._utils import _to_grid_points
from ...misc.hat_matrix import HatMatrix, NadarayaWatsonHatMatrix
from ...typing._base import GridPointsLike
from ...typing._numpy import NDArrayFloat
from ._linear import _LinearSmoother


class KernelSmoother(_LinearSmoother):
    r"""Kernel smoothing method.

    This module allows to perform functional data smoothing.

    Let :math:`t = (t_1, t_2, ..., t_n)` be the
    points of discretisation and :math:`X` the vector of observations at that
    points. Then, the smoothed values, :math:`\hat{X}`, at the points
    :math:`t' = (t_1', t_2', ..., t_m')` are obtained as

    .. math::
        \hat{X} = \hat{H} X

    where :math:`\hat{H}` is a matrix described in
    :class:`~skfda.misc.hat_matrix.HatMatrix`.

    Examples:
        >>> from skfda import FDataGrid
        >>> from skfda.misc.hat_matrix import NadarayaWatsonHatMatrix
        >>> fd = FDataGrid(
        ...     grid_points=[1, 2, 4, 5, 7],
        ...     data_matrix=[[1, 2, 3, 4, 5]],
        ... )
        >>> kernel_estimator = NadarayaWatsonHatMatrix(bandwidth=3.5)
        >>> smoother = KernelSmoother(kernel_estimator=kernel_estimator)
        >>> fd_smoothed = smoother.fit_transform(fd)
        >>> fd_smoothed.data_matrix.round(2)
        array([[[ 2.42],
                [ 2.61],
                [ 3.03],
                [ 3.24],
                [ 3.65]]])
        >>> smoother.hat_matrix().round(3)
        array([[ 0.294, 0.282, 0.204, 0.153, 0.068],
               [ 0.249, 0.259, 0.22 , 0.179, 0.093],
               [ 0.165, 0.202, 0.238, 0.229, 0.165],
               [ 0.129, 0.172, 0.239, 0.249, 0.211],
               [ 0.073, 0.115, 0.221, 0.271, 0.319]])
        >>> kernel_estimator = NadarayaWatsonHatMatrix(bandwidth=2)
        >>> smoother = KernelSmoother(kernel_estimator=kernel_estimator)
        >>> fd_smoothed = smoother.fit_transform(fd)
        >>> fd_smoothed.data_matrix.round(2)
        array([[[ 1.84],
                [ 2.18],
                [ 3.09],
                [ 3.55],
                [ 4.28]]])
        >>> smoother.hat_matrix().round(3)
        array([[ 0.425, 0.375, 0.138, 0.058, 0.005],
               [ 0.309, 0.35 , 0.212, 0.114, 0.015],
               [ 0.103, 0.193, 0.319, 0.281, 0.103],
               [ 0.046, 0.11 , 0.299, 0.339, 0.206],
               [ 0.006, 0.022, 0.163, 0.305, 0.503]])

        The output points can be changed:

        >>> kernel_estimator = NadarayaWatsonHatMatrix(bandwidth=2)
        >>> smoother = KernelSmoother(
        ...     kernel_estimator=kernel_estimator,
        ...     output_points=[1, 2, 3, 4, 5, 6, 7],
        ... )
        >>> fd_smoothed = smoother.fit_transform(fd)
        >>> fd_smoothed.data_matrix.round(2)
        array([[[ 1.84],
                [ 2.18],
                [ 2.61],
                [ 3.09],
                [ 3.55],
                [ 3.95],
                [ 4.28]]])
        >>> smoother.hat_matrix().round(3)
        array([[ 0.425,  0.375,  0.138,  0.058,  0.005],
               [ 0.309,  0.35 ,  0.212,  0.114,  0.015],
               [ 0.195,  0.283,  0.283,  0.195,  0.043],
               [ 0.103,  0.193,  0.319,  0.281,  0.103],
               [ 0.046,  0.11 ,  0.299,  0.339,  0.206],
               [ 0.017,  0.053,  0.238,  0.346,  0.346],
               [ 0.006,  0.022,  0.163,  0.305,  0.503]])

    Args:
        kernel_estimator: Method used to
            calculate the hat matrix (default =
            :class:`~skfda.misc.NadarayaWatsonHatMatrix`)
        weights: weight coefficients for each point.
        output_points: The output points. If omitted, the
            input points are used.

    So far only non parametric methods are implemented because we are only
    relying on a discrete representation of functional data.

    """

    def __init__(
        self,
        kernel_estimator: Optional[HatMatrix] = None,
        *,
        weights: Optional[NDArrayFloat] = None,
        output_points: Optional[GridPointsLike] = None,
    ):
        self.kernel_estimator = kernel_estimator
        self.weights = weights
        self.output_points = output_points
        self._cv = False  # For testing purposes only

    def _hat_matrix(
        self,
        input_points: GridPointsLike,
        output_points: GridPointsLike,
    ) -> NDArrayFloat:

        input_points = _to_grid_points(input_points)
        output_points = _to_grid_points(output_points)

        if self.kernel_estimator is None:
            self.kernel_estimator = NadarayaWatsonHatMatrix()

        delta_x = np.subtract.outer(output_points[0], input_points[0])

        return self.kernel_estimator(
            delta_x=delta_x,
            weights=self.weights,
            X_train=input_points,
            X=output_points,
            _cv=self._cv,
        )
