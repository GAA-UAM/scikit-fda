"""Registration of functional data module.

This module contains routines related to the registration procedure.
"""
import collections

import scipy.integrate
from scipy.interpolate import PchipInterpolator

import numpy as np

from ..._utils import check_is_univariate


__author__ = "Pablo Marcos Manchón"
__email__ = "pablo.marcosm@estudiante.uam.es"


def invert_warping(fdatagrid, *, output_points=None):
    r"""Compute the inverse of a diffeomorphism.

    Let :math:`\gamma : [a,b] \rightarrow [a,b]` be a function strictly
    increasing, calculates the corresponding inverse
    :math:`\gamma^{-1} : [a,b] \rightarrow [a,b]` such that
    :math:`\gamma^{-1} \circ \gamma = \gamma \circ \gamma^{-1} = \gamma_{id}`.

    Uses a PCHIP interpolator to compute approximately the inverse.

    Args:
        fdatagrid (:class:`FDataGrid`): Functions to be inverted.
        eval_points: (array_like, optional): Set of points where the
            functions are interpolated to obtain the inverse, by default uses
            the sample points of the fdatagrid.

    Returns:
        :class:`FDataGrid`: Inverse of the original functions.

    Raises:
        ValueError: If the functions are not strictly increasing or are
            multidimensional.

    Examples:

        >>> import numpy as np
        >>> from skfda import FDataGrid
        >>> from skfda.preprocessing.registration import invert_warping

        We will construct the warping :math:`\gamma : [0,1] \rightarrow [0,1]`
        wich maps t to t^3.

        >>> t = np.linspace(0, 1)
        >>> gamma = FDataGrid(t**3, t)
        >>> gamma
        FDataGrid(...)

        We will compute the inverse.

        >>> inverse = invert_warping(gamma)
        >>> inverse
        FDataGrid(...)

        The result of the composition should be approximately the identity
        function .

        >>> identity = gamma.compose(inverse)
        >>> identity([0, 0.25, 0.5, 0.75, 1]).round(3)
        array([[[ 0.  ],
                [ 0.25],
                [ 0.5 ],
                [ 0.75],
                [ 1.  ]]])

    """

    check_is_univariate(fdatagrid)

    if output_points is None:
        output_points = fdatagrid.grid_points[0]

    y = fdatagrid(output_points)[..., 0]

    data_matrix = np.empty((fdatagrid.n_samples, len(output_points)))

    for i in range(fdatagrid.n_samples):
        data_matrix[i] = PchipInterpolator(y[i], output_points)(output_points)

    return fdatagrid.copy(data_matrix=data_matrix, grid_points=output_points)


def _normalize_scale(t, a=0, b=1):
    """Perfoms an afine translation to normalize an interval.

    Args:
        t (numpy.ndarray): Array of dim 1 or 2 with at least 2 values.
        a (float): Starting point of the new interval. Defaults 0.
        b (float): Stopping point of the new interval. Defaults 1.

    Returns:
        (numpy.ndarray): Array with the transformed interval.
    """

    t = t.T  # Broadcast to normalize multiple arrays
    t1 = (t - t[0]).astype(float)  # Translation to [0, t[-1] - t[0]]
    t1 *= (b - a) / (t[-1] - t[0])  # Scale to [0, b-a]
    t1 += a  # Translation to [a, b]
    t1[0] = a  # Fix possible round errors
    t1[-1] = b

    return t1.T


def normalize_warping(warping, domain_range=None):
    r"""Rescale a warping to normalize their :term:`domain`.

    Given a set of warpings :math:`\gamma_i:[a,b]\rightarrow  [a,b]` it is
    used an affine traslation to change the domain of the transformation to
    other domain, :math:`\tilde \gamma_i:[\tilde a,\tilde b] \rightarrow
    [\tilde a, \tilde b]`.

    Args:
        warping (:class:`FDatagrid`): Set of warpings to rescale.
        domain_range (tuple, optional): New domain range of the warping. By
            default it is used the same domain range.
    Return:
        (:class:`FDataGrid`): FDataGrid with the warpings normalized.

    """

    if domain_range is None:
        domain_range = warping.domain_range[0]

    data_matrix = _normalize_scale(warping.data_matrix[..., 0], *domain_range)
    grid_points = _normalize_scale(warping.grid_points[0], *domain_range)

    return warping.copy(data_matrix=data_matrix, grid_points=grid_points,
                        domain_range=domain_range)
