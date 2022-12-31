"""Registration of functional data module.

This module contains routines related to the registration procedure.
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Optional

import numpy as np
from scipy.interpolate import PchipInterpolator

from ..typing._base import DomainRangeLike
from ..typing._numpy import ArrayLike, NDArrayFloat

if TYPE_CHECKING:
    from ..representation import FDataGrid


def invert_warping(
    warping: FDataGrid,
    *,
    output_points: Optional[ArrayLike] = None,
) -> FDataGrid:
    r"""
    Compute the inverse of a diffeomorphism.

    Let :math:`\gamma : [a,b] \rightarrow [a,b]` be a function strictly
    increasing, calculates the corresponding inverse
    :math:`\gamma^{-1} : [a,b] \rightarrow [a,b]` such that
    :math:`\gamma^{-1} \circ \gamma = \gamma \circ \gamma^{-1} = \gamma_{id}`.

    Uses a PCHIP interpolator to compute approximately the inverse.

    Args:
        warping: Functions to be inverted.
        output_points: Set of points where the
            functions are interpolated to obtain the inverse, by default uses
            the sample points of the fdatagrid.

    Returns:
        Inverse of the original functions.

    Raises:
        ValueError: If the functions are not strictly increasing or are
            multidimensional.

    Examples:
        >>> import numpy as np
        >>> from skfda import FDataGrid

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
    from ..misc.validation import check_fdata_dimensions

    check_fdata_dimensions(
        warping,
        dim_domain=1,
        dim_codomain=1,
    )

    output_points = (
        warping.grid_points[0]
        if output_points is None
        else np.asarray(output_points)
    )

    y = warping(output_points)[..., 0]

    data_matrix = np.empty((warping.n_samples, len(output_points)))

    for i in range(warping.n_samples):
        data_matrix[i] = PchipInterpolator(y[i], output_points)(output_points)

    return warping.copy(data_matrix=data_matrix, grid_points=output_points)


def normalize_scale(
    t: NDArrayFloat,
    a: float = 0,
    b: float = 1,
) -> NDArrayFloat:
    """
    Perfoms an afine translation to normalize an interval.

    Args:
        t: Array of dim 1 or 2 with at least 2 values.
        a: Starting point of the new interval. Defaults 0.
        b: Stopping point of the new interval. Defaults 1.

    Returns:
        Array with the transformed interval.

    """
    t = t.T  # Broadcast to normalize multiple arrays
    t1 = np.array(t, copy=True)
    t1 -= t[0]  # Translation to [0, t[-1] - t[0]]
    t1 *= (b - a) / (t[-1] - t[0])  # Scale to [0, b-a]
    t1 += a  # Translation to [a, b]
    t1[0] = a  # Fix possible round errors
    t1[-1] = b

    return t1.T


def normalize_warping(
    warping: FDataGrid,
    domain_range: Optional[DomainRangeLike] = None,
) -> FDataGrid:
    r"""
    Rescale a warping to normalize their :term:`domain`.

    Given a set of warpings :math:`\gamma_i:[a,b]\rightarrow  [a,b]` it is
    used an affine traslation to change the domain of the transformation to
    other domain, :math:`\tilde \gamma_i:[\tilde a,\tilde b] \rightarrow
    [\tilde a, \tilde b]`.

    Args:
        warping: Set of warpings to rescale.
        domain_range: New domain range of the warping. By
            default it is used the same domain range.

    Returns:
        Normalized warpings.

    """
    from ..misc.validation import validate_domain_range

    domain_range_tuple = (
        warping.domain_range[0]
        if domain_range is None
        else validate_domain_range(domain_range)[0]
    )

    data_matrix = normalize_scale(
        warping.data_matrix[..., 0],
        *domain_range_tuple,
    )
    grid_points = normalize_scale(warping.grid_points[0], *domain_range_tuple)

    return warping.copy(
        data_matrix=data_matrix,
        grid_points=grid_points,
        domain_range=domain_range,
    )
