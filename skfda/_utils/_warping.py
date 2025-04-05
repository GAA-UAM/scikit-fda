"""Registration of functional data module.

This module contains routines related to the registration procedure.
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Optional, Protocol

import numpy as np
from scipy.interpolate import PchipInterpolator

from ..preprocessing.missing import MissingValuesInterpolation
from ..typing._base import DomainRangeLike
from ..typing._numpy import ArrayLike, NDArrayFloat

if TYPE_CHECKING:
    from ..representation import FDataGrid


class LineEnergyFunction(Protocol):
    """
    Computes the energies of line segments.

    Returns the matrix containing the partial energies of all line
    segments between a candidate point and the target.

    """

    def __call__(
        self,
        /,
        original: FDataGrid,
        target: FDataGrid,
        *,
        grid_points: NDArrayFloat,
        row: int,
        column: int,
    ) -> NDArrayFloat:
        """Returns energies of all lines from each candidate point."""


def _dp_recover_warpings(
    row_indexes: NDArrayFloat,
    column_indexes: NDArrayFloat,
    grid_points: NDArrayFloat,
) -> FDataGrid:
    """
    Recover the warpings from the dynamic programming algorithm.

    It goes backwards, from the last point, using the row and column indexes
    we stored for each point (containing the index of the best candidate
    point).

    This is iterative and cannot be vectorized, except over samples. Moreover,
    the number of line segments for each warping may be different. In order to
    solve that, we use masking for setting only the fixed values, leaving the
    others as NaN, and then we do linear interpolation over the NaN values.

    """
    n_samples = row_indexes.shape[0]
    n_points = grid_points.shape[0]

    # Recover the warpings
    times = np.zeros((n_samples, n_points))

    previous_row_idx = np.full((n_samples, 1), fill_value=n_points - 1)
    previous_column_idx = np.full((n_samples, 1), fill_value=n_points - 1)

    for time_idx in range(n_points - 1, 0, -1):

        # 1 if we have to set a warping value in this time index, 0 if not
        index_match = previous_row_idx == time_idx

        warping_values = grid_points[previous_column_idx]

        # We will left as NaN the times to be interpolated linearly.
        times[:, time_idx] = np.where(index_match, warping_values, np.nan)

        # Find new indexes.
        previous_row_idx = np.where(
            index_match,
            row_indexes[:, time_idx, previous_column_idx],
            previous_row_idx,
        )
        previous_column_idx = np.where(
            index_match,
            column_indexes[:, time_idx, previous_column_idx],
            previous_column_idx,
        )

    warpings = FDataGrid(
        data_matrix=times,
        grid_points=grid_points,
    )

    return MissingValuesInterpolation().transform(warpings)


def dynamic_programming_match(
    original: FDataGrid,
    target: FDataGrid,
    line_energy_function: LineEnergyFunction,
) -> FDataGrid:
    r"""
    Find an optimal warping to transform a set of curves into another.

    The following assumes that functions are curves in the [0, 1] interval.

    The optimal warping would be one that minimizes the :math:`L^2` distance
    between the warped function and the target function, called the cost
    function:

    .. math::
        \hat{\gamma} = \arg \min_{\gamma \in \Gamma}
        \int_0^1 (x_1(\gamma(t)) - x_2(t))^2 dt.

    Ideally the optimal warping for the whole function would also be the
    optimal warping to adjust any part of the function.
    Thus, we can define a partial cost function

    .. math::
        E(s, t, \gamma) = \int_s^t (x_1(\gamma(\tau)) - x_2(\tau))^2 d\tau.

    With that definition, the original cost function is
    :math:`E(0, 1, \gamma)`.

    We can then attempt to minimize the global cost function by discretizing
    the warping and minimize the partial cost function at each segment.

    As the warping has to be monotonic, the idea is to define a warping as a
    piecewise function in a :math:`t \times t` grid, with the constraint that
    the first line segment starts at (0, 0), the final one ends at (1, 1),
    and each line segment goes from the end of the previous one (i, j), to
    a point (i + Δi, j + Δj), with Δi and Δj non-negative.

    Then we can, using dynamic programming, compute the minimum partial cost
    at each point (i, j), considering the partial costs for each point
    (i', j') with i'< i and j' < j and the partial cost of a line from (i', j')
    to (i, j).

    The algorithm as described is quadratic in t.

    """
    n_samples = original.n_samples
    grid_points = original.grid_points[0]
    n_points = len(grid_points)

    row_indexes = np.zeros((n_samples, n_points, n_points), dtype=np.int64)
    column_indexes = np.zeros((n_samples, n_points, n_points), dtype=np.int64)
    energy = np.zeros((n_samples, n_points, n_points))

    # Discourage jumps to (1, 1) at the end
    energy[-1, :] = np.inf
    energy[:, -1] = np.inf
    energy[-1, -1] = 0

    for row in range(1, n_points):
        for column in range(1, n_points):
            candidate_points_partial_energy = energy[:, :row, :column]

            # TODO: this can be further vectorized and extracted
            # out of the loop
            candidate_points_line_energy = line_energy_function(
                original,
                target,
                grid_points=grid_points,
                row=row,
                column=column,
            )
            partial_energies = (
                candidate_points_partial_energy + candidate_points_line_energy
            )

            ravel_partial_energies = np.reshape(
                partial_energies,
                (n_samples, -1),
            )
            min_idx = np.argmin(ravel_partial_energies, axis=-1)
            rows_idx, columns_idx = np.unravel_index(
                min_idx,
                partial_energies.shape[1:],
            )
            row_indexes[:, row, column] = rows_idx
            column_indexes[:, row, column] = columns_idx
            energy[:, row, column] = ravel_partial_energies[:, min_idx]

    # Recover the warpings
    times = np.zeros((n_samples, n_points))

    previous_row_idx = np.full((n_samples, 1), fill_value=n_points - 1)
    previous_column_idx = np.full((n_samples, 1), fill_value=n_points - 1)

    for time_idx in range(n_points - 1, 0, -1):

        # 1 if we have to set a warping value in this time index, 0 if not
        index_match = previous_row_idx == time_idx

        warping_values = grid_points[previous_column_idx]

        # We will left as NaN the times to be interpolated linearly.
        times[:, time_idx] = np.where(index_match, warping_values, np.nan)

        # Find new indexes.
        previous_row_idx = np.where(
            index_match,
            row_indexes[time_idx, previous_column_idx],
            previous_row_idx,
        )
        previous_column_idx = np.where(
            index_match,
            column_indexes[time_idx, previous_column_idx],
            previous_column_idx,
        )

    warpings = FDataGrid(
        data_matrix=times,
        grid_points=grid_points,
    )

    return MissingValuesInterpolation().transform(warpings)


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
