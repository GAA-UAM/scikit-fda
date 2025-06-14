"""Registration of functional data module.

This module contains routines related to the registration procedure.
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Protocol

import numpy as np
from scipy.interpolate import PchipInterpolator
from typing_extensions import override

if TYPE_CHECKING:
    from ..representation import FDataGrid
    from ..typing._base import DomainRangeLike
    from ..typing._numpy import ArrayLike, NDArrayFloat, NDArrayInt


class LineEnergyFunction(Protocol):
    """
    Computes the energies of line segments.

    Returns the matrix containing the partial energies of all line
    segments between a candidate point and the target.

    """
    def dp_start(
        self,
        /,
        original: FDataGrid,
        target: FDataGrid,
        *,
        grid_dim: int,
    ) -> None:
        """
        Called at the start of the DP algorithm.

        This can be used to cache the needed values that are the
        same for each row and column.

        """

    def dp_change_row(
        self,
        /,
        original: FDataGrid,
        target: FDataGrid,
        *,
        row: int,
        grid_dim: int,
    ) -> None:
        """
        Called when a row is changed in the DP algorithm.

        This can be used to cache the needed values that are the
        same for each row.

        Note:
            There is no analog function for columns, as the column
            always change (and it requires less computations, because
            the integral domain does not change).

        """

    def __call__(
        self,
        /,
        original: FDataGrid,
        target: FDataGrid,
        *,
        row: int,
        column: int,
        grid_dim: int,
    ) -> NDArrayFloat:
        """Returns energies of all lines from each candidate point."""


def _dp_recover_warpings(
    row_indexes: NDArrayInt,
    column_indexes: NDArrayInt,
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

    Args:
        row_indexes: Array containing for each position the index of the
            previous row in the path.
        column_indexes: Array containing for each position the index of the
            previous column in the path.
        grid_points: Grid points of the functions.

    Return:
        Reconstructed paths.

    Examples:
        Consider a simple case with 4 discretization points:

        >>> import numpy as np
        >>> grid_points = np.array([0, 0.33, 0.66, 1])

        We have two samples, and the following indexes matrices (we set to -1
        the  unused entries for clarity):

        >>> row_indexes = np.array(
        ...     [
        ...         [
        ...             [-1, -1, -1, -1],
        ...             [-1, -1,  0, -1],
        ...             [-1, -1, -1, -1],
        ...             [-1, -1, -1,  1],
        ...         ],
        ...         [
        ...             [-1, -1, -1, -1],
        ...             [ 0, -1, -1, -1],
        ...             [-1, -1,  1, -1],
        ...             [-1, -1, -1,  2],
        ...         ],
        ...     ]
        ... )
        >>> column_indexes = np.array(
        ...     [
        ...         [
        ...             [-1, -1, -1, -1],
        ...             [-1, -1,  0, -1],
        ...             [-1, -1, -1, -1],
        ...             [-1, -1, -1,  2],
        ...         ],
        ...         [
        ...             [-1, -1, -1, -1],
        ...             [ 0, -1, -1, -1],
        ...             [-1, -1,  0, -1],
        ...             [-1, -1, -1,  2],
        ...         ],
        ...     ]
        ... )

        >>> warpings = _dp_recover_warpings(
        ...     row_indexes=row_indexes,
        ...     column_indexes=column_indexes,
        ...     grid_points=grid_points,
        ... )
        >>> warpings.data_matrix
        array([[[ 0.        ],
                [ 0.66      ],
                [ 0.82746269],
                [ 1.        ]],
               [[ 0.        ],
                [ 0.        ],
                [ 0.66      ],
                [ 1.        ]]])

    """
    from ..preprocessing.missing import MissingValuesInterpolation
    from ..representation import FDataGrid

    n_samples = row_indexes.shape[0]
    n_points = grid_points.shape[0]

    # Recover the warpings
    times = np.zeros((n_samples, n_points))

    previous_row_idx: NDArrayInt = np.full(
        (n_samples,),
        fill_value=n_points - 1,
    )
    previous_column_idx: NDArrayInt = np.full(
        (n_samples,),
        fill_value=n_points - 1,
    )

    arange_idx = np.arange(n_samples)

    for time_idx in reversed(range(n_points)):

        # 1 if we have to set a warping value in this time index, 0 if not
        index_match = previous_row_idx == time_idx

        warping_values = grid_points[previous_column_idx]

        # We will left as NaN the times to be interpolated linearly.
        times[:, time_idx] = np.where(index_match, warping_values, np.nan)

        # Find new indexes.
        previous_row_idx = np.where(
            index_match,
            row_indexes[arange_idx, time_idx, previous_column_idx],
            previous_row_idx,
        )
        previous_column_idx = np.where(
            index_match,
            column_indexes[arange_idx, time_idx, previous_column_idx],
            previous_column_idx,
        )

    warpings = FDataGrid(
        data_matrix=times,
        grid_points=grid_points,
    )

    na_interpolator = MissingValuesInterpolation[FDataGrid]()
    return na_interpolator.transform(warpings)


class L2LineEnergy(LineEnergyFunction):
    r"""
    L2 line energy with roughness penalty.

    The energy computed is:

    .. math::
            d(x, y) = \int_{t_i}^{t_{row}} (x(w(t)) - y(t))^2 dt
            + \lambda (1 - \sqrt{w'(t)})^2

    where :math:`\lambda` is the penalty factor.

    Args:
        penalty: The penalization factor. The default, 0, is no penalization.
        slope_scaling: Wether to scale the original data by the square root
            of the slope of the interval. This is necessary when we work with
            the SRSF of the curves instead of with the curves themselves.

    Examples:
        Consider a simple case with 4 irregularly sampled discretization
        points:

        >>> import numpy as np
        >>> grid_points = np.array([0, 0.5, 0.75, 1])

        We use the identity function :math:`x(t) = t` and the piecewise
        linear function :math:`y(t) = \max(0, 2t - 1)`:

        >>> from skfda import FDataGrid
        >>> original = FDataGrid(grid_points, grid_points=grid_points)
        >>> target = FDataGrid(
        ...     np.maximum(0, 2 * grid_points - 1),
        ...     grid_points=grid_points,
        ... )

        Consider the default case with no penalization:
        >>> l2_line_energy = L2LineEnergy()

        We will consider use the full grid for now:

        >>> grid_dim = len(grid_points)

        Before calling it, the DP algorithm will call the ``dp_start``
        method, at the beginning of the algorithm, to give the opportunity to
        cache variables that are common for all the candidate points.

        >>> l2_line_energy.dp_start(
        ...    original,
        ...    target,
        ...    grid_dim=grid_dim,
        ... )

        We will consider the final case ``row = 3``, ``column=3``. Before
        calling the function, the DP algorithm will call ``dp_change_row``
        whenever the row is changed, to give the opportunity to
        cache variables that are common for all the candidate points in
        the same row.

        >>> l2_line_energy.dp_change_row(
        ...    original,
        ...    target,
        ...    grid_dim=grid_dim,
        ...    row=3,
        ... )

        >>> l2_line_energy(
        ...    original,
        ...    target,
        ...    grid_dim=grid_dim,
        ...    row=3,
        ...    column=3,
        ... )
        array([[[ 0.109375  ,  0.30859375,  0.47558594],
                [ 0.        ,  0.046875  ,  0.10546875],
                [ 0.03125   ,  0.        ,  0.0078125 ]]])

        Note that the cells ``(1, 0)`` and ``(2, 1)`` have 0 energy.
        This is because they correspond to linear warpings that align
        perfectly :math:`x` in the intervals :math:`(0.5, 1)` and
        :math:`(0.75, 1)`, respectively.

        Now consider a possible intermediate case with ``row = 2`` and
        ``column=1``:
        >>> l2_line_energy.dp_change_row(
        ...    original,
        ...    target,
        ...    grid_dim=grid_dim,
        ...    row=2,
        ... )
        >>> l2_line_energy(
        ...    original,
        ...    target,
        ...    grid_dim=grid_dim,
        ...    row=2,
        ...    column=1,
        ... )
        array([[[ 0.04166667],
                [ 0.        ]]])

        This has again 0 energy at ``(1, 0)``, because it is possible to
        align perfectly :math:`x` in the interval :math:`(0.5, 0.75)`.

        With the parameter ``grid_dim`` we can control the size of the
        grid used, so that the algorithm is still tractable with many
        points:

        >>> grid_dim = 1

        >>> l2_line_energy.dp_start(
        ...    original,
        ...    target,
        ...    grid_dim=grid_dim,
        ... )

        >>> l2_line_energy.dp_change_row(
        ...    original,
        ...    target,
        ...    grid_dim=grid_dim,
        ...    row=3,
        ... )

        >>> grid_1 = l2_line_energy(
        ...    original,
        ...    target,
        ...    grid_dim=grid_dim,
        ...    row=3,
        ...    column=3,
        ... )
        >>> grid_1
        array([[[ 0.0078125]]])

        >>> grid_dim = 2

        >>> l2_line_energy.dp_start(
        ...    original,
        ...    target,
        ...    grid_dim=grid_dim,
        ... )

        >>> l2_line_energy.dp_change_row(
        ...    original,
        ...    target,
        ...    grid_dim=grid_dim,
        ...    row=3,
        ... )

        >>> grid_2 = l2_line_energy(
        ...    original,
        ...    target,
        ...    grid_dim=grid_dim,
        ...    row=3,
        ...    column=3,
        ... )
        >>> grid_2
        array([[[ 0.046875  ,  0.10546875],
                [ 0.        ,  0.0078125 ]]])

        As it can be seen, the results correspond to the lower-right part
        of the complete grid.

        It is also possible to penalize deviations from the identity
        function:

        >>> l2_line_energy = L2LineEnergy(penalty=1)

        >>> grid_dim = len(grid_points)

        >>> l2_line_energy.dp_start(
        ...    original,
        ...    target,
        ...    grid_dim=grid_dim,
        ... )

        >>> l2_line_energy.dp_change_row(
        ...    original,
        ...    target,
        ...    grid_dim=grid_dim,
        ...    row=3,
        ... )

        >>> l2_line_energy(
        ...    original,
        ...    target,
        ...    grid_dim=grid_dim,
        ...    row=3,
        ...    column=3,
        ... )
        array([[[ 0.109375  ,  0.39438019,  0.72558594],
                [ 0.08578644,  0.046875  ,  0.14836197],
                [ 0.28125   ,  0.04289322,  0.0078125 ]]])

        Note that the terms on the main diagonal are not penalized in this
        case, as there is no difference in slope with respect to the
        identity function.

    """

    def __init__(
        self,
        *,
        penalty: float = 0,
        slope_scaling: bool = False,
    ) -> None:
        self.penalty = penalty
        self.slope_scaling = slope_scaling

    @override
    def dp_start(
        self,
        /,
        original: FDataGrid,
        target: FDataGrid,
        *,
        grid_dim: int,
    ) -> None:
        self.grid_points = original.grid_points[0]
        self.interval_lenghts = np.diff(
            self.grid_points,
            prepend=self.grid_points[0],
        )

        data_matrix = original.data_matrix[..., 0]

        self.data_diff = np.diff(
            data_matrix,
            prepend=data_matrix[:, :1],
        )

        padding_col = np.zeros((data_matrix.shape[0], 1))

        self.left_grid_points = np.concat(
            (self.grid_points[:1], self.grid_points),
        )

        diff = np.concat(
            (self.data_diff, padding_col),
            axis=1,
        )

        # First lenght is 0, so skip it
        # It will only be used multiplied by 0 in that case, anyway
        diff[:, 1:-1] /= self.interval_lenghts[1:]

        self.left_and_diff = np.stack((
            np.concat(
                (data_matrix[:, :1], data_matrix),
                axis=1,
            ),
            diff,
        ))

    @override
    def dp_change_row(
        self,
        /,
        original: FDataGrid,
        target: FDataGrid,
        *,
        row: int,
        grid_dim: int,
    ) -> None:
        first_row_index = max(0, row - grid_dim)
        t_row = self.grid_points[row]
        t_i = self.grid_points[first_row_index:row, None]
        self.total_interval_length = t_row - t_i

        t = self.grid_points[first_row_index:row + 1]
        self.l_matrix = (
            (t - t_i) / self.total_interval_length
        )[:, None, :]

        self.y_t = target.data_matrix[
            :,
            None,
            None,
            first_row_index:row + 1,
            0,
        ]

        row_interval_lenghts = self.interval_lenghts[
            first_row_index:row + 1
        ]
        quadrature_weights = row_interval_lenghts / 2
        quadrature_weights[:-1] += row_interval_lenghts[1:] / 2

        # We set the weights of unused points to 0
        quadrature_weights = np.triu(
            np.tile(quadrature_weights, (len(t_i), 1)),
        )

        # Final correction: adjust the weight at the extreme
        # We need to remove t_i - t_{i-1} only at the leftmost point
        quadrature_weights_diag = np.diagonal(quadrature_weights)
        np.fill_diagonal(
            quadrature_weights,
            quadrature_weights_diag - row_interval_lenghts[:-1] / 2,
        )

        # Add column dimension
        quadrature_weights = quadrature_weights[:, None, :]
        self.quadrature_weights = quadrature_weights

    def evaluate_fdatagrid_linear_interpolation(
        self,
        evaluation_points: NDArrayFloat,
    ) -> NDArrayFloat:
        """Evaluate the functions at some points using linear interpolation."""
        indexes = np.searchsorted(
            self.grid_points,
            evaluation_points,
            side="left",
        )
        # Extrapolation is not necessary, as the warping values are inside the
        # domain

        grid_points_prev = self.left_grid_points[indexes]

        interp_parameter = evaluation_points - grid_points_prev

        left, data_diff = np.take(self.left_and_diff, indices=indexes, axis=2)

        left += interp_parameter * data_diff
        return left  # type: ignore[no-any-return]

    @override
    def __call__(  # noqa: WPS210
        self,
        original: FDataGrid,
        target: FDataGrid,
        *,
        row: int,
        column: int,
        grid_dim: int,
    ) -> NDArrayFloat:
        r"""
        Compute the line energies for the :math:`L^2` distance.

        This computes, for each row ``i`` and column ``j``, with ``i < row``
        and ``j < column`` the following distance:

        .. math::
            d(x, y) = \int_{t_i}^{t_{row}} (x(w(t)) - y(t))^2 dt

        where :math:`w`, the warping, is a straight line, that is,
        :math:`w(t) = (1 - l) t_j + l t_column` with
        :math:`l = (t - t_i) / (t_{row} - t_i)`.

        Args:
            original: Functions to be aligned.
            target: Target function(s) to align to.
            row: The row index of the candidate point.
            column: The column index of the candidate point.
            grid_dim: Dimension of the grid used in the alignment
                algorithm. Only the direct lines from points whose grid
                separation with the candidate point is less or equal than
                ``grid_dim`` are considered.

        Returns:
            Energy of direct line warpings to the candidate point.

        """
        grid_points = self.grid_points

        first_column_index = max(0, column - grid_dim)

        t_column = grid_points[column]

        t_j = grid_points[first_column_index:column, None]

        total_interval_length = self.total_interval_length

        l_matrix = self.l_matrix
        w = t_j + l_matrix * (t_column - t_j)

        # Shape: N x row x column x t
        x_t =self.evaluate_fdatagrid_linear_interpolation(w)

        w_slope = (t_column - t_j.T) / total_interval_length
        w_slope_root = np.sqrt(w_slope)

        if self.slope_scaling:
            x_t *= w_slope_root[None, ..., None]

        integrand = (x_t - self.y_t)**2

        quadrature_weights = self.quadrature_weights

        integrand *= quadrature_weights
        integral = np.sum(integrand, axis=-1)

        roughness = self.penalty * (
            (1 - w_slope_root)**2 * total_interval_length
        )

        integral += roughness
        return integral  # type: ignore[no-any-return]

def dynamic_programming_match(  # noqa: WPS210
    original: FDataGrid,
    target: FDataGrid,
    *,
    line_energy_function: LineEnergyFunction,
    grid_dim: int | None = None,
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

    Args:
        original: Functions to be aligned.
        target: Target function(s) to align to.
        line_energy_function: Function used to compute the energy for a line
            segment of the warpings.
        grid_dim: Dimension of the grid used in the alignment algorithm. Only
            the direct lines from points whose grid separation with the
            candidate point is less or equal than ``grid_dim`` are considered.

    Returns:
        The warpings that align the functions using the DP algorithm.

    Examples:
        Consider a simple case with 100 equally spaced discretization
        points:

        >>> import numpy as np
        >>> grid_points = np.linspace(0, 1, 100)

        We want to align the functions :math:`x_1(t) = t**2` and the
        function :math:`x_2(t) = \sin(\frac{\pi}{2} t)`
        to the identity function :math:`y(t) = t`.
        All these are functions that start and end at the same points ((0, 0)
        and (1, 1), respectively), and they are monotonic. Thus, the exact,
        expected solution for the registration problem with the identity is
        that the warpings are their inverses:

        >>> from skfda import FDataGrid
        >>> original = FDataGrid(
        ...     [
        ...         grid_points**2,
        ...         np.sin(np.pi / 2 * grid_points),
        ...     ],
        ...     grid_points=grid_points,
        ... )
        >>> target = FDataGrid(
        ...     grid_points,
        ...     grid_points=grid_points,
        ... )

        We limit the grid size, for performance reasons:
        >>> l2_line_energy = L2LineEnergy()
        >>> warpings = dynamic_programming_match(
        ...     original,
        ...     target,
        ...     line_energy_function=l2_line_energy,
        ...     grid_dim=7,
        ... )

        We check now that the found warpings are close to their inverses:
        >>> np.allclose(
        ...     warpings.data_matrix[0, ..., 0],
        ...     np.sqrt(grid_points),
        ...     atol=0.05,
        ... )
        True
        >>> np.allclose(
        ...     warpings.data_matrix[1, ..., 0],
        ...     2 / np.pi * np.arcsin(grid_points),
        ...     atol=0.05,
        ... )
        True

        Note that the allowed slopes are restricted by the grid, and thus
        a small discrepancy is expected.

    """
    n_samples = original.n_samples
    grid_points = original.grid_points[0]
    n_points = len(grid_points)
    if grid_dim is None:
        grid_dim = n_points

    arange_idx = np.arange(n_samples)

    row_indexes = np.zeros((n_samples, n_points, n_points), dtype=np.int64)
    column_indexes = np.zeros((n_samples, n_points, n_points), dtype=np.int64)
    energy = np.full((n_samples, n_points, n_points), fill_value=np.inf)

    # Discourage jumps from (0, 0) at the beginning
    energy[:, 0, :] = np.inf
    energy[:, :, 0] = np.inf
    energy[:, 0, 0] = 0

    line_energy_function.dp_start(
        original,
        target,
        grid_dim=grid_dim,
    )

    for row in range(1, n_points):

        line_energy_function.dp_change_row(
            original,
            target,
            grid_dim=grid_dim,
            row=row,
        )

        for column in range(1, n_points):
            first_row_index = max(0, row - grid_dim)
            first_column_index = max(0, column - grid_dim)

            candidate_points_partial_energy = energy[
                :,
                first_row_index:row,
                first_column_index:column,
            ]

            # This can be further vectorized and extracted
            # out of the loop, but not sure if it is worth it.
            # In particular, the memory usage could be much higher for
            # almost no performance gains.
            candidate_points_line_energy = line_energy_function(
                original,
                target,
                row=row,
                column=column,
                grid_dim=grid_dim,
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
            rows_idx += first_row_index
            columns_idx += first_column_index
            row_indexes[:, row, column] = rows_idx
            column_indexes[:, row, column] = columns_idx
            energy[:, row, column] = ravel_partial_energies[
                arange_idx,
                min_idx,
            ]

    return _dp_recover_warpings(
        row_indexes=row_indexes,
        column_indexes=column_indexes,
        grid_points=grid_points,
    )


def invert_warping(
    warping: FDataGrid,
    *,
    output_points: ArrayLike | None = None,
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
    domain_range: DomainRangeLike | None = None,
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
