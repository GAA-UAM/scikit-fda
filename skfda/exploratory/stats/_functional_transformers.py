"""Functional Transformers Module."""

from __future__ import annotations

from typing import Tuple, Union

import numpy as np

from ...representation import FDataBasis, FDataGrid


def _calculate_curves_occupation_(
    curve_y_coordinates: np.ndarray,
    curve_x_coordinates: np.ndarray,
    interval: Tuple,
) -> np.ndarray:
    y1, y2 = interval

    # Reshape original curves so they have one dimension less
    new_shape = curve_y_coordinates.shape[1::-1]
    curve_y_coordinates = curve_y_coordinates.reshape(
        new_shape[::-1],
    )

    # Calculate interval sizes on the X axis
    x_rotated = np.roll(curve_x_coordinates, 1)
    intervals_x_axis = curve_x_coordinates - x_rotated

    # Calculate which points are inside the interval given (y1,y2) on Y axis
    greater_than_y1 = curve_y_coordinates >= y1
    less_than_y2 = curve_y_coordinates <= y2
    inside_interval_bools = greater_than_y1 & less_than_y2

    # Correct booleans so they are not repeated
    bools_interval = np.roll(
        inside_interval_bools, 1, axis=1,
    ) & inside_interval_bools

    # Calculate intervals on X axis where the points are inside Y axis interval
    intervals_x_inside = np.multiply(bools_interval, intervals_x_axis)

    # Delete first element of each interval as it will be a negative number
    intervals_x_inside = np.delete(intervals_x_inside, 0, axis=1)

    return np.sum(intervals_x_inside, axis=1)


def occupation_measure(
    data: Union[FDataGrid, FDataBasis],
    intervals: np.ndarray,
    *,
    n_points: Union[int, None] = None,
) -> np.ndarray:
    r"""
    Calculate the occupation measure of a grid.

    It performs the following map.
        ..math:
            :math:`f_1(X) = |t: X(t)\in T_p|,\dots,|t: X(t)\in T_p|`

        where :math:`{T_1,\dots,T_p}` are disjoint intervals in
        :math:`\mathbb{R}` and | | stands for the Lebesgue measure.

        Args:
            data: FDataGrid or FDataBasis where we want to calculate
            the occupation measure.
            intervals: ndarray of tuples containing the
            intervals we want to consider. The shape should be
            (n_sequences,2)
            n_points: Number of points to evaluate in the domain.
            By default will be used the points defined on the FDataGrid.
            On a FDataBasis this value should be specified.
        Returns:
            ndarray of shape (n_intervals, n_samples)
            with the transformed data.

    Example:
        We will create the FDataGrid that we will use to extract
        the occupation measure
        >>> from skfda.representation import FDataGrid
        >>> import numpy as np
        >>> t = np.linspace(0, 10, 100)
        >>> fd_grid = FDataGrid(
        ...     data_matrix=[
        ...         t,
        ...         2 * t,
        ...         np.sin(t),
        ...     ],
        ...     grid_points=t,
        ... )

        Finally we call to the occupation measure function with the
        intervals that we want to consider. In our case (0.0, 1.0)
        and (2.0, 3.0). We need also to specify the number of points
        we want that the function takes into account to interpolate.
        We are going to use 501 points.
        >>> from skfda.exploratory.stats import occupation_measure
        >>> np.around(
        ...     occupation_measure(
        ...         fd_grid,
        ...         [(0.0, 1.0), (2.0, 3.0)],
        ...         n_points=501,
        ...     ),
        ...     decimals=2,
        ... )
        array([[ 1.  ,  0.5 ,  6.27],
               [ 0.98,  0.48,  0.  ]])

    """
    if isinstance(data, FDataBasis) and n_points is None:
        raise ValueError(
            "Number of points to consider, should be given "
            + " as an argument for a FDataBasis. Instead None was passed.",
        )

    for interval_check in intervals:
        y1, y2 = interval_check
        if y2 < y1:
            raise ValueError(
                "Interval limits (a,b) should satisfy a <= b. "
                + str(interval_check) + " doesn't",
            )

    if n_points is None:
        function_x_coordinates = data.grid_points[0]
        function_y_coordinates = data.data_matrix
    else:
        function_x_coordinates = np.arange(
            data.domain_range[0][0],
            data.domain_range[0][1],
            (data.domain_range[0][1] - data.domain_range[0][0]) / n_points,
        )
        function_y_coordinates = data(function_x_coordinates)

    return np.asarray([
        _calculate_curves_occupation_(  # noqa: WPS317
            function_y_coordinates,
            function_x_coordinates,
            interval,
        )
        for interval in intervals
    ])
