"""Functional Transformers Module."""

from __future__ import annotations

from typing import Optional, Sequence, Tuple, Union

import numpy as np

from ..._utils import check_is_univariate
from ...representation import FDataBasis, FDataGrid
from ...representation._typing import NDArrayFloat


def local_averages(
    data: Union[FDataGrid, FDataBasis],
    n_intervals: int,
) -> np.ndarray:
    r"""
    Calculate the local averages of a given data.

    Take functional data as a grid or a basis and performs
    the following map:

    .. math::
        f_1(X) = \frac{1}{|T_1|} \int_{T_1} X(t) dt,\dots, \\
        f_p(X) = \frac{1}{|T_p|} \int_{T_p} X(t) dt

    where {T_1,\dots,T_p} are disjoint intervals of the interval [a,b]

    It is calculated for a given number of intervals,
    which are of equal sizes.
    Args:
        data: FDataGrid or FDataBasis where we want to
        calculate the local averages.
        n_intervals: number of intervals we want to consider.
    Returns:
        ndarray of shape (n_intervals, n_samples, n_dimensions)
        with the transformed data for FDataBasis and (n_intervals, n_samples)
        for FDataGrid.

    Example:

        We import the Berkeley Growth Study dataset.
        We will use only the first 30 samples to make the
        example easy.
        >>> from skfda.datasets import fetch_growth
        >>> dataset = fetch_growth(return_X_y=True)[0]
        >>> X = dataset[:30]

        Then we decide how many intervals we want to consider (in our case 2)
        and call the function with the dataset.
        >>> import numpy as np
        >>> from skfda.exploratory.stats import local_averages
        >>> np.around(local_averages(X, 2), decimals=2)
        array([[ 116.94,  111.86,  107.29,  111.35,  104.39,  109.43,  109.16,
                 112.91,  109.19,  117.95,  112.14,  114.3 ,  111.48,  114.85,
                 116.25,  114.6 ,  111.02,  113.57,  108.88,  109.6 ,  109.7 ,
                 108.54,  109.18,  106.92,  109.44,  109.84,  115.32,  108.16,
                 119.29,  110.62],
               [ 177.26,  157.62,  154.97,  163.83,  156.66,  157.67,  155.31,
                 169.02,  154.18,  174.43,  161.33,  170.14,  164.1 ,  170.1 ,
                 166.65,  168.72,  166.85,  167.22,  159.4 ,  162.76,  155.7 ,
                 158.01,  160.1 ,  155.95,  157.95,  163.53,  162.29,  153.1 ,
                 178.48,  161.75]])
    """
    left, right = data.domain_range[0]

    intervals, step = np.linspace(
        left,
        right,
        num=n_intervals + 1,
        retstep=True,
    )

    integrated_data = [
        data.integrate(interval=((intervals[i], intervals[i + 1]))) / step
        for i in range(n_intervals)
    ]
    return np.asarray(integrated_data)


def _calculate_curves_occupation_(
    curve_y_coordinates: NDArrayFloat,
    curve_x_coordinates: NDArrayFloat,
    intervals: Sequence[Tuple[float, float]],
) -> NDArrayFloat:

    y1, y2 = np.asarray(intervals).T

    if any(np.greater(y1, y2)):
        raise ValueError(
            "Interval limits (a,b) should satisfy a <= b.",
        )

    # Reshape original curves so they have one dimension less
    curve_y_coordinates = curve_y_coordinates[:, :, 0]

    # Calculate interval sizes on the X axis
    intervals_x_axis = curve_x_coordinates[1:] - curve_x_coordinates[:-1]

    # Calculate which points are inside the interval given (y1,y2) on Y axis
    greater_than_y1 = curve_y_coordinates >= y1[:, np.newaxis, np.newaxis]
    less_than_y2 = curve_y_coordinates <= y2[:, np.newaxis, np.newaxis]
    inside_interval_bools = greater_than_y1 & less_than_y2

    # Calculate intervals on X axis where the points are inside Y axis interval
    intervals_x_inside = inside_interval_bools * intervals_x_axis

    return np.sum(intervals_x_inside, axis=2)


def occupation_measure(
    data: Union[FDataGrid, FDataBasis],
    intervals: Sequence[Tuple[float, float]],
    *,
    n_points: Optional[int] = None,
) -> NDArrayFloat:
    r"""
    Calculate the occupation measure of a grid.

    It performs the following map.
        ..math:
            :math:`f_1(X) = |t: X(t)\in T_p|,\dots,|t: X(t)\in T_p|`

        where :math:`{T_1,\dots,T_p}` are disjoint intervals in
        :math:`\mathbb{R}` and | | stands for the Lebesgue measure.

    The calculations are based on the grid of points of the x axis. In case of
    FDataGrid the original grid is taken unless n_points is specified. In case
    of FDataBasis the number of points of the x axis to be considered is passed
    through the n_points parameter compulsory.
    If the result of this function is not accurate enough try to increase the
    grid of points of the x axis. Either by increasing n_points or passing a
    FDataGrid with more x grid points per curve.


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
        array([[ 0.98,  0.5 ,  6.28],
               [ 1.02,  0.52,  0.  ]])

    """
    if isinstance(data, FDataBasis) and n_points is None:
        raise ValueError(
            "Number of points to consider, should be given "
            + " as an argument for a FDataBasis. Instead None was passed.",
        )

    check_is_univariate(data)

    if n_points is None:
        function_x_coordinates = data.grid_points[0]
        function_y_coordinates = data.data_matrix
    else:
        function_x_coordinates = np.linspace(
            data.domain_range[0][0],
            data.domain_range[0][1],
            num=n_points,
        )
        function_y_coordinates = data(function_x_coordinates[1:])

    return _calculate_curves_occupation_(  # noqa: WPS317
        function_y_coordinates,
        function_x_coordinates,
        intervals,
    )
