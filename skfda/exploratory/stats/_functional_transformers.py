"""Functional Transformers Module."""

from __future__ import annotations

from typing import Callable, Optional, Sequence, Tuple, Union

import numpy as np

from ..._utils import check_is_univariate
from ...representation import FDataBasis, FDataGrid
from ...representation._typing import NDArrayFloat, NDArrayInt


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
        array([[[ 116.94],
                [ 111.86],
                [ 107.29],
                [ 111.35],
                [ 104.39],
                [ 109.43],
                [ 109.16],
                [ 112.91],
                [ 109.19],
                [ 117.95],
                [ 112.14],
                [ 114.3 ],
                [ 111.48],
                [ 114.85],
                [ 116.25],
                [ 114.6 ],
                [ 111.02],
                [ 113.57],
                [ 108.88],
                [ 109.6 ],
                [ 109.7 ],
                [ 108.54],
                [ 109.18],
                [ 106.92],
                [ 109.44],
                [ 109.84],
                [ 115.32],
                [ 108.16],
                [ 119.29],
                [ 110.62]],
               [[ 177.26],
                [ 157.62],
                [ 154.97],
                [ 163.83],
                [ 156.66],
                [ 157.67],
                [ 155.31],
                [ 169.02],
                [ 154.18],
                [ 174.43],
                [ 161.33],
                [ 170.14],
                [ 164.1 ],
                [ 170.1 ],
                [ 166.65],
                [ 168.72],
                [ 166.85],
                [ 167.22],
                [ 159.4 ],
                [ 162.76],
                [ 155.7 ],
                [ 158.01],
                [ 160.1 ],
                [ 155.95],
                [ 157.95],
                [ 163.53],
                [ 162.29],
                [ 153.1 ],
                [ 178.48],
                [ 161.75]]])

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


def number_up_crossings(
    data: FDataGrid,
    levels: NDArrayFloat,
) -> NDArrayInt:
    r"""
    Calculate the number of up crossings to a level of a FDataGrid.

    Let f_1(X) = N_i, where N_i is the number of up crossings of X
    to a level c_i \in \mathbb{R}, i = 1,\dots,p.

    Recall that the process X(t) is said to have an up crossing of c
    at :math:`t_0 > 0` if for some :math:`\epsilon >0`, X(t) $\leq$
    c if t :math:'\in (t_0 - \epsilon, t_0) and X(t) $\geq$ c if
    :math:`t\in (t_0, t_0+\epsilon)`.

    If the trajectories are differentiable, then
    :math:`N_i = card\{t \in[a,b]: X(t) = c_i, X' (t) > 0\}.`

        Args:
            data: FDataGrid where we want to calculate
            the number of up crossings.
            levels: sequence of numbers including the levels
            we want to consider for the crossings.
        Returns:
            ndarray of shape (n_samples, len(levels))\
            with the values of the counters.

    Example:

    For this example we will use a well known function so the correct
    functioning of this method can be checked.
    We will create and use a DataFrame with a sample extracted from
    the Bessel Function of first type and order 0.
    First of all we import the Bessel Function and create the X axis
    data grid. Then we create the FdataGrid.
    >>> from skfda.exploratory.stats import number_up_crossings
    >>> from scipy.special import jv
    >>> import numpy as np
    >>> x_grid = np.linspace(0, 14, 14)
    >>> fd_grid = FDataGrid(
    ...     data_matrix=[jv([0], x_grid)],
    ...     grid_points=x_grid,
    ... )
    >>> fd_grid.data_matrix
    array([[[ 1.        ],
            [ 0.73041066],
            [ 0.13616752],
            [-0.32803875],
            [-0.35967936],
            [-0.04652559],
            [ 0.25396879],
            [ 0.26095573],
            [ 0.01042895],
            [-0.22089135],
            [-0.2074856 ],
            [ 0.0126612 ],
            [ 0.20089319],
            [ 0.17107348]]])

    Finally we evaluate the number of up crossings method with the FDataGrid
    created.
    >>> number_up_crossings(fd_grid, np.asarray([0]))
    array([[2]])
    """
    curves = data.data_matrix[:, :, 0]

    distances = np.asarray([
        level - curves
        for level in levels
    ])

    points_greater = distances >= 0
    points_smaller = distances <= 0
    points_smaller_rotated = np.concatenate(
        [
            points_smaller[:, :, 1:],
            points_smaller[:, :, :1],
        ],
        axis=2,
    )

    return np.sum(
        points_greater & points_smaller_rotated,
        axis=2,
    ).T


def moments_of_norm(
    data: FDataGrid,
    p: int,
) -> np.ndarray:
    r"""
    Calculate the moments of the norm of the process of a FDataGrid.

    It performs the following map:
    :math:`f_1(X)=\mathbb{E}(||X||),\dots,f_p(X)=\mathbb{E}(||X||^p)`.

        Args:
            data: FDataGrid where we want to calculate
            the moments of the norm of the process.
        Returns:
            ndarray of shape (n_dimensions, n_samples)\
            with the values of the moments.

    Example:

    We import the Canadian Weather dataset
    >>> from skfda.datasets import fetch_weather
    >>> X = fetch_weather(return_X_y=True)[0]

    Then we call the function with the dataset.
    >>> import numpy as np
    >>> from skfda.exploratory.stats import moments_of_norm
    >>> np.around(moments_of_norm(X, 1), decimals=2)
        array([[  7.05,   4.06],
               [  8.98,   3.99],
               [  8.38,   4.04],
               [  8.13,   3.46],
               [  9.17,   3.29],
               [  9.95,   3.09],
               [ 11.55,   2.2 ],
               [ 10.92,   2.46],
               [ 10.84,   2.55],
               [ 10.53,   3.31],
               [ 10.02,   3.04],
               [ 10.97,   2.58],
               [ 11.01,   2.5 ],
               [ 10.15,   2.14],
               [ 10.18,   2.62],
               [ 10.36,   1.93],
               [ 12.36,   1.4 ],
               [ 12.28,   1.23],
               [ 13.13,   1.12],
               [ 11.62,   1.02],
               [ 12.05,   1.11],
               [ 13.5 ,   0.99],
               [ 10.16,   1.27],
               [  8.83,   1.1 ],
               [ 10.32,   0.74],
               [  9.96,   3.16],
               [  9.62,   2.33],
               [  8.4 ,   1.67],
               [  7.02,   7.1 ],
               [ 10.06,   0.74],
               [ 14.29,   0.9 ],
               [ 14.47,   0.73],
               [ 13.05,   1.14],
               [ 15.97,   0.71],
               [ 17.65,   0.39]])
    """
    return moments(data, lambda x: pow(np.abs(x), 1))


def moments(
    data: FDataGrid,
    f: Callable[[np.ndarray], np.ndarray],
) -> np.ndarray:
    """
    Calculate the moments of a FDataGrid.

        Args:
            data: FDataGrid where we want to calculate
            the moments.
            f: function that specifies the moments to be calculated.
        Returns:
            ndarray of shape (n_dimensions, n_samples)\
            with the values of the moments.

    Example:
    We will use this funtion to calculate the moments of the
    norm of a FDataGrid.
    We will first import the Canadian Weather dataset.
    >>> from skfda.datasets import fetch_weather
    >>> X = fetch_weather(return_X_y=True)[0]

    We will define a function that calculates the moments of the norm.
    >>> f = lambda x: pow(np.abs(x), 1)

    Then we call the function with the dataset and the function.
    >>> import numpy as np
    >>> from skfda.exploratory.stats import moments

    >>> np.around(moments(X, f), decimals=2)
    array([[  7.05,   4.06],
           [  8.98,   3.99],
           [  8.38,   4.04],
           [  8.13,   3.46],
           [  9.17,   3.29],
           [  9.95,   3.09],
           [ 11.55,   2.2 ],
           [ 10.92,   2.46],
           [ 10.84,   2.55],
           [ 10.53,   3.31],
           [ 10.02,   3.04],
           [ 10.97,   2.58],
           [ 11.01,   2.5 ],
           [ 10.15,   2.14],
           [ 10.18,   2.62],
           [ 10.36,   1.93],
           [ 12.36,   1.4 ],
           [ 12.28,   1.23],
           [ 13.13,   1.12],
           [ 11.62,   1.02],
           [ 12.05,   1.11],
           [ 13.5 ,   0.99],
           [ 10.16,   1.27],
           [  8.83,   1.1 ],
           [ 10.32,   0.74],
           [  9.96,   3.16],
           [  9.62,   2.33],
           [  8.4 ,   1.67],
           [  7.02,   7.1 ],
           [ 10.06,   0.74],
           [ 14.29,   0.9 ],
           [ 14.47,   0.73],
           [ 13.05,   1.14],
           [ 15.97,   0.71],
           [ 17.65,   0.39]])

    """
    return np.mean(f(data.data_matrix), axis=1)
