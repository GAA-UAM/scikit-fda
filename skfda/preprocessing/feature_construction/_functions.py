"""Functional Transformers Module."""

from __future__ import annotations

import itertools
from typing import Optional, Sequence, Tuple, TypeVar, Union, cast, overload

import numpy as np
from typing_extensions import Literal, Protocol, TypeGuard

from ..._utils import nquad_vec
from ...misc.validation import check_fdata_dimensions, validate_domain_range
from ...representation import FData, FDataBasis, FDataGrid
from ...typing._base import DomainRangeLike
from ...typing._numpy import ArrayLike, NDArrayBool, NDArrayFloat, NDArrayInt

T = TypeVar("T", bound=Union[NDArrayFloat, FDataGrid])


class _BasicUfuncProtocol(Protocol):

    @overload
    def __call__(self, __arg: FDataGrid) -> FDataGrid:  # noqa: WPS112
        pass

    @overload
    def __call__(self, __arg: NDArrayFloat) -> NDArrayFloat:  # noqa: WPS112
        pass

    def __call__(self, __arg: T) -> T:  # noqa: WPS112
        pass


def _sequence_of_ints(data: Sequence[object]) -> TypeGuard[Sequence[int]]:
    """Check that every element is an integer."""
    return all(isinstance(d, int) for d in data)


def local_averages(
    data: FData,
    *,
    domains: int | Sequence[int] | Sequence[DomainRangeLike],
) -> NDArrayFloat:
    r"""
    Calculate the local averages of given data in the desired domains.

    It takes functional data and performs the following map:

    .. math::
        f_1(X) = \frac{1}{|T_1|} \int_{T_1} X(t) dt,\dots, \\
        f_p(X) = \frac{1}{|T_p|} \int_{T_p} X(t) dt

    where :math:`T_1, \dots, T_p` are subregions of the original
    :term:`domain`.

    Args:
        data: FDataGrid or FDataBasis where we want to
            calculate the local averages.
        domains: Domains for each local average. It is possible to
            pass a number or a list of numbers to automatically split
            each dimension in that number of intervals and use them for
            the averages.

    Returns:
        ndarray of shape (n_samples, n_domains, codomain_dimension) with
        the transformed data.

    See also:
        :class:`~skfda.preprocessing.feature_construction.LocalAveragesTransformer`

    Examples:
        We import the Berkeley Growth Study dataset.
        We will use only the first 3 samples to make the
        example easy

        >>> from skfda.datasets import fetch_growth
        >>> dataset = fetch_growth(return_X_y=True)[0]
        >>> X = dataset[:3]

        We can choose the intervals used for the local averages. For example,
        we could in this case use the averages at different stages of
        development of the child: from 1 to 3 years, from 3 to 10 and from
        10 to 18:

        >>> import numpy as np
        >>> from skfda.preprocessing.feature_construction import local_averages
        >>> averages = local_averages(
        ...     X,
        ...     domains=[(1, 3), (3, 10), (10, 18)],
        ... )
        >>> np.around(averages, decimals=2)
        array([[[  91.37],
                [ 126.52],
                [ 179.02]],
               [[  87.51],
                [ 120.71],
                [ 158.81]],
               [[  86.36],
                [ 115.04],
                [ 156.37]]])

        A different possibility is to decide how many intervals we want to
        consider.  For example, we could want to split the domain in 2
        intervals of the same length.

        >>> np.around(local_averages(X, domains=2), decimals=2)
        array([[[ 116.94],
                [ 177.26]],
               [[ 111.86],
                [ 157.62]],
               [[ 107.29],
                [ 154.97]]])
    """
    if isinstance(domains, int):
        domains = [domains] * data.dim_domain

    if _sequence_of_ints(domains):
        # Get a list of arrays with the interval endpoints
        interval_endpoints = [
            np.linspace(
                domain_range[0],
                domain_range[1],
                num=n_intervals + 1,
            )
            for n_intervals, domain_range in zip(domains, data.domain_range)
        ]

        # Get all combinations of intervals as ranges
        domains = list(
            itertools.product(
                *[zip(p, p[1:]) for p in interval_endpoints],
            ),
        )

    domains = cast(Sequence[DomainRangeLike], domains)

    integrated_data = [
        unconditional_expected_value(
            data,
            lambda x: x,
            domain=domain,
        )
        for domain in domains
    ]
    return np.swapaxes(integrated_data, 0, 1)


def _calculate_curves_occupation(
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
    greater_than_y1 = (
        curve_y_coordinates[:, :, np.newaxis]
        >= y1[np.newaxis, np.newaxis, :]
    )
    less_than_y2 = (
        curve_y_coordinates[:, :, np.newaxis]
        <= y2[np.newaxis, np.newaxis, :]
    )
    inside_interval_bools = greater_than_y1 & less_than_y2

    # Calculate intervals on X axis where the points are inside Y axis interval
    intervals_x_inside = (
        inside_interval_bools
        * intervals_x_axis[:, np.newaxis]
    )

    return np.sum(intervals_x_inside, axis=1)  # type: ignore[no-any-return]


def occupation_measure(
    data: FData,
    intervals: Sequence[Tuple[float, float]],
    *,
    n_points: Optional[int] = None,
) -> NDArrayFloat:
    r"""
    Calculate the occupation measure of a grid.

    It performs the following map:

    ..math:
        :math:`f_1(X) = |t: X(t)\in T_p|,\dots,|t: X(t)\in T_p|`

    where :math:`{T_1,\dots,T_p}` are disjoint intervals in
    :math:`\mathbb{R}` and | | stands for the Lebesgue measure.

    The calculations are based on evaluation at a grid of points. In case of
    :class:`FDataGrid` the original grid is taken unless ``n_points`` is
    specified. In case of :class:`FDataBasis` it is mandatory to pass the
    number of points. If the result of this function is not accurate enough
    try to increase the grid of points.

    Args:
        data: Functional data where we want to calculate the occupation
            measure.
        intervals: ndarray of tuples containing the
            intervals we want to consider. The shape should be
            (n_sequences,2)
        n_points: Number of points to evaluate in the domain.
            By default will be used the points defined on the FDataGrid.
            On a FDataBasis this value should be specified.

    Returns:
        ndarray of shape (n_samples, n_intervals)
        with the transformed data.

    Examples:
        We will create the FDataGrid that we will use to extract
        the occupation measure.

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

        >>> from skfda.preprocessing.feature_construction import (
        ...     occupation_measure,
        ... )
        >>> np.around(
        ...     occupation_measure(
        ...         fd_grid,
        ...         [(0.0, 1.0), (2.0, 3.0)],
        ...         n_points=501,
        ...     ),
        ...     decimals=2,
        ... )
        array([[ 0.98,  1.  ],
               [ 0.5 ,  0.52],
               [ 6.28,  0.  ]])

    """
    if isinstance(data, FDataBasis) and n_points is None:
        raise ValueError(
            "Number of points to consider, should be given "
            + " as an argument for a FDataBasis. Instead None was passed.",
        )

    check_fdata_dimensions(
        data,
        dim_domain=1,
        dim_codomain=1,
    )

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

    return _calculate_curves_occupation(  # noqa: WPS317
        function_y_coordinates,
        function_x_coordinates,
        intervals,
    )


def number_crossings(
    fd: FDataGrid,
    *,
    levels: ArrayLike = 0,
    direction: Literal["up", "down", "all"] = "all",
) -> NDArrayInt:
    r"""
    Calculate the number of crossings to a level of a FDataGrid.

    Let f_1(X) = N_i, where N_i is the number of up crossings of X
    to a level c_i \in \mathbb{R}, i = 1,\dots,p.

    Recall that the process X(t) is said to have an up crossing of c
    at :math:`t_0 > 0` if for some :math:`\epsilon >0`, X(t) $\leq$
    c if t :math:'\in (t_0 - \epsilon, t_0) and X(t) $\geq$ c if
    :math:`t\in (t_0, t_0+\epsilon)`.

    If the trajectories are differentiable, then
    :math:`N_i = card\{t \in[a,b]: X(t) = c_i, X' (t) > 0\}.`

        Args:
            fd: FDataGrid where we want to calculate
                the number of up crossings.
            levels: Sequence of numbers including the levels
                we want to consider for the crossings. By
                default it calculates zero-crossings.
            direction: Whether to consider only up-crossings,
                down-crossings or both.

        Returns:
            ndarray of shape (n_samples, len(levels))\
            with the values of the counters.

    Examples:
        For this example we will use a well known function so the correct
        functioning of this method can be checked.
        We will create and use a DataFrame with a sample extracted from
        the Bessel Function of first type and order 0.
        First of all we import the Bessel Function and create the X axis
        data grid. Then we create the FdataGrid.

        >>> from skfda.preprocessing.feature_construction import (
        ...     number_crossings,
        ... ) 
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

        Finally we evaluate the number of up crossings method with the
        FDataGrid created.

        >>> number_crossings(fd_grid, levels=0, direction="up")
        array([[2]])
    """
    # This is only defined for univariate functions
    check_fdata_dimensions(fd, dim_domain=1, dim_codomain=1)

    levels = np.atleast_1d(levels)
    curves = fd.data_matrix[:, :, 0]

    distances = np.subtract.outer(levels, curves)

    points_greater = distances >= 0
    points_smaller = distances <= 0

    growing = distances[:, :, :-1] < points_greater[:, :, 1:]
    lowering = distances[:, :, :-1] > points_greater[:, :, 1:]

    upcrossing_positions: NDArrayBool = (
        points_smaller[:, :, :-1] & points_greater[:, :, 1:] & growing
    )

    downcrossing_positions: NDArrayBool = (
        points_greater[:, :, :-1] & points_smaller[:, :, 1:] & lowering
    )

    positions = {
        "all": upcrossing_positions | downcrossing_positions,
        "up": upcrossing_positions,
        "down": downcrossing_positions,
    }

    return np.sum(  # type: ignore[no-any-return]
        positions[direction],
        axis=2,
    ).T


def unconditional_central_moment(
    data: FDataGrid,
    n: int,
    *,
    domain: DomainRangeLike | None = None,
) -> NDArrayFloat:
    r"""
    Calculate a specified unconditional central moment.

    The unconditional central moments are defined as the unconditional
    moments where the mean is subtracted from each sample before the
    integration. The n-th unconditional central moment is calculated as
    follows, where p is the number of observations:

    .. math::
        f_1(x(t))=\frac{1}{\left(b-a\right)}\int_a^b
        \left(x_1(t) - \mu_1\right)^n dt, \dots,
        f_p(x(t))=\frac{1}{\left(b-a\right)}\int_a^b
        \left(x_p(t) - \mu_p\right)^n dt

        Args:
            data: FDataGrid where we want to calculate
                a particular unconditional central moment.
            n: order of the moment.
            domain: Integration domain. By default, the whole domain is used.

        Returns:
            ndarray of shape (n_dimensions, n_samples) with the values of the
            specified moment.

    Example:
        We will calculate the first unconditional central moment of the
        Canadian Weather data set. In order to simplify the example, we will
        use only the first five samples.
        First we proceed to import the data set.

        >>> from skfda.datasets import fetch_weather
        >>> X = fetch_weather(return_X_y=True)[0]

        Then we call the function with the samples that we want to consider and
        the specified moment order.

        >>> import numpy as np
        >>> from skfda.preprocessing.feature_construction import (
        ...    unconditional_central_moment,
        ... )
        >>> np.around(unconditional_central_moment(X[:5], 1), decimals=2)
        array([[ 0.01,  0.01],
               [ 0.02,  0.01],
               [ 0.02,  0.01],
               [ 0.02,  0.01],
               [ 0.01,  0.01]])

    """
    mean = unconditional_expected_value(
        data,
        lambda x: x,
        domain=domain,
    )

    return unconditional_expected_value(
        data,
        lambda x: np.power(x - mean, n),
        domain=domain,
    )


def unconditional_moment(
    data: Union[FDataBasis, FDataGrid],
    n: int,
    *,
    domain: DomainRangeLike | None = None,
) -> NDArrayFloat:
    r"""
    Calculate a specified unconditional moment.

    The n-th unconditional moments of p real-valued continuous functions
    are calculated as:
    .. math::
        f_1(x(t))=\frac{1}{\left( b-a\right)}\int_a^b \left(x_1(t)\right)^ndt,
        \dots,
        f_p(x(t))=\frac{1}{\left( b-a\right)}\int_a^b  \left(x_p(t)\right)^n dt
        Args:
            data: FDataGrid or FDataBasis where we want to calculate
                a particular unconditional moment.
            n: Order of the moment.
            domain: Integration domain. By default, the whole domain is used.

        Returns:
            ndarray of shape (n_dimensions, n_samples) with the values of the
            specified moment.

    Examples:
        We will calculate the first unconditional moment of the Canadian
        Weather data set. In order to simplify the example, we will use only
        the first five samples.
        First we proceed to import the data set.

        >>> from skfda.datasets import fetch_weather
        >>> X = fetch_weather(return_X_y=True)[0]

        Then we call the function with the samples that we want to consider and
        the specified moment order.

        >>> import numpy as np
        >>> from skfda.preprocessing.feature_construction import (
        ...     unconditional_moment,
        ... )
        >>> np.around(unconditional_moment(X[:5], 1), decimals=2)
        array([[ 4.7 ,  4.03],
               [ 6.16,  3.96],
               [ 5.52,  4.01],
               [ 6.82,  3.44],
               [ 5.25,  3.29]])

    """
    return unconditional_expected_value(
        data,
        lambda x: np.power(x, n),
        domain=domain,
    )


def unconditional_expected_value(
    data: FData,
    function: _BasicUfuncProtocol,
    *,
    domain: DomainRangeLike | None = None,
) -> NDArrayFloat:
    r"""
    Calculate the unconditional expected value of a function.

    Next formula shows for a defined transformation :math: `g(x(t))`
    and p observations, how the unconditional expected values are calculated:

    .. math::
            f_1(x(t))=\frac{1}{\left( b-a\right)}\int_a^b g
            \left(x_1(t)\right)dt,\dots,
            f_p(x(t))=\frac{1}{\left( b-a\right)}\int_a^b g
            \left(x_p(t)\right) dt

        Args:
            data: FDataGrid or FDataBasis where we want to calculate
                the expected value.
            function: function that specifies how the expected value to is
                calculated. It has to be a function of X(t).
            domain: Integration domain. By default, the whole domain is used.

        Returns:
            ndarray of shape (n_dimensions, n_samples) with the values of the
            expectations.

    Examples:
        We will use this funtion to calculate the logarithmic first moment
        of the first 5 samples of the Berkeley Growth dataset.
        We will start by importing it.

        >>> from skfda.datasets import fetch_growth
        >>> X = fetch_growth(return_X_y=True)[0]

        We will define a function that calculates the inverse first moment.

        >>> import numpy as np
        >>> f = lambda x: np.power(np.log(x), 1)

        Then we call the function with the dataset and the function.

        >>> from skfda.preprocessing.feature_construction import (
        ...     unconditional_expected_value,
        ... )
        >>> np.around(unconditional_expected_value(X[:5], f), decimals=2)
            array([[ 4.96],
                   [ 4.88],
                   [ 4.85],
                   [ 4.9 ],
                   [ 4.84]])

    """
    if domain is None:
        domain = data.domain_range
    else:
        domain = validate_domain_range(domain)

    lebesgue_measure = np.prod(
        [
            (iterval[1] - iterval[0])
            for iterval in domain
        ],
    )

    if isinstance(data, FDataGrid):
        return function(data).integrate(domain=domain) / lebesgue_measure

    def integrand(*args: NDArrayFloat) -> NDArrayFloat:  # noqa: WPS430
        f1 = data(args)[:, 0, :]
        return function(f1)

    return nquad_vec(
        integrand,
        domain,
    ) / lebesgue_measure
