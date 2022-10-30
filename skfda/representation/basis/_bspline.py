from __future__ import annotations

import warnings
from typing import Sequence

from ...typing._base import DomainRangeLike
from ._bspline_basis import BSplineBasis


class BSpline(BSplineBasis):
    r"""BSpline basis.

    BSpline basis elements are defined recursively as:

    .. math::
        B_{i, 1}(x) = 1 \quad \text{if } t_i \le x < t_{i+1},
        \quad 0 \text{ otherwise}

    .. math::
        B_{i, k}(x) = \frac{x - t_i}{t_{i+k} - t_i} B_{i, k-1}(x)
        + \frac{t_{i+k+1} - x}{t_{i+k+1} - t_{i+1}} B_{i+1, k-1}(x)

    Where k indicates the order of the spline.

    .. deprecated:: 0.8
        Use BSplineBasis instead.

    Implementation details: In order to allow a discontinuous behaviour at
    the boundaries of the domain it is necessary to placing m knots at the
    boundaries [RS05]_. This is automatically done so that the user only has to
    specify a single knot at the boundaries.


    Parameters:
        domain_range: A tuple of length 2 containing the initial and
            end values of the interval over which the basis can be evaluated.
        n_basis: Number of functions in the basis.
        order: Order of the splines. One greather than their degree.
        knots: List of knots of the spline functions.

    Examples:
        Constructs specifying number of basis and order.

        >>> bss = BSpline(n_basis=8, order=4)

        If no order is specified defaults to 4 because cubic splines are
        the most used. So the previous example is the same as:

        >>> bss = BSplineBasis(n_basis=8)

        It is also possible to create a BSpline basis specifying the knots.

        >>> bss = BSplineBasis(knots=[0, 0.2, 0.4, 0.6, 0.8, 1])

        Once we create a basis we can evaluate each of its functions at a
        set of points.

        >>> bss = BSplineBasis(n_basis=3, order=3)
        >>> bss([0, 0.5, 1])
        array([[[ 1.  ],
                [ 0.25],
                [ 0.  ]],
               [[ 0.  ],
                [ 0.5 ],
                [ 0.  ]],
               [[ 0.  ],
                [ 0.25],
                [ 1.  ]]])

        And evaluates first derivative

        >>> deriv = bss.derivative()
        >>> deriv([0, 0.5, 1])
        array([[[-2.],
                [-1.],
                [ 0.]],
               [[ 2.],
                [ 0.],
                [-2.]],
               [[ 0.],
                [ 1.],
                [ 2.]]])

    References:
        .. [RS05] Ramsay, J., Silverman, B. W. (2005). *Functional Data
            Analysis*. Springer. 50-51.

    """

    def __init__(  # noqa: WPS238
        self,
        domain_range: DomainRangeLike | None = None,
        n_basis: int | None = None,
        order: int = 4,
        knots: Sequence[float] | None = None,
    ) -> None:
        super().__init__(
            domain_range=domain_range,
            n_basis=n_basis,
            order=order,
            knots=knots,
        )
        warnings.warn(
            "The BSplines class is deprecated. Use "
            "BSplineBasis instead.",
            DeprecationWarning,
        )
