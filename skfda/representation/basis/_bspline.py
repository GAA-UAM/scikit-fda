import numpy as np
import scipy.interpolate
from numpy import polyint, polymul, polyval
from scipy.interpolate import BSpline as SciBSpline, PPoly

from ..._utils import _domain_range
from ._basis import Basis


class BSpline(Basis):
    r"""BSpline basis.

    BSpline basis elements are defined recursively as:

    .. math::
        B_{i, 1}(x) = 1 \quad \text{if } t_i \le x < t_{i+1},
        \quad 0 \text{ otherwise}

    .. math::
        B_{i, k}(x) = \frac{x - t_i}{t_{i+k} - t_i} B_{i, k-1}(x)
        + \frac{t_{i+k+1} - x}{t_{i+k+1} - t_{i+1}} B_{i+1, k-1}(x)

    Where k indicates the order of the spline.

    Implementation details: In order to allow a discontinuous behaviour at
    the boundaries of the domain it is necessary to placing m knots at the
    boundaries [RS05]_. This is automatically done so that the user only has to
    specify a single knot at the boundaries.

    Attributes:
        domain_range (tuple): A tuple of length 2 containing the initial and
            end values of the interval over which the basis can be evaluated.
        n_basis (int): Number of functions in the basis.
        order (int): Order of the splines. One greather than their degree.
        knots (list): List of knots of the spline functions.

    Examples:
        Constructs specifying number of basis and order.

        >>> bss = BSpline(n_basis=8, order=4)

        If no order is specified defaults to 4 because cubic splines are
        the most used. So the previous example is the same as:

        >>> bss = BSpline(n_basis=8)

        It is also possible to create a BSpline basis specifying the knots.

        >>> bss = BSpline(knots=[0, 0.2, 0.4, 0.6, 0.8, 1])

        Once we create a basis we can evaluate each of its functions at a
        set of points.

        >>> bss = BSpline(n_basis=3, order=3)
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

    def __init__(self, domain_range=None, n_basis=None, order=4, knots=None):
        """Bspline basis constructor.

        Args:
            domain_range (tuple, optional): Definition of the interval where
                the basis defines a space. Defaults to (0,1) if knots are not
                specified. If knots are specified defaults to the first and
                last element of the knots.
            n_basis (int, optional): Number of splines that form the basis.
            order (int, optional): Order of the splines. One greater that
                their degree. Defaults to 4 which mean cubic splines.
            knots (array_like): List of knots of the splines. If domain_range
                is specified the first and last elements of the knots have to
                match with it.

        """

        if domain_range is not None:
            domain_range = _domain_range(domain_range)

            if len(domain_range) != 1:
                raise ValueError("Domain range should be unidimensional.")

            domain_range = domain_range[0]

        # Knots default to equally space points in the domain_range
        if knots is None:
            if n_basis is None:
                raise ValueError("Must provide either a list of knots or the"
                                 "number of basis.")
        else:
            knots = tuple(knots)
            knots = sorted(knots)
            if domain_range is None:
                domain_range = (knots[0], knots[-1])
            else:
                if domain_range[0] != knots[0] or domain_range[1] != knots[-1]:
                    raise ValueError("The ends of the knots must be the same "
                                     "as the domain_range.")

        # n_basis default to number of knots + order of the splines - 2
        if n_basis is None:
            n_basis = len(knots) + order - 2

        if (n_basis - order + 2) < 2:
            raise ValueError(f"The number of basis ({n_basis}) minus the "
                             f"order of the bspline ({order}) should be "
                             f"greater than 3.")

        self._order = order
        self._knots = None if knots is None else tuple(knots)
        super().__init__(domain_range=domain_range, n_basis=n_basis)

        # Checks
        if self.n_basis != self.order + len(self.knots) - 2:
            raise ValueError(f"The number of basis ({self.n_basis}) has to "
                             f"equal the order ({self.order}) plus the "
                             f"number of knots ({len(self.knots)}) minus 2.")

    @property
    def knots(self):
        if self._knots is None:
            return tuple(np.linspace(*self.domain_range[0],
                                     self.n_basis - self.order + 2))
        else:
            return self._knots

    @property
    def order(self):
        return self._order

    def _evaluation_knots(self):
        """
        Get the knots adding m knots to the boundary in order to allow a
        discontinuous behaviour at the boundaries of the domain [RS05]_.

        References:
            .. [RS05] Ramsay, J., Silverman, B. W. (2005). *Functional Data
                Analysis*. Springer. 50-51.
        """
        return np.array((self.knots[0],) * (self.order - 1) + self.knots +
                        (self.knots[-1],) * (self.order - 1))

    def _evaluate(self, eval_points):

        # Input is scalar
        eval_points = eval_points[..., 0]

        # Places m knots at the boundaries
        knots = self._evaluation_knots()

        # c is used the select which spline the function splev below computes
        c = np.zeros(len(knots))

        # Initialise empty matrix
        mat = np.empty((self.n_basis, len(eval_points)))

        # For each basis computes its value for each evaluation point
        for i in range(self.n_basis):
            # write a 1 in c in the position of the spline calculated in each
            # iteration
            c[i] = 1
            # compute the spline
            mat[i] = scipy.interpolate.splev(eval_points,
                                             (knots, c, self.order - 1))
            c[i] = 0

        return mat

    def _derivative_basis_and_coefs(self, coefs, order=1):
        if order >= self.order:
            return (
                BSpline(n_basis=1, domain_range=self.domain_range, order=1),
                np.zeros((len(coefs), 1)))

        deriv_splines = [self._to_scipy_BSpline(coefs[i]).derivative(order)
                         for i in range(coefs.shape[0])]

        deriv_coefs = [BSpline._from_scipy_BSpline(spline)[1]
                       for spline in deriv_splines]

        deriv_basis = BSpline._from_scipy_BSpline(deriv_splines[0])[0]

        return deriv_basis, np.array(deriv_coefs)[:, 0:deriv_basis.n_basis]

    def rescale(self, domain_range=None):
        r"""Return a copy of the basis with a new domain range, with the
            corresponding values rescaled to the new bounds.
            The knots of the BSpline will be rescaled in the new interval.

            Args:
                domain_range (tuple, optional): Definition of the interval
                    where the basis defines a space. Defaults uses the same as
                    the original basis.
        """

        knots = np.array(self.knots, dtype=np.dtype('float'))

        if domain_range is not None:  # Rescales the knots
            knots -= knots[0]
            knots *= ((domain_range[1] - domain_range[0]
                       ) / (self.knots[-1] - self.knots[0]))
            knots += domain_range[0]

            # Fix possible round error
            knots[0] = domain_range[0]
            knots[-1] = domain_range[1]

        else:
            # TODO: Allow multiple dimensions
            domain_range = self.domain_range[0]

        return BSpline(domain_range, self.n_basis, self.order, knots)

    def __repr__(self):
        """Representation of a BSpline basis."""
        return (f"{self.__class__.__name__}(domain_range={self.domain_range}, "
                f"n_basis={self.n_basis}, order={self.order}, "
                f"knots={self.knots})")

    def _gram_matrix(self):
        # Places m knots at the boundaries
        knots = self._evaluation_knots()

        # c is used the select which spline the function
        # PPoly.from_spline below computes
        c = np.zeros(len(knots))

        # Initialise empty list to store the piecewise polynomials
        ppoly_lst = []

        no_0_intervals = np.where(np.diff(knots) > 0)[0]

        # For each basis gets its piecewise polynomial representation
        for i in range(self.n_basis):

            # Write a 1 in c in the position of the spline
            # transformed in each iteration
            c[i] = 1

            # Gets the piecewise polynomial representation and gets
            # only the positions for no zero length intervals
            # This polynomial are defined relatively to the knots
            # meaning that the column i corresponds to the ith knot.
            # Let the ith knot be a
            # Then f(x) = pp(x - a)
            pp = PPoly.from_spline((knots, c, self.order - 1))
            pp_coefs = pp.c[:, no_0_intervals]

            # We have the coefficients for each interval in coordinates
            # (x - a), so we will need to subtract a when computing the
            # definite integral
            ppoly_lst.append(pp_coefs)
            c[i] = 0

        # Now for each pair of basis computes the inner product after
        # applying the linear differential operator
        matrix = np.zeros((self.n_basis, self.n_basis))

        for interval in range(len(no_0_intervals)):
            for i in range(self.n_basis):
                poly_i = np.trim_zeros(ppoly_lst[i][:,
                                                    interval], 'f')
                # Indefinite integral
                square = polymul(poly_i, poly_i)
                integral = polyint(square)

                # Definite integral
                matrix[i, i] += np.diff(polyval(
                    integral, self.knots[interval: interval + 2]
                    - self.knots[interval]))[0]

                # The Gram matrix is banded, so not all intervals are used
                for j in range(i + 1, min(i + self.order, self.n_basis)):
                    poly_j = np.trim_zeros(ppoly_lst[j][:, interval], 'f')

                    # Indefinite integral
                    integral = polyint(polymul(poly_i, poly_j))

                    # Definite integral
                    matrix[i, j] += np.diff(polyval(
                        integral, self.knots[interval: interval + 2]
                        - self.knots[interval])
                    )[0]

                    # The matrix is symmetric
                    matrix[j, i] = matrix[i, j]

        return matrix

    def _to_scipy_BSpline(self, coefs):

        knots = np.concatenate((
            np.repeat(self.knots[0], self.order - 1),
            self.knots,
            np.repeat(self.knots[-1], self.order - 1)))

        return SciBSpline(knots, coefs, self.order - 1)

    @staticmethod
    def _from_scipy_BSpline(bspline):
        order = bspline.k
        knots = bspline.t

        # Remove additional knots at the borders
        if order != 0:
            knots = knots[order: -order]

        coefs = bspline.c
        domain_range = [knots[0], knots[-1]]

        return BSpline(domain_range, order=order + 1, knots=knots), coefs

    @property
    def inknots(self):
        """Return number of basis."""
        return self.knots[1:len(self.knots) - 1]

    def __eq__(self, other):
        return (super().__eq__(other)
                and self.order == other.order
                and self.knots == other.knots)

    def __hash__(self):
        return hash((super().__hash__(), self.order, self.knots))
