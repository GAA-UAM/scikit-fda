"""This module defines functional data object in a basis function system
representation and the corresponding basis class.

"""

from abc import ABC, abstractmethod

import numpy
import matplotlib.pyplot
import scipy.interpolate
import scipy.linalg
import scipy.integrate
from numpy import polyder, polyint, polymul, polyval
from scipy.interpolate import PPoly
from scipy.special import binom

from . import grid

__author__ = "Miguel Carbajo Berrocal"
__email__ = "miguel.carbajo@estudiante.uam.es"


# aux functions
def _polypow(p, n=2):
    if n > 2:
        return polymul(p, _polypow(p, n - 1))
    if n == 2:
        return polymul(p, p)
    elif n == 1:
        return p
    elif n == 0:
        return [1]
    else:
        raise ValueError("n must be greater than 0.")


class Basis(ABC):
    """Defines the structure of a basis function system.

    Attributes:
        domain_range (tuple): a tuple of length 2 containing the initial and
            end values of the interval over which the basis can be evaluated.
        nbasis (int): number of functions in the basis.

    """

    def __init__(self, domain_range=(0, 1), nbasis=1):
        """Basis constructor.

        Args:
            domain_range (tuple, optional): Definition of the interval where
                the basis defines a space. Defaults to (0,1).
            nbasis: Number of functions that form the basis. Defaults to 1.
        """
        # Some checks
        if domain_range[0] >= domain_range[1]:
            raise ValueError("The interval {} is not well-defined.".format(
                domain_range))
        if nbasis < 1:
            raise ValueError("The number of basis has to be strictly "
                             "possitive.")
        self.domain_range = domain_range
        self.nbasis = nbasis
        self._drop_index_lst = []

        super().__init__()

    @abstractmethod
    def _compute_matrix(self, eval_points, derivative=0):
        """Computes the basis or its derivatives given a list of values.

        Args:
            eval_points (array_like): List of points where the basis is
                evaluated.
            derivative (int, optional): Order of the derivative. Defaults to 0.

        Returns:
            (:obj:`numpy.darray`): Matrix whose rows are the values of the each
            basis function or its derivatives at the values specified in
            eval_points.

        """
        pass

    def evaluate(self, eval_points, derivative=0):
        """Evaluates the basis function system or its derivatives at a list of
        given values.

        Args:
            eval_points (array_like): List of points where the basis is
                evaluated.
            derivative (int, optional): Order of the derivative. Defaults to 0.

        Returns:
            (numpy.darray): Matrix whose rows are the values of the each
            basis function or its derivatives at the values specified in
            eval_points.

        """
        eval_points = numpy.asarray(eval_points)
        if numpy.any(numpy.isnan(eval_points)):
            raise ValueError("The list of points where the function is "
                             "evaluated can not contain nan values.")

        return self._compute_matrix(eval_points, derivative)

    def plot(self, ax=None, derivative=0, **kwargs):
        """Plots the basis object or its derivatives.

        Args:
            ax (axis object, optional): axis over with the graphs are plotted.
                Defaults to matplotlib current axis.
            derivative (int, optional): Order of the derivative. Defaults to 0.
            **kwargs: keyword arguments to be [RS05]_passed to the
                matplotlib.pyplot.plot function.

        Returns:
            List of lines that were added to the plot.

        """
        if ax is None:
            ax = matplotlib.pyplot.gca()
        # Number of points where the basis are evaluated
        npoints = max(501, 10 * self.nbasis)
        # List of points where the basis are evaluated
        eval_points = numpy.linspace(self.domain_range[0],
                                     self.domain_range[1],
                                     npoints)
        # Basis evaluated in the previous list of points
        mat = self.evaluate(eval_points, derivative)
        # Plot
        return ax.plot(eval_points, mat.T, **kwargs)

    def _evaluate_single_basis_coefficients(self, coefficients, basis_index, x,
                                            cache):
        if x not in cache:
            res = numpy.zeros(self.nbasis)
            for i, k in enumerate(coefficients):
                if callable(k):
                    res += k(x) * self._compute_matrix([x], i)[:, 0]
                else:
                    res += k * self._compute_matrix([x], i)[:, 0]
            cache[x] = res
        return cache[x][basis_index]

    def _numerical_penalty(self, coefficients):
        penalty_matrix = numpy.zeros((self.nbasis, self.nbasis))
        cache = {}
        for i in range(self.nbasis):
            penalty_matrix[i, i] = scipy.integrate.quad(
                lambda x: (self._evaluate_single_basis_coefficients(
                    coefficients, i, x, cache) ** 2),
                self.domain_range[0], self.domain_range[1]
            )[0]
            for j in range(i + 1, self.nbasis):
                penalty_matrix[i, j] = scipy.integrate.quad(
                    lambda x: (self._evaluate_single_basis_coefficients(
                        coefficients, i, x, cache)
                               * self._evaluate_single_basis_coefficients(
                                coefficients, j, x, cache)),
                    self.domain_range[0], self.domain_range[1]
                )[0]
                penalty_matrix[j, i] = penalty_matrix[i, j]
        return penalty_matrix

    @abstractmethod
    def penalty(self, derivative_degree=None, coefficients=None):
        r""" Returns a penalty matrix given a differential operator.

        The differential operator can be either a derivative of a certain
        degree or a more complex operator.

        The penalty matrix is defined as [RS05-5-6-2]_:

        .. math::
            R_{ij} = \int L\phi_i(s) L\phi_j(s) ds

        where :math:`\phi_i(s)` for :math:`i=1, 2, ..., n` are the basis
        functions and :math:`L` is a differential operator.

        Args:
            derivative_degree (int): Integer indicating the order of the
                derivative or . For instance 2 means that the differential
                operator is :math:`f''(x)`.
            coefficients (list): List of coefficients representing a
                differential operator. An iterable indicating
                coefficients of derivatives (which can be functions). For
                instance the tuple (1, 0, numpy.sin) means :math:`1
                + sin(x)D^{2}`. Only used if derivative degree is None.

        Returns:
            numpy.array: Penalty matrix.

        References:
            .. [RS05-5-6-2] Ramsay, J., Silverman, B. W. (2005). Specifying the
               roughness penalty. In *Functional Data Analysis* (pp. 106-107).
               Springler.

        """
        pass

    def __repr__(self):
        return "{}(domain_range={}, nbasis={})".format(
            self.__class__.__name__, self.domain_range, self.nbasis)


class Monomial(Basis):
    """Monomial basis.

    Basis formed by powers of the argument :math:`t`:

    .. math::
        1, t, t^2, t^3...

    Attributes:
        domain_range (tuple): a tuple of length 2 containing the initial and
            end values of the interval over which the basis can be evaluated.
        nbasis (int): number of functions in the basis.

    Examples:
        Defines a monomial base over the interval :math:`[0, 5]` consisting
        on the first 3 powers of :math:`t`: :math:`1, t, t^2`.

        >>> bs_mon = Monomial((0,5), nbasis=3)

        And evaluates all the functions in the basis in a list of descrete
        values.

        >>> bs_mon.evaluate([0, 1, 2])
        array([[1., 1., 1.],
               [0., 1., 2.],
               [0., 1., 4.]])

        And also evaluates its derivatives

        >>> bs_mon.evaluate([0, 1, 2], derivative=1)
        array([[0., 0., 0.],
               [1., 1., 1.],
               [0., 2., 4.]])
        >>> bs_mon.evaluate([0, 1, 2], derivative=2)
        array([[0., 0., 0.],
               [0., 0., 0.],
               [2., 2., 2.]])

    """

    def _compute_matrix(self, eval_points, derivative=0):
        """Computes the basis or its derivatives given a list of values.

        For each of the basis computes its value for each of the points in
        the list passed as argument to the method.

        Args:
            eval_points (array_like): List of points where the basis is
                evaluated.
            derivative (int, optional): Order of the derivative. Defaults to 0.

        Returns:
            (:obj:`numpy.darray`): Matrix whose rows are the values of the each
            basis function or its derivatives at the values specified in
            eval_points.

        """
        # Initialise empty matrix
        mat = numpy.zeros((self.nbasis, len(eval_points)))

        # For each basis computes its value for each evaluation
        if derivative == 0:
            for i in range(self.nbasis):
                mat[i] = eval_points ** i
        else:
            for i in range(self.nbasis):
                if derivative <= i:
                    factor = i
                    for j in range(2, derivative + 1):
                        factor *= (i - j + 1)
                    mat[i] = factor * eval_points ** (i - derivative)

        return mat

    def penalty(self, derivative_degree=None, coefficients=None):
        r""" Returns a penalty matrix given a differential operator.

        The differential operator can be either a derivative of a certain
        degree or a more complex operator.

        The penalty matrix is defined as [RS05-5-6-2]_:

        .. math::
            R_{ij} = \int L\phi_i(s) L\phi_j(s) ds

        where :math:`\phi_i(s)` for :math:`i=1, 2, ..., n` are the basis
        functions and :math:`L` is a differential operator.

        Args:
            derivative_degree (int): Integer indicating the order of the
                derivative or . For instance 2 means that the differential
                operator is :math:`f''(x)`.
            coefficients (list): List of coefficients representing a
                differential operator. An iterable indicating
                coefficients of derivatives (which can be functions). For
                instance the tuple (1, 0, numpy.sin) means :math:`1
                + sin(x)D^{2}`. Only used if derivative degree is None.


        Returns:
            numpy.array: Penalty matrix.

        Examples:

            >>> Monomial(nbasis=4).penalty(2)
            array([[ 0.,  0.,  0.,  0.],
                   [ 0.,  0.,  0.,  0.],
                   [ 0.,  0.,  4.,  6.],
                   [ 0.,  0.,  6., 12.]])

        References:
            .. [RS05-5-6-2] Ramsay, J., Silverman, B. W. (2005). Specifying the
                roughness penalty. In *Functional Data Analysis* (pp. 106-107).
                Springler.

        """
        if derivative_degree is None:
            return self._numerical_penalty(coefficients)

        integration_domain = self.domain_range

        # initialize penalty matrix as all zeros
        penalty_matrix = numpy.zeros((self.nbasis, self.nbasis))
        # iterate over the cartesion product of the basis system with itself
        for ibasis in range(self.nbasis):
            # notice that the index ibasis it is also the exponent of the
            # monomial
            # ifac is the factor resulting of deriving the monomial as many
            # times as indicates de differential operator
            if derivative_degree > 0:
                ifac = ibasis
                for k in range(2, derivative_degree + 1):
                    ifac *= ibasis - k + 1
            else:
                ifac = 1

            for jbasis in range(self.nbasis):
                # notice that the index jbasis it is also the exponent of the
                # monomial
                # jfac is the factor resulting of deriving the monomial as
                # many times as indicates de differential operator
                if derivative_degree > 0:
                    jfac = jbasis
                    for k in range(2, derivative_degree + 1):
                        jfac *= jbasis - k + 1
                else:
                    jfac = 1

                # if any of the two monomial has lower degree than the order of
                # the derivative indicated by the differential operator that
                # factor equals 0, so no calculation are needed
                if (ibasis >= derivative_degree
                        and jbasis >= derivative_degree):
                    # Calculates exactly the result of the integral
                    # Exponent after applying the differential operator and
                    # integrating
                    ipow = ibasis + jbasis - 2 * derivative_degree + 1
                    # coefficient after integrating
                    penalty_matrix[ibasis, jbasis] = (
                            (integration_domain[1] ** ipow
                             - integration_domain[0] ** ipow)
                            * ifac * jfac / ipow)
                    penalty_matrix[jbasis, ibasis] = penalty_matrix[ibasis,
                                                                    jbasis]

        return penalty_matrix


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
        nbasis (int): Number of functions in the basis.
        order (int): Order of the splines. One greather than their degree.
        knots (list): List of knots of the spline functions.

    Examples:
        Constructs specifying number of basis and order.

        >>> bss = BSpline(nbasis=8, order=4)

        If no order is specified defaults to 4 because cubic splines are
        the most used. So the previous example is the same as:

        >>> bss = BSpline(nbasis=8)

        It is also possible to create a BSpline basis specifying the knots.

        >>> bss = BSpline(knots=[0, 0.2, 0.4, 0.6, 0.8, 1])

        Once we create a basis we can evaluate each of its functions at a
        set of points.

        >>> bss = BSpline(nbasis=3, order=3)
        >>> bss.evaluate([0, 0.5, 1])
        array([[1.  , 0.25, 0.  ],
               [0.  , 0.5 , 0.  ],
               [0.  , 0.25, 1.  ]])

        And evaluates first derivative

        >>> bss.evaluate([0, 0.5, 1], derivative=1)
        array([[-2., -1.,  0.],
               [ 2.,  0., -2.],
               [ 0.,  1.,  2.]])

    References:
        .. [RS05] Ramsay, J., Silverman, B. W. (2005). *Functional Data
            Analysis*. Springler. 50-51.

    """

    def __init__(self, domain_range=None, nbasis=None, order=4, knots=None):
        """BSpline basis constructor.

        Args:
            domain_range (tuple, optional): Definition of the interval where
                the basis defines a space. Defaults to (0,1) if knots are not
                specified. If knots are specified defaults to the first and
                last element of the knots.
            nbasis (int, optional): Number of splines that form the basis.
            order (int, optional): Order of the splines. One greater that
                their degree. Defaults to 4 which mean cubic splines.
            knots (array_like): List of knots of the splines. If domain_range
                is specified the first and last elements of the knots have to
                match with it.
        """
        # Knots default to equally space points in the domain_range
        if knots is None:
            if nbasis is None:
                raise ValueError("Must provide either a list of knots or the"
                                 "number of basis.")
            if domain_range is None:
                domain_range = (0, 1)
            knots = list(numpy.linspace(domain_range[0], domain_range[1],
                                        nbasis - order + 2))
        else:
            knots = list(knots)
            knots.sort()

        # nbasis default to number of knots + order of the splines - 2
        if nbasis is None:
            nbasis = len(knots) + order - 2
            if domain_range is None:
                domain_range = (knots[0], knots[-1])

        if domain_range[0] != knots[0] or domain_range[1] != knots[-1]:
            raise ValueError("The ends of the knots must be the same as "
                             "the domain_range.")

        # Checks
        if nbasis != order + len(knots) - 2:
            raise ValueError("The number of basis has to equal the order "
                             "plus the number of knots minus 2.")

        self.order = order
        self.knots = list(knots)
        super().__init__(domain_range, nbasis)

    def _compute_matrix(self, eval_points, derivative=0):
        """Computes the basis or its derivatives given a list of values.

        It uses the scipy implementation of BSplines to compute the values
        for each element of the basis.

        Args:
            eval_points (array_like): List of points where the basis system is
                evaluated.
            derivative (int, optional): Order of the derivative. Defaults to 0.

        Returns:
            (:obj:`numpy.darray`): Matrix whose rows are the values of the each
            basis function or its derivatives at the values specified in
            eval_points.

        Implementation details: In order to allow a discontinuous behaviour at
        the boundaries of the domain it is necessary to placing m knots at the
        boundaries [RS05]_. This is automatically done so that the user only
        has to specify a single knot at the boundaries.

        References:
            .. [RS05] Ramsay, J., Silverman, B. W. (2005). *Functional Data
                Analysis*. Springler. 50-51.

        """
        # Places m knots at the boundaries
        knots = numpy.array([self.knots[0]] * (self.order - 1) + self.knots
                            + [self.knots[-1]] * (self.order - 1))
        # c is used the select which spline the function splev below computes
        c = numpy.zeros(len(knots))

        # Initialise empty matrix
        mat = numpy.empty((self.nbasis, len(eval_points)))

        # For each basis computes its value for each evaluation point
        for i in range(self.nbasis):
            # write a 1 in c in the position of the spline calculated in each
            # iteration
            c[i] = 1
            # compute the spline
            mat[i] = scipy.interpolate.splev(eval_points, (knots, c,
                                                           self.order - 1),
                                             der=derivative)
            c[i] = 0

        return mat

    def penalty(self, derivative_degree=None, coefficients=None):
        r""" Returns a penalty matrix given a differential operator.

        The differential operator can be either a derivative of a certain
        degree or a more complex operator.

        The penalty matrix is defined as [RS05-5-6-2]_:

        .. math::
            R_{ij} = \int L\phi_i(s) L\phi_j(s) ds

        where :math:`\phi_i(s)` for :math:`i=1, 2, ..., n` are the basis
        functions and :math:`L` is a differential operator.

        Args:
            derivative_degree (int): Integer indicating the order of the
                derivative or . For instance 2 means that the differential
                operator is :math:`f''(x)`.
            coefficients (list): List of coefficients representing a
                differential operator. An iterable indicating
                coefficients of derivatives (which can be functions). For
                instance the tuple (1, 0, numpy.sin) means :math:`1
                + sin(x)D^{2}`. Only used if derivative degree is None.

        Returns:
            numpy.array: Penalty matrix.

        References:
            .. [RS05-5-6-2] Ramsay, J., Silverman, B. W. (2005). Specifying the
                roughness penalty. In *Functional Data Analysis* (pp. 106-107).
                Springler.

        """
        if derivative_degree is not None:
            if derivative_degree >= self.order:
                raise ValueError("Penalty matrix cannot be evaluated for "
                                 "derivative of order {} for B-splines of "
                                 "order {}".format(derivative_degree,
                                                   self.order))
            if derivative_degree == self.order - 1:
                # The derivative of the bsplines are constant in the intervals
                # defined between knots
                knots = numpy.array(self.knots)
                mid_inter = (knots[1:] + knots[:-1]) / 2
                constants = self.evaluate(mid_inter,
                                          derivative=derivative_degree).T
                knots_intervals = numpy.diff(self.knots)
                # Integration of product of constants
                return constants.T @ numpy.diag(knots_intervals) @ constants

            if numpy.all(numpy.diff(self.knots) != 0):
                # Compute exactly using the piecewise polynomial
                # representation of splines

                # Places m knots at the boundaries
                knots = numpy.array(
                    [self.knots[0]] * (self.order - 1) + self.knots
                    + [self.knots[-1]] * (self.order - 1))
                # c is used the select which spline the function
                # PPoly.from_spline below computes
                c = numpy.zeros(len(knots))

                # Initialise empty list to store the piecewise polynomials
                ppoly_lst = []

                no_0_intervals = numpy.where(numpy.diff(knots) > 0)[0]

                # For each basis gets its piecewise polynomial representation
                for i in range(self.nbasis):
                    # write a 1 in c in the position of the spline
                    # transformed in each iteration
                    c[i] = 1
                    # gets the piecewise polynomial representation and gets
                    # only the positions for no zero length intervals
                    # This polynomial are defined relatively to the knots
                    # meaning that the column i corresponds to the ith knot.
                    # Let the ith not be a
                    # Then f(x) = pp(x - a)
                    pp = (PPoly.from_spline((knots, c, self.order - 1))
                          .c[:, no_0_intervals])
                    # We need the actual coefficients of f, not pp. So we
                    # just recursively calculate the new coefficients
                    coeffs = pp.copy()
                    for j in range(self.order - 1):
                        coeffs[j + 1:] += (
                                (binom(self.order - j - 1,
                                       range(1, self.order - j))
                                 * numpy.vstack(((-a) ** numpy.array(
                                            range(1, self.order - j)) for a in
                                                 self.knots[:-1]))
                                 ).T * pp[j])
                    ppoly_lst.append(coeffs)
                    c[i] = 0

                # Now for each pair of basis computes the inner product after
                # applying the linear differential operator
                penalty_matrix = numpy.zeros((self.nbasis, self.nbasis))
                for interval in range(len(no_0_intervals)):
                    for i in range(self.nbasis):
                        poly_i = numpy.trim_zeros(ppoly_lst[i][:,
                                                  interval], 'f')
                        if len(poly_i) <= derivative_degree:
                            # if the order of the polynomial is lesser or
                            # equal to the derivative the result of the
                            # integral will be 0
                            continue
                        # indefinite integral
                        integral = polyint(_polypow(polyder(
                            poly_i, derivative_degree), 2))
                        # definite integral
                        penalty_matrix[i, i] += numpy.diff(polyval(
                            integral, self.knots[interval: interval + 2]))[0]

                        for j in range(i + 1, self.nbasis):
                            poly_j = numpy.trim_zeros(ppoly_lst[j][:,
                                                      interval], 'f')
                            if len(poly_j) <= derivative_degree:
                                # if the order of the polynomial is lesser
                                # or equal to the derivative the result of
                                # the integral will be 0
                                continue
                                # indefinite integral
                            integral = polyint(
                                polymul(polyder(poly_i, derivative_degree),
                                        polyder(poly_j, derivative_degree)))
                            # definite integral
                            penalty_matrix[i, j] += numpy.diff(polyval(
                                integral, self.knots[interval: interval + 2])
                            )[0]
                            penalty_matrix[j, i] = penalty_matrix[i, j]
                return penalty_matrix
        else:
            # if the order of the derivative is greater or equal to the order
            # of the bspline minus 1
            if len(coefficients) >= self.order:
                raise ValueError("Penalty matrix cannot be evaluated for "
                                 "derivative of order {} for B-splines of "
                                 "order {}"
                                 .format(len(coefficients) - 1,
                                         self.order))

        # compute using the inner product
        return self._numerical_penalty(coefficients)

    def __repr__(self):
        return ("{}(domain_range={}, nbasis={}, order={}, knots={})".format(
            self.__class__.__name__, self.domain_range, self.nbasis, self.order,
            self.knots))


class Fourier(Basis):
    r"""Fourier basis.

    Defines a functional basis for representing functions on a fourier
    series expansion of period :math:`T`. The number of basis is always odd.
    If instantiated with an even number of basis, they will be incremented
    automatically by one.

    .. math::
        \phi_0(t) = \frac{1}{\sqrt{2}}

    .. math::
        \phi_{2n -1}(t) = sin\left(\frac{2 \pi n}{T} t\right)

    .. math::
        \phi_{2n}(t) = cos\left(\frac{2 \pi n}{T} t\right)

    Actually this basis functions are not orthogonal but not orthonormal. To
    achieve this they are divided by its norm: :math:`\sqrt{\frac{T}{2}}`.

    Attributes:
        domain_range (tuple): A tuple of length 2 containing the initial and
            end values of the interval over which the basis can be evaluated.
        nbasis (int): Number of functions in the basis.
        period (int or float): Period (:math:`T`).

    Examples:
        Constructs specifying number of basis, definition interval and period.

        >>> fb = Fourier([0, numpy.pi], nbasis=3, period=1)
        >>> fb.evaluate([0, numpy.pi / 4, numpy.pi / 2, numpy.pi]).round(2)
        array([[ 1.  ,  1.  ,  1.  ,  1.  ],
               [ 1.41,  0.31, -1.28,  0.89],
               [ 0.  , -1.38, -0.61,  1.1 ]])

        And evaluate second derivative

        >>> fb.evaluate([0, numpy.pi / 4, numpy.pi / 2, numpy.pi],
        ...             derivative = 2).round(2)
        array([[  0.  ,   0.  ,   0.  ,   0.  ],
               [-55.83, -12.32,  50.4 , -35.16],
               [ -0.  ,  54.46,  24.02, -43.37]])



    """

    def __init__(self, domain_range=(0, 1), nbasis=3, period=1):
        self.period = period
        # If number of basis is even, add 1
        nbasis += 1 - nbasis % 2
        super().__init__(domain_range, nbasis)

    def _compute_matrix(self, eval_points, derivative=0):
        """Computes the basis or its derivatives given a list of values.

        Args:
            eval_points (array_like): List of points where the basis is
                evaluated.
            derivative (int, optional): Order of the derivative. Defaults to 0.

        Returns:
            (:obj:`numpy.darray`): Matrix whose rows are the values of the each
            basis function or its derivatives at the values specified in
            eval_points.

        """
        if derivative < 0:
            raise ValueError("derivative only takes non-negative values.")

        omega = 2 * numpy.pi / self.period
        omega_t = omega * eval_points
        nbasis = self.nbasis if self.nbasis % 2 != 0 else self.nbasis + 1

        # Initialise empty matrix
        mat = numpy.empty((self.nbasis, len(eval_points)))
        if derivative == 0:
            # First base function is a constant
            # The division by numpy.sqrt(2) is so that it has the same norm as
            # the sine and cosine: sqrt(period / 2)
            mat[0] = numpy.ones(len(eval_points)) / numpy.sqrt(2)
            if nbasis > 1:
                # 2*pi*n*x / period
                args = numpy.outer(range(1, nbasis // 2 + 1), omega_t)
                index = range(2, nbasis, 2)
                # even indexes are sine functions
                mat[index] = numpy.sin(args)
                index = range(1, nbasis - 1, 2)
                # odd indexes are cosine functions
                mat[index] = numpy.cos(args)
        # evaluates the derivatives
        else:
            # First base function is a constant, so its derivative is 0.
            mat[0] = numpy.zeros(len(eval_points))
            if nbasis > 1:
                # (2*pi*n / period) ^ n_derivative
                factor = numpy.outer(
                    (-1) ** (derivative // 2)
                    * (numpy.array(range(1, nbasis // 2 + 1)) * omega)
                    ** derivative,
                    numpy.ones(len(eval_points)))
                # 2*pi*n*x / period
                args = numpy.outer(range(1, nbasis // 2 + 1), omega_t)
                # even indexes
                index_e = range(2, nbasis, 2)
                # odd indexes
                index_o = range(1, nbasis - 1, 2)
                if derivative % 2 == 0:
                    mat[index_e] = factor * numpy.sin(args)
                    mat[index_o] = factor * numpy.cos(args)
                else:
                    mat[index_e] = factor * numpy.cos(args)
                    mat[index_o] = -factor * numpy.sin(args)

        # normalise
        mat = mat / numpy.sqrt(self.period / 2)
        return mat

    def penalty(self, derivative_degree=None, coefficients=None):
        r""" Returns a penalty matrix given a differential operator.

        The differential operator can be either a derivative of a certain
        degree or a more complex operator.

        The penalty matrix is defined as [RS05-5-6-2]_:

        .. math::
            R_{ij} = \int L\phi_i(s) L\phi_j(s) ds

        where :math:`\phi_i(s)` for :math:`i=1, 2, ..., n` are the basis
        functions and :math:`L` is a differential operator.

        Args:
            derivative_degree (int): Integer indicating the order of the
                derivative or . For instance 2 means that the differential
                operator is :math:`f''(x)`.
            coefficients (list): List of coefficients representing a
                differential operator. An iterable indicating
                coefficients of derivatives (which can be functions). For
                instance the tuple (1, 0, numpy.sin) means :math:`1
                + sin(x)D^{2}`. Only used if derivative degree is None.

        Returns:
            numpy.array: Penalty matrix.

        References:
            .. [RS05-5-6-2] Ramsay, J., Silverman, B. W. (2005). Specifying the
                roughness penalty. In *Functional Data Analysis* (pp. 106-107).
                Springler.

        """
        if isinstance(derivative_degree, int):
            omega = 2 * numpy.pi / self.period
            # the derivatives of the functions of the basis are also orthogonal
            # so only the diagonal is different from 0.
            penalty_matrix = numpy.zeros(self.nbasis)
            if derivative_degree == 0:
                penalty_matrix[0] = 1
            else:
                # the derivative of a constant is 0
                # the first basis function is a constant
                penalty_matrix[0] = 0
            index_even = numpy.array(range(2, self.nbasis, 2))
            exponents = index_even / 2
            # factor resulting of deriving the basis function the times
            # indcated in the derivative_degree
            factor = (exponents * omega) ** (2 * derivative_degree)
            # the norm of the basis functions is 1 so only the result of the
            # integral is just the factor
            penalty_matrix[index_even - 1] = factor
            penalty_matrix[index_even] = factor
            return numpy.diag(penalty_matrix)
        else:
            # implement using inner product
            return self._numerical_penalty(coefficients)

    def __repr__(self):
        return ("{}(domain_range={}, nbasis={}, period={})".format(
            self.__class__.__name__, self.domain_range, self.nbasis,
            self.period))


class FDataBasis:
    r"""Basis representation of functional data.

    Class representation for functional data in the form of a set of basis
    functions multplied by a set of coefficients.

    .. math::
        f(x) = \sum_{k=1}{K}c_k\phi_k

    Where n is the number of basis functions, :math:`c = (c_1, c_2, ...,
    c_K)` the vector of coefficients and  :math:`\phi = (\phi_1, \phi_2,
    ..., \phi_K)` the basis function system.

    Attributes:
        basis (:obj:`Basis`): Basis function system.
        coefficients (numpy.darray): List or matrix of coefficients. Has to
            have the same length or number of columns as the number of basis
            function in the basis. If a matrix, each row contains the
            coefficients that multiplied by the basis functions produce each
            functional datum.

    Examples:
        >>> basis = Monomial(nbasis=4)
        >>> coefficients = [1, 1, 3, .5]
        >>> FDataBasis(basis, coefficients)
        FDataBasis(basis=Monomial(...), coefficients=[[1.  1.  3.  0.5]])

    """

    def __init__(self, basis, coefficients):
        """Constructor of FDataBasis.

        Args:
            basis (:obj:`Basis`): Basis function system.
            coefficients (array_like): List or matrix of coefficients. Has to
                have the same length or number of columns as the number of
                basis function in the basis.
        """
        coefficients = numpy.atleast_2d(coefficients)
        if coefficients.shape[1] != basis.nbasis:
            raise ValueError("The length or number of columns of coefficients "
                             "has to be the same equal to the number of "
                             "elements of the basis.")
        self.basis = basis
        self.coefficients = coefficients

    @classmethod
    def from_data(cls, data_matrix, sample_points, basis, weight_matrix=None,
                  smoothness_parameter=0, differential_operator=2,
                  penalty_matrix=None, method='cholesky'):
        r"""Raw data to a smooth functional form.

        Takes functional data in a discrete form and makes an approximates it
        to the closest function that can be generated by the basis.a

        The fit is made so as to reduce the penalized sum of squared errors
        [RS05-5-2-5]_:
        .. math::
            PENSSE(c) = (y - \Phi c)' W (y - \Phi c) + \lambda c'Rc

        where :math:`y` is the vector or matrix of observations, :math:`\Phi`
        the matrix whose columns are the basis functions evaluated at the
        sampling points, :math:`c` the coefficient vector or matrix to be
        estimated, :math:`\lambda` a smoothness paramerter and :math:`c'Rc` the
        matrix representation of the roughness penalty :math:`\int \left[ L(
        x(s)) \right] ^2 ds` where :math:`L` is a linear differential operator.

        Each element of :math:`R` has the following close form:
        .. math::
            R_{ij} = \int L\phi_i(s) L\phi_j(s) ds

        Args:
            data_matrix (array_like): List or matrix containing the
                observations. If a matrix each row represents a single
                functional datum and the columns the different observations.
            sample_points (array_like): Values of the domain where the previous
                data were taken.
            basis: (Basis): Basis used.
            weight_matrix (array_like, optional): Matrix to weight the
                observations. Defaults to the identity matrix.
            smoothness_parameter (int or float, optional): Smoothness parameter.
                Trying with several factors in a logarythm scale is suggested.
                If 0 no smoothing is performed. Defaults to 0.
            differential_operator (int or list or tuple, optional): Integer or
                list of coefficients representing a differential operator. A
                differential operator can be either a integer (indicating the
                order of the derivative) or an iterable indicating
                coefficients of derivatives (which can be functions). For
                instance 2 means that the differential operator is
                :math:`f''(x)` and the tuple (1, 0, numpy.sin), :math:`1
                + sin(x)D^{2}`. Defaults to 2, meaning the second derivative.
            penalty_matrix (array_like, optional): Penalty matrix. If
                supplied the differential operator is not used and instead
                the matrix supplied by this argument is used.
            method (str): Algorithm used for calculating the coefficients using
                the least squares method. The values admitted are 'cholesky'
                and 'qr' for Cholesky and QR factorisation methods
                respectively.

        Returns:
            FDataBasis: Represention of the data in a functional form as
                product of coefficients by basis functions.

        Examples:
            >>> import numpy as np
            >>> t = np.linspace(0, 1, 5)
            >>> x = np.sin(2 * np.pi * t) + np.cos(2 * np.pi * t)
            >>> x
            array([ 1.,  1., -1., -1.,  1.])

            >>> basis = Fourier((0, 1), nbasis=3)
            >>> fd = FDataBasis.from_data(x, t, basis)
            >>> fd.coefficients.round(2)
            array([[0.  , 0.71, 0.71]])

        References:
            .. [RS05-5-2-5] Ramsay, J., Silverman, B. W. (2005). How spline
                smooths are computed. In *Functional Data Analysis* (pp. 86-87).
                Springler.

        """

        # n is the samples
        # m is the observations
        # k is the number of elements of the basis

        # Each sample in a column (m x n)
        data_matrix = numpy.atleast_2d(data_matrix).T

        # Each basis in a column
        basis_values = basis.evaluate(sample_points).T

        # If no weight matrix is given all the weights are one
        if not weight_matrix:
            weight_matrix = numpy.identity(basis_values.shape[0])

        # We need to solve the equation
        # (phi' W phi + lambda * R) C = phi' W Y
        # where:
        #  phi is the basis_values
        #  W is the weight matrix
        #  lambda the smoothness parameter
        #  C the coefficient matrix (the unknown)
        #  Y is the data_matrix

        if data_matrix.shape[0] > basis.nbasis or smoothness_parameter > 0:
            method = method.lower()
            if method == 'cholesky':
                right_matrix = basis_values.T @ weight_matrix @ data_matrix
                left_matrix = basis_values.T @ weight_matrix @ basis_values

                # Adds the roughness penalty to the equation
                if smoothness_parameter > 0:
                    if not penalty_matrix:
                        penalty_matrix = basis.penalty(differential_operator)
                    left_matrix += smoothness_parameter * penalty_matrix

                coefficients = scipy.linalg.cho_solve(scipy.linalg.cho_factor(
                    left_matrix, lower=True), right_matrix)

                # The ith column is the coefficients of the ith basis for each
                #  sample
                coefficients = coefficients.T

            elif method == 'qr':
                if smoothness_parameter > 0 or weight_matrix:
                    raise NotImplementedError('QR method not implemented for '
                                              'rougness penalty or weight '
                                              'matrices')
                # Resolves the equation
                # B.T @ B @ C = B.T @ D
                # by means of the QR decomposition

                # B = Q @ R
                q, r = scipy.linalg.qr(basis_values)
                right_matrix = q.T @ data_matrix

                # R @ C = Q.T @ D
                coefficients = numpy.linalg.solve(r, right_matrix)
                # The ith column is the coefficients of the ith basis for each
                # sample
                coefficients = coefficients.T

            else:
                raise ValueError("Unknown method.")

        elif data_matrix.shape[0] == basis.nbasis:
            # If the number of basis equals the number of points and no
            # smoothing is required
            coefficients = numpy.linalg.solve(basis_values, data_matrix)

        else:  # data_matrix.shape[0] < basis.nbasis
            raise ValueError("The number of basis functions ({}) exceed the "
                             "number of points to be smoothed ({})."
                             .format(basis.nbasis, data_matrix.shape[0]))

        return cls(basis, coefficients)

    @property
    def nsamples(self):
        """Number of samples"""
        return self.coefficients.shape[0]

    @property
    def nbasis(self):
        """Number of basis"""
        return self.basis.nbasis

    @property
    def domain_range(self):
        """Definition range"""
        return self.basis.domain_range

    def evaluate(self, eval_points, derivative=0):
        """Evaluates the object or its derivatives at a list of values.

        Args:
            eval_points (array_like): List of points where the functions are
                evaluated.
            derivative (int, optional): Order of the derivative. Defaults to 0.

        Returns:
            (numpy.darray): Matrix whose rows are the values of the each
            function at the values specified in eval_points.

        """
        # each column is the values of one element of the basis
        basis_values = self.basis.evaluate(eval_points, derivative).T

        res_matrix = numpy.empty((self.nsamples, len(eval_points)))

        for i in range(self.nsamples):
            _matrix = basis_values * self.coefficients[i]
            res_matrix[i] = _matrix.sum(axis=1)

        return res_matrix

    def plot(self, ax=None, derivative=0, **kwargs):
        """Plots the FDataBasis object or its derivatives.

        Args:
            ax (axis object, optional): axis over with the graphs are plotted.
                Defaults to matplotlib current axis.
            derivative (int, optional): Order of the derivative. Defaults to 0.
            **kwargs: keyword arguments to be passed to the
                matplotlib.pyplot.plot function.

        Returns:
            List of lines that were added to the plot.

        """
        if ax is None:
            ax = matplotlib.pyplot.gca()
        npoints = max(501, 10 * self.nbasis)
        # List of points where the basis are evaluated
        eval_points = numpy.linspace(self.domain_range[0],
                                     self.domain_range[1],
                                     npoints)
        # Basis evaluated in the previous list of points
        mat = self.evaluate(eval_points, derivative)
        # Plot
        return ax.plot(eval_points, mat.T, **kwargs)

    def mean(self):
        """ Computes the mean of all the samples in a FDataBasis object.

        Returns:
            :obj:`FDataBasis`: A FDataBais object with just one sample
            representing the mean of all the samples in the original
            FDataBasis object.

        Examples:
            >>> basis = Monomial(nbasis=4)
            >>> coefficients = [[0.5, 1, 2, .5], [1.5, 1, 4, .5]]
            >>> FDataBasis(basis, coefficients).mean()
            FDataBasis(basis=..., coefficients=[[1.  1.  3.  0.5]])

        """
        return FDataBasis(self.basis, numpy.mean(self.coefficients, axis=0))

    def gmean(self, eval_points=None):
        """Computes the geometric mean of the functional data object.

        A numerical approach its used. The object its transformed into its
        discrete representation and then the geometric mean is computed and
        then the object is taken back to the basis representation.

        Args:
            eval_points (array_like, optional): Set of points where the
                functions are evaluated to obtain the discrete
                representation of the object. If none are passed it calls
                numpy.linspace with bounds equal to the ones defined in
                self.domain_range and the number of points the maximum
                between 501 and 10 times the number of basis.

        Returns:
            FDataBasis: Geometric mean of the original object.
        """
        return self.to_grid(eval_points).gmean().to_basis(self.basis)

    def to_grid(self, eval_points=None):
        """Returns the discrete representation of the object.

        Args:
            eval_points (array_like, optional): Set of points where the
                functions are evaluated. If none are passed it calls
                numpy.linspace with bounds equal to the ones defined in
                self.domain_range and the number of points the maximum
                between 501 and 10 times the number of basis.

        Returns:
              FDataGrid: Discrete representation of the functional data
              object.

        Examples:
            >>> fd = FDataBasis(coefficients=[[1, 1, 1], [1, 0, 1]],
            ...                 basis=Monomial((0,5), nbasis=3))
            >>> fd.to_grid([0, 1, 2])
            FDataGrid(
                array([[1., 3., 7.],
                       [1., 2., 5.]]),
                sample_points=[array([0, 1, 2])],
                sample_range=array([[0, 5]]),
                dataset_label='Data set',
                axes_labels=None)

        """
        if eval_points is None:
            npoints = max(501, 10 * self.nbasis)
            numpy.linspace(self.domain_range[0],
                           self.domain_range[1],
                           npoints)

        return grid.FDataGrid(self.evaluate(eval_points),
                              sample_points=eval_points,
                              sample_range=self.domain_range)

    def __repr__(self):
        return "{}(basis={}, coefficients={})".format(
            self.__class__.__name__, self.basis, self.coefficients)

    def __call__(self, eval_points):
        """Evaluates the functions in the object at a list of values.

        Args:
            eval_points (array_like): List of points where the functions are
                evaluated.

        Returns:
            (numpy.darray): Matrix whose rows are the values of the each
            function at the values specified in eval_points.

        """
        return self.evaluate(eval_points)
