"""Module for functional data manipulation in a basis system.

Defines functional data object in a basis function system representation and
the corresponding basis classes.

"""
from abc import ABC, abstractmethod
import copy
import scipy.signal

from numpy import polyder, polyint, polymul, polyval
import scipy.integrate
from scipy.interpolate import BSpline as SciBSpline
from scipy.interpolate import PPoly
import scipy.interpolate
from scipy.special import binom
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted

import numpy as np

from .._utils import _list_of_arrays
from ._fdatabasis import FDataBasis, FDataBasisDType


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


def _check_domain(domain_range):
    for domain in domain_range:
        if len(domain) != 2 or domain[0] >= domain[1]:
            raise ValueError(f"The interval {domain} is not well-defined.")


def _same_domain(one_domain_range, other_domain_range):
    return np.array_equal(one_domain_range, other_domain_range)


class Basis(ABC):
    """Defines the structure of a basis function system.

    Attributes:
        domain_range (tuple): a tuple of length 2 containing the initial and
            end values of the interval over which the basis can be evaluated.
        n_basis (int): number of functions in the basis.

    """

    def __init__(self, domain_range=None, n_basis=1):
        """Basis constructor.

        Args:
            domain_range (tuple or list of tuples, optional): Definition of the
                interval where the basis defines a space. Defaults to (0,1).
            n_basis: Number of functions that form the basis. Defaults to 1.
        """

        if domain_range is not None:
            # TODO: Allow multiple dimensions
            domain_range = _list_of_arrays(domain_range)

            # Some checks
            _check_domain(domain_range)

        if n_basis < 1:
            raise ValueError("The number of basis has to be strictly "
                             "possitive.")

        self._domain_range = domain_range
        self.n_basis = n_basis
        self._drop_index_lst = []

        super().__init__()

    @property
    def domain_range(self):
        if self._domain_range is None:
            return [np.array([0, 1])]
        else:
            return self._domain_range

    @domain_range.setter
    def domain_range(self, value):
        self._domain_range = value

    @abstractmethod
    def _evaluate(self, eval_points, derivative=0):
        """Subclasses must override this to provide basis evaluation."""
        pass

    @abstractmethod
    def _derivative(self, coefs, order=1):
        pass

    def evaluate(self, eval_points, derivative=0):
        """Evaluate Basis objects and its derivatives.

        Evaluates the basis function system or its derivatives at a list of
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
        if derivative < 0:
            raise ValueError("derivative only takes non-negative values.")

        eval_points = np.asarray(eval_points)
        if np.any(np.isnan(eval_points)):
            raise ValueError("The list of points where the function is "
                             "evaluated can not contain nan values.")

        return self._evaluate(eval_points, derivative)

    def __call__(self, *args, **kwargs):
        return self.evaluate(*args, **kwargs)

    def plot(self, chart=None, *, derivative=0, **kwargs):
        """Plot the basis object or its derivatives.

        Args:
            chart (figure object, axe or list of axes, optional): figure over
                with the graphs are plotted or axis over where the graphs are
                plotted.
            derivative (int or tuple, optional): Order of derivative to be
                plotted. Defaults 0.
            **kwargs: keyword arguments to be passed to the
                fdata.plot function.

        Returns:
            fig (figure): figure object in which the graphs are plotted.

        """
        self.to_basis().plot(chart=chart, derivative=derivative, **kwargs)

    def _numerical_penalty(self, lfd):
        """Return a penalty matrix using a numerical approach.

        See :func:`~basis.Basis.penalty`.

        Args:
            lfd (LinearDifferentialOperator, list or int): Linear
            differential operator. If it is not a LinearDifferentialOperator
            object, it will be converted to one.
        """
        from skfda.misc import LinearDifferentialOperator

        if not isinstance(lfd, LinearDifferentialOperator):
            lfd = LinearDifferentialOperator(lfd)

        indices = np.triu_indices(self.n_basis)

        def cross_product(x):
            """Multiply the two lfds"""
            res = lfd(self)([x])[:, 0]

            return res[indices[0]] * res[indices[1]]

        # Range of first dimension
        domain_range = self.domain_range[0]

        penalty_matrix = np.empty((self.n_basis, self.n_basis))

        # Obtain the integrals for the upper matrix
        triang_vec = scipy.integrate.quad_vec(
            cross_product, domain_range[0], domain_range[1])[0]

        # Set upper matrix
        penalty_matrix[indices] = triang_vec

        # Set lower matrix
        penalty_matrix[(indices[1], indices[0])] = triang_vec

        return penalty_matrix

    def _penalty(self, lfd):
        """
        Subclasses may override this for computing analytically
        the penalty matrix in the cases when that is possible.

        Returning NotImplemented will use numerical computation
        of the penalty matrix.
        """
        return NotImplemented

    def penalty(self, lfd):
        r"""Return a penalty matrix given a differential operator.

        The differential operator can be either a derivative of a certain
        degree or a more complex operator.

        The penalty matrix is defined as [RS05-5-6-2]_:

        .. math::
            R_{ij} = \int L\phi_i(s) L\phi_j(s) ds

        where :math:`\phi_i(s)` for :math:`i=1, 2, ..., n` are the basis
        functions and :math:`L` is a differential operator.

        Args:
            lfd (LinearDifferentialOperator, list or int): Linear
            differential operator. If it is not a LinearDifferentialOperator
            object, it will be converted to one.

        Returns:
            numpy.array: Penalty matrix.

        References:
            .. [RS05-5-6-2] Ramsay, J., Silverman, B. W. (2005). Specifying the
               roughness penalty. In *Functional Data Analysis* (pp. 106-107).
               Springer.

        """
        from skfda.misc import LinearDifferentialOperator

        if not isinstance(lfd, LinearDifferentialOperator):
            lfd = LinearDifferentialOperator(lfd)

        matrix = self._penalty(lfd)

        if matrix is NotImplemented:
            return self._numerical_penalty(lfd)
        else:
            return matrix

    @abstractmethod
    def basis_of_product(self, other):
        pass

    @abstractmethod
    def rbasis_of_product(self, other):
        pass

    @staticmethod
    def default_basis_of_product(one, other):
        """Default multiplication for a pair of basis"""
        if not _same_domain(one.domain_range, other.domain_range):
            raise ValueError("Ranges are not equal.")

        norder = min(8, one.n_basis + other.n_basis)
        n_basis = max(one.n_basis + other.n_basis, norder + 1)
        return BSpline(one.domain_range, n_basis, norder)

    def rescale(self, domain_range=None):
        r"""Return a copy of the basis with a new domain range, with the
            corresponding values rescaled to the new bounds.

            Args:
                domain_range (tuple, optional): Definition of the interval
                    where the basis defines a space. Defaults uses the same as
                    the original basis.
        """

        if domain_range is None:
            domain_range = self.domain_range

        return type(self)(domain_range, self.n_basis)

    def same_domain(self, other):
        r"""Returns if two basis are defined on the same domain range.

            Args:
                other (Basis): Basis to check the domain range definition
        """
        return _same_domain(self.domain_range, other.domain_range)

    def copy(self):
        """Basis copy"""
        return copy.deepcopy(self)

    def to_basis(self):
        return FDataBasis(self.copy(), np.identity(self.n_basis))

    def _list_to_R(self, knots):
        retstring = "c("
        for i in range(0, len(knots)):
            retstring = retstring + str(knots[i]) + ", "
        return retstring[0:len(retstring) - 2] + ")"

    def _to_R(self):
        raise NotImplementedError

    def _inner_matrix(self, other=None):
        r"""Return the Inner Product Matrix of a pair of basis.

        The Inner Product Matrix is defined as

        .. math::
            IP_{ij} = \langle\phi_i, \theta_j\rangle

        where :math:`\phi_i` is the ith element of the basi and
        :math:`\theta_j` is the jth element of the second basis.
        This matrix helps on the calculation of the inner product
        between objects on two basis and for the change of basis.

        Args:
            other (:class:`Basis`): Basis to compute the inner product
            matrix. If not basis is given, it computes the matrix with
            itself returning the Gram Matrix

        Returns:
            numpy.array: Inner Product Matrix of two basis

        """
        if other is None or self == other:
            return self.gram_matrix()

        first = self.to_basis()
        second = other.to_basis()

        inner = np.zeros((self.n_basis, other.n_basis))

        for i in range(self.n_basis):
            for j in range(other.n_basis):
                inner[i, j] = first[i].inner_product(second[j], None, None)

        return inner

    def gram_matrix(self):
        r"""Return the Gram Matrix of a basis

        The Gram Matrix is defined as

        .. math::
            G_{ij} = \langle\phi_i, \phi_j\rangle

        where :math:`\phi_i` is the ith element of the basis. This is a
        symmetric matrix and positive-semidefinite.

        Returns:
            numpy.array: Gram Matrix of the basis.

        """
        fbasis = self.to_basis()

        gram = np.zeros((self.n_basis, self.n_basis))

        for i in range(fbasis.n_basis):
            for j in range(i, fbasis.n_basis):
                gram[i, j] = fbasis[i].inner_product(fbasis[j], None, None)
                gram[j, i] = gram[i, j]

        return gram

    def inner_product(self, other):
        return np.transpose(other.inner_product(self.to_basis()))

    def _add_same_basis(self, coefs1, coefs2):
        return self.copy(), coefs1 + coefs2

    def _add_constant(self, coefs, constant):
        coefs = coefs.copy()
        constant = np.array(constant)
        coefs[:, 0] = coefs[:, 0] + constant

        return self.copy(), coefs

    def _sub_same_basis(self, coefs1, coefs2):
        return self.copy(), coefs1 - coefs2

    def _sub_constant(self, coefs, other):
        coefs = coefs.copy()
        other = np.array(other)
        coefs[:, 0] = coefs[:, 0] - other

        return self.copy(), coefs

    def _mul_constant(self, coefs, other):
        coefs = coefs.copy()
        other = np.atleast_2d(other).reshape(-1, 1)
        coefs = coefs * other

        return self.copy(), coefs

    def __repr__(self):
        """Representation of a Basis object."""
        return (f"{self.__class__.__name__}(domain_range={self.domain_range}, "
                f"n_basis={self.n_basis})")

    def __eq__(self, other):
        """Equality of Basis"""
        return (type(self) == type(other)
                and _same_domain(self.domain_range, other.domain_range)
                and self.n_basis == other.n_basis)


class Constant(Basis):
    """Constant basis.

    Basis for constant functions

    Attributes:
        domain_range (tuple): a tuple of length 2 containing the initial and
            end values of the interval over which the basis can be evaluated.

    Examples:
        Defines a contant base over the interval :math:`[0, 5]` consisting
        on the constant function 1 on :math:`[0, 5]`.

        >>> bs_cons = Constant((0,5))

    """

    def __init__(self, domain_range=None):
        """Constant basis constructor.

        Args:
            domain_range (tuple): Tuple defining the domain over which the
            function is defined.

        """
        super().__init__(domain_range, 1)

    def _evaluate(self, eval_points, derivative=0):
        return (np.ones((1, len(eval_points))) if derivative == 0
                else np.zeros((1, len(eval_points))))

    def _derivative(self, coefs, order=1):
        return (self.copy(), coefs.copy() if order == 0
                else self.copy(), np.zeros(coefs.shape))

    def _penalty(self, lfd):
        coefs = lfd.constant_weights()
        if coefs is None:
            return NotImplemented

        return np.array([[coefs[0] ** 2 *
                          (self.domain_range[0][1] -
                           self.domain_range[0][0])]])

    def basis_of_product(self, other):
        """Multiplication of a Constant Basis with other Basis"""
        if not _same_domain(self.domain_range, other.domain_range):
            raise ValueError("Ranges are not equal.")

        return other.copy()

    def rbasis_of_product(self, other):
        """Multiplication of a Constant Basis with other Basis"""
        return other.copy()

    def _to_R(self):
        drange = self.domain_range[0]
        return "create.constant.basis(rangeval = c(" + str(drange[0]) + "," +\
               str(drange[1]) + "))"


class Monomial(Basis):
    """Monomial basis.

    Basis formed by powers of the argument :math:`t`:

    .. math::
        1, t, t^2, t^3...

    Attributes:
        domain_range (tuple): a tuple of length 2 containing the initial and
            end values of the interval over which the basis can be evaluated.
        n_basis (int): number of functions in the basis.

    Examples:
        Defines a monomial base over the interval :math:`[0, 5]` consisting
        on the first 3 powers of :math:`t`: :math:`1, t, t^2`.

        >>> bs_mon = Monomial((0,5), n_basis=3)

        And evaluates all the functions in the basis in a list of descrete
        values.

        >>> bs_mon.evaluate([0, 1, 2])
        array([[1, 1, 1],
               [0, 1, 2],
               [0, 1, 4]])

        And also evaluates its derivatives

        >>> bs_mon.evaluate([0, 1, 2], derivative=1)
        array([[0, 0, 0],
               [1, 1, 1],
               [0, 2, 4]])
        >>> bs_mon.evaluate([0, 1, 2], derivative=2)
        array([[0, 0, 0],
               [0, 0, 0],
               [2, 2, 2]])

    """

    def _coef_mat(self, derivative):
        """
        Obtain the matrix of coefficients.

        Each column of coef_mat contains the numbers that must be multiplied
        together in order to obtain the coefficient of each basis function
        Thus, column i will contain i, i - 1, ..., i - derivative + 1.
        """

        seq = np.arange(self.n_basis)
        coef_mat = np.linspace(seq, seq - derivative + 1,
                               derivative, dtype=int)

        return seq, coef_mat

    def _coefs_exps_derivatives(self, derivative):
        """
        Return coefficients and exponents of the derivatives.

        This function is used for computing the basis functions and evaluate.

        When the exponent would be negative (the coefficient in that case
        is zero) returns 0 as the exponent (to prevent division by zero).
        """
        seq, coef_mat = self._coef_mat(derivative)
        coefs = np.prod(coef_mat, axis=0)

        exps = np.maximum(seq - derivative, 0)

        return coefs, exps

    def _evaluate(self, eval_points, derivative=0):

        coefs, exps = self._coefs_exps_derivatives(derivative)

        raised = np.power.outer(eval_points, exps)

        return (coefs * raised).T

    def _derivative(self, coefs, order=1):
        return (Monomial(self.domain_range, self.n_basis - order),
                np.array([np.polyder(x[::-1], order)[::-1]
                          for x in coefs]))

    def _evaluate_constant_lfd(self, weights):
        """
        Evaluate constant weights of a linear differential operator
        over the basis functions.
        """

        max_derivative = len(weights) - 1

        _, coef_mat = self._coef_mat(max_derivative)

        # Compute coefficients for each derivative
        coefs = np.cumprod(coef_mat, axis=0)

        # Add derivative 0 row
        coefs = np.concatenate((np.ones((1, self.n_basis)), coefs))

        # Now each row correspond to each basis and each column to
        # each derivative
        coefs_t = coefs.T

        # Multiply by the weights
        weighted_coefs = coefs_t * weights
        assert len(weighted_coefs) == self.n_basis

        # Now each row has the right weight, but the polynomials are in a
        # decreasing order and with different exponents

        # Resize the coefs so that there are as many rows as the number of
        # basis
        # The matrix is now triangular
        # refcheck is False to prevent exceptions while debugging
        weighted_coefs = np.copy(weighted_coefs.T)
        weighted_coefs.resize(self.n_basis,
                              self.n_basis, refcheck=False)
        weighted_coefs = weighted_coefs.T

        # Shift the coefficients so that they correspond to the right
        # exponent
        indexes = np.tril_indices(self.n_basis)
        polynomials = np.zeros_like(weighted_coefs)
        polynomials[indexes[0], indexes[1] -
                    indexes[0] - 1] = weighted_coefs[indexes]

        # At this point, each row of the matrix correspond to a polynomial
        # that is the result of applying the linear differential operator
        # to each element of the basis

        return polynomials

    def _penalty(self, lfd):

        weights = lfd.constant_weights()
        if weights is None:
            return NotImplemented

        polynomials = self._evaluate_constant_lfd(weights)

        # Expand the polinomials with 0, so that the multiplication fits
        # inside. It will need the double of the degree
        length_with_padding = polynomials.shape[1] * 2 - 1

        # Multiplication of polynomials is a convolution.
        # The convolution can be performed in parallel applying a Fourier
        # transform and then doing a normal multiplication in that
        # space, coverting back with the inverse Fourier transform
        fft = np.fft.rfft(polynomials, length_with_padding)

        # We compute only the upper matrix, as the penalty matrix is
        # symmetrical
        indices = np.triu_indices(self.n_basis)
        fft_mul = fft[indices[0]] * fft[indices[1]]

        integrand = np.fft.irfft(fft_mul, length_with_padding)

        integration_domain = self.domain_range[0]

        # To integrate, divide by the position and increase the exponent
        # in the evaluation
        denom = np.arange(integrand.shape[1], 0, -1)
        integrand /= denom

        # Add column of zeros at the right to increase exponent
        integrand = np.pad(integrand,
                           pad_width=((0, 0),
                                      (0, 1)),
                           mode='constant')

        # Now, apply Barrow's rule
        # polyval applies Horner method over the first dimension,
        # so we need to transpose
        x_right = np.polyval(integrand.T, integration_domain[1])
        x_left = np.polyval(integrand.T, integration_domain[0])

        integral = x_right - x_left

        penalty_matrix = np.empty((self.n_basis, self.n_basis))

        # Set upper matrix
        penalty_matrix[indices] = integral

        # Set lower matrix
        penalty_matrix[(indices[1], indices[0])] = integral

        return penalty_matrix

    def basis_of_product(self, other):
        """Multiplication of a Monomial Basis with other Basis"""
        if not _same_domain(self.domain_range, other.domain_range):
            raise ValueError("Ranges are not equal.")

        if isinstance(other, Monomial):
            return Monomial(self.domain_range, self.n_basis + other.n_basis)

        return other.rbasis_of_product(self)

    def rbasis_of_product(self, other):
        """Multiplication of a Monomial Basis with other Basis"""
        return Basis.default_basis_of_product(self, other)

    def _to_R(self):
        drange = self.domain_range[0]
        return "create.monomial.basis(rangeval = c(" + str(drange[0]) + "," +\
               str(drange[1]) + "), nbasis = " + str(self.n_basis) + ")"


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
        >>> bss.evaluate([0, 0.5, 1])
        array([[ 1.  ,  0.25,  0.  ],
               [ 0.  ,  0.5 ,  0.  ],
               [ 0.  ,  0.25,  1.  ]])

        And evaluates first derivative

        >>> bss.evaluate([0, 0.5, 1], derivative=1)
        array([[-2., -1.,  0.],
               [ 2.,  0., -2.],
               [ 0.,  1.,  2.]])

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
            domain_range = _list_of_arrays(domain_range)

            if len(domain_range) != 1:
                raise ValueError("Domain range should be unidimensional.")

            domain_range = domain_range[0]

        # Knots default to equally space points in the domain_range
        if knots is None:
            if n_basis is None:
                raise ValueError("Must provide either a list of knots or the"
                                 "number of basis.")
        else:
            knots = list(knots)
            knots.sort()
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

        self.order = order
        self.knots = None if knots is None else list(knots)
        super().__init__(domain_range, n_basis)

        # Checks
        if self.n_basis != self.order + len(self.knots) - 2:
            raise ValueError(f"The number of basis ({self.n_basis}) has to "
                             f"equal the order ({self.order}) plus the "
                             f"number of knots ({len(self.knots)}) minus 2.")

    @property
    def knots(self):
        if self._knots is None:
            return list(np.linspace(*self.domain_range[0],
                                    self.n_basis - self.order + 2))
        else:
            return self._knots

    @knots.setter
    def knots(self, value):
        self._knots = value

    def _evaluate(self, eval_points, derivative=0):
        """Compute the basis or its derivatives given a list of values.

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
                Analysis*. Springer. 50-51.

        """
        if derivative > (self.order - 1):
            return np.zeros((self.n_basis, len(eval_points)))

        # Places m knots at the boundaries
        knots = np.array([self.knots[0]] * (self.order - 1) + self.knots +
                         [self.knots[-1]] * (self.order - 1))
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
            mat[i] = scipy.interpolate.splev(eval_points, (knots, c,
                                                           self.order - 1),
                                             der=derivative)
            c[i] = 0

        return mat

    def _derivative(self, coefs, order=1):
        deriv_splines = [self._to_scipy_BSpline(coefs[i]).derivative(order)
                         for i in range(coefs.shape[0])]

        deriv_coefs = [BSpline._from_scipy_BSpline(spline)[1]
                       for spline in deriv_splines]

        deriv_basis = BSpline._from_scipy_BSpline(deriv_splines[0])[0]

        return deriv_basis, np.array(deriv_coefs)[:, 0:deriv_basis.n_basis]

    def _penalty(self, lfd):

        coefs = lfd.constant_weights()
        if coefs is None:
            return NotImplemented

        nonzero = np.flatnonzero(coefs)
        if len(nonzero) != 1:
            return NotImplemented

        derivative_degree = nonzero[0]

        if derivative_degree >= self.order:
            raise ValueError(f"Penalty matrix cannot be evaluated for "
                             f"derivative of order {derivative_degree} for"
                             f" B-splines of order {self.order}")
        if derivative_degree == self.order - 1:
            # The derivative of the bsplines are constant in the intervals
            # defined between knots
            knots = np.array(self.knots)
            mid_inter = (knots[1:] + knots[:-1]) / 2
            constants = self.evaluate(mid_inter,
                                      derivative=derivative_degree).T
            knots_intervals = np.diff(self.knots)
            # Integration of product of constants
            return constants.T @ np.diag(knots_intervals) @ constants

        if np.all(np.diff(self.knots) != 0):
            # Compute exactly using the piecewise polynomial
            # representation of splines

            # Places m knots at the boundaries
            knots = np.array(
                [self.knots[0]] * (self.order - 1) + self.knots
                + [self.knots[-1]] * (self.order - 1))
            # c is used the select which spline the function
            # PPoly.from_spline below computes
            c = np.zeros(len(knots))

            # Initialise empty list to store the piecewise polynomials
            ppoly_lst = []

            no_0_intervals = np.where(np.diff(knots) > 0)[0]

            # For each basis gets its piecewise polynomial representation
            for i in range(self.n_basis):
                # write a 1 in c in the position of the spline
                # transformed in each iteration
                c[i] = 1
                # gets the piecewise polynomial representation and gets
                # only the positions for no zero length intervals
                # This polynomial are defined relatively to the knots
                # meaning that the column i corresponds to the ith knot.
                # Let the ith not be a
                # Then f(x) = pp(x - a)
                pp = (PPoly.from_spline(
                    (knots, c, self.order - 1)).c[:, no_0_intervals])                    # We need the actual coefficients of f, not pp. So we
                # just recursively calculate the new coefficients
                coeffs = pp.copy()
                for j in range(self.order - 1):
                    coeffs[j + 1:] += (
                        (binom(self.order - j - 1,
                               range(1, self.order - j)) *
                         np.vstack([(-a) **
                                    np.array(range(1, self.order - j))
                                    for a in self.knots[:-1]])).T *
                        pp[j])
                ppoly_lst.append(coeffs)
                c[i] = 0

            # Now for each pair of basis computes the inner product after
            # applying the linear differential operator
            penalty_matrix = np.zeros((self.n_basis, self.n_basis))
            for interval in range(len(no_0_intervals)):
                for i in range(self.n_basis):
                    poly_i = np.trim_zeros(ppoly_lst[i][:,
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
                    penalty_matrix[i, i] += np.diff(polyval(
                        integral, self.knots[interval: interval + 2]))[0]

                    for j in range(i + 1, self.n_basis):
                        poly_j = np.trim_zeros(ppoly_lst[j][:,
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
                        penalty_matrix[i, j] += np.diff(polyval(
                            integral, self.knots[interval: interval + 2])
                        )[0]
                        penalty_matrix[j, i] = penalty_matrix[i, j]
            return penalty_matrix

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

    def __eq__(self, other):
        """Equality of Basis"""
        return (super().__eq__(other)
                and self.order == other.order
                and self.knots == other.knots)

    def basis_of_product(self, other):
        """Multiplication of two Bspline Basis"""
        if not _same_domain(self.domain_range, other.domain_range):
            raise ValueError("Ranges are not equal.")

        if isinstance(other, Constant):
            return other.rbasis_of_product(self)

        if isinstance(other, BSpline):
            uniqueknots = np.union1d(self.inknots, other.inknots)

            multunique = np.zeros(len(uniqueknots), dtype=np.int32)
            for i in range(len(uniqueknots)):
                mult1 = np.count_nonzero(self.inknots == uniqueknots[i])
                mult2 = np.count_nonzero(other.inknots == uniqueknots[i])
                multunique[i] = max(mult1, mult2)

            m2 = 0
            allknots = np.zeros(np.sum(multunique))
            for i in range(len(uniqueknots)):
                m1 = m2
                m2 = m2 + multunique[i]
                allknots[m1:m2] = uniqueknots[i]

            norder1 = self.n_basis - len(self.inknots)
            norder2 = other.n_basis - len(other.inknots)
            norder = min(norder1 + norder2 - 1, 20)

            allbreaks = ([self.domain_range[0][0]] +
                         np.ndarray.tolist(allknots) +
                         [self.domain_range[0][1]])
            n_basis = len(allbreaks) + norder - 2
            return BSpline(self.domain_range, n_basis, norder, allbreaks)
        else:
            norder = min(self.n_basis - len(self.inknots) + 2, 8)
            n_basis = max(self.n_basis + other.n_basis, norder + 1)
            return BSpline(self.domain_range, n_basis, norder)

    def rbasis_of_product(self, other):
        """Multiplication of a Bspline Basis with other basis"""

        norder = min(self.n_basis - len(self.inknots) + 2, 8)
        n_basis = max(self.n_basis + other.n_basis, norder + 1)
        return BSpline(self.domain_range, n_basis, norder)

    def _to_R(self):
        drange = self.domain_range[0]
        return ("create.bspline.basis(rangeval = c(" + str(drange[0]) + "," +
                str(drange[1]) + "), nbasis = " + str(self.n_basis) +
                ", norder = " + str(self.order) + ", breaks = " +
                self._list_to_R(self.knots) + ")")

    def _to_scipy_BSpline(self, coefs):

        knots = np.concatenate((
            np.repeat(self.knots[0], self.order - 1),
            self.knots,
            np.repeat(self.knots[-1], self.order - 1)))

        return SciBSpline(knots, coefs, self.order - 1)

    @staticmethod
    def _from_scipy_BSpline(bspline):
        order = bspline.k
        knots = bspline.t[order: -order]
        coefs = bspline.c
        domain_range = [knots[0], knots[-1]]

        return BSpline(domain_range, order=order + 1, knots=knots), coefs

    @property
    def inknots(self):
        """Return number of basis."""
        return self.knots[1:len(self.knots) - 1]


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
        n_basis (int): Number of functions in the basis.
        period (int or float): Period (:math:`T`).

    Examples:
        Constructs specifying number of basis, definition interval and period.

        >>> fb = Fourier((0, np.pi), n_basis=3, period=1)
        >>> fb.evaluate([0, np.pi / 4, np.pi / 2, np.pi]).round(2)
        array([[ 1.  ,  1.  ,  1.  ,  1.  ],
               [ 0.  , -1.38, -0.61,  1.1 ],
               [ 1.41,  0.31, -1.28,  0.89]])

        And evaluate second derivative

        >>> fb.evaluate([0, np.pi / 4, np.pi / 2, np.pi],
        ...             derivative = 2).round(2)
        array([[  0.  ,   0.  ,   0.  ,   0.  ],
               [ -0.  ,  54.46,  24.02, -43.37],
               [-55.83, -12.32,  50.4 , -35.16]])



    """

    def __init__(self, domain_range=None, n_basis=3, period=None):
        """Construct a Fourier object.

        It forces the object to have an odd number of basis. If n_basis is
        even, it is incremented by one.

        Args:
            domain_range (tuple): Tuple defining the domain over which the
            function is defined.
            n_basis (int): Number of basis functions.
            period (int or float): Period of the trigonometric functions that
                define the basis.

        """

        if domain_range is not None:
            domain_range = _list_of_arrays(domain_range)

            if len(domain_range) != 1:
                raise ValueError("Domain range should be unidimensional.")

            domain_range = domain_range[0]

        self.period = period
        # If number of basis is even, add 1
        n_basis += 1 - n_basis % 2
        super().__init__(domain_range, n_basis)

    @property
    def period(self):
        if self._period is None:
            return self.domain_range[0][1] - self.domain_range[0][0]
        else:
            return self._period

    @period.setter
    def period(self, value):
        self._period = value

    def _functions_pairs_coefs_derivatives(self, derivative=0):
        """
        Compute functions to use, amplitudes and phase of a derivative.
        """
        functions = [np.sin, np.cos]
        signs = [1, 1, -1, -1]
        omega = 2 * np.pi / self.period

        deriv_functions = (functions[derivative % len(functions)],
                           functions[(derivative + 1) % len(functions)])

        deriv_signs = (signs[derivative % len(signs)],
                       signs[(derivative + 1) % len(signs)])

        seq = 1 + np.arange((self.n_basis - 1) // 2)
        seq_pairs = np.array([seq, seq]).T
        power_pairs = (omega * seq_pairs)**derivative
        amplitude_coefs_pairs = deriv_signs * power_pairs
        phase_coef_pairs = omega * seq_pairs

        return deriv_functions, amplitude_coefs_pairs, phase_coef_pairs

    def _evaluate(self, eval_points, derivative=0):
        """Compute the basis or its derivatives given a list of values.

        Args:
            eval_points (array_like): List of points where the basis is
                evaluated.
            derivative (int, optional): Order of the derivative. Defaults to 0.

        Returns:
            (:obj:`numpy.darray`): Matrix whose rows are the values of the each
            basis function or its derivatives at the values specified in
            eval_points.

        """
        (functions,
         amplitude_coefs,
         phase_coefs) = self._functions_pairs_coefs_derivatives(derivative)

        normalization_denominator = np.sqrt(self.period / 2)

        # Multiply the phase coefficients elementwise
        res = np.einsum('ij,k->ijk', phase_coefs, eval_points)

        # Apply odd and even functions
        for i in [0, 1]:
            functions[i](res[:, i, :], out=res[:, i, :])

        # Multiply the amplitude and ravel the result
        res *= amplitude_coefs[..., np.newaxis]
        res = res.reshape(-1, len(eval_points))
        res /= normalization_denominator

        # Add constant basis
        if derivative == 0:
            constant_basis = np.full(
                shape=(1, len(eval_points)),
                fill_value=1 / (np.sqrt(2) * normalization_denominator))
        else:
            constant_basis = np.zeros(shape=(1, len(eval_points)))

        res = np.concatenate((constant_basis, res))

        return res

    def _derivative(self, coefs, order=1):

        omega = 2 * np.pi / self.period
        deriv_factor = (np.arange(1, (self.n_basis + 1) / 2) * omega) ** order

        deriv_coefs = np.zeros(coefs.shape)

        cos_sign, sin_sign = ((-1) ** int((order + 1) / 2),
                              (-1) ** int(order / 2))

        if order % 2 == 0:
            deriv_coefs[:, 1::2] = sin_sign * coefs[:, 1::2] * deriv_factor
            deriv_coefs[:, 2::2] = cos_sign * coefs[:, 2::2] * deriv_factor
        else:
            deriv_coefs[:, 2::2] = sin_sign * coefs[:, 1::2] * deriv_factor
            deriv_coefs[:, 1::2] = cos_sign * coefs[:, 2::2] * deriv_factor

        # normalise
        return self.copy(), deriv_coefs

    def basis_of_product(self, other):
        """Multiplication of two Fourier Basis"""
        if not _same_domain(self.domain_range, other.domain_range):
            raise ValueError("Ranges are not equal.")

        if isinstance(other, Fourier) and self.period == other.period:
            return Fourier(self.domain_range, self.n_basis + other.n_basis - 1,
                           self.period)
        else:
            return other.rbasis_of_product(self)

    def rbasis_of_product(self, other):
        """Multiplication of a Fourier Basis with other Basis"""
        return Basis.default_basis_of_product(other, self)

    def rescale(self, domain_range=None, *, rescale_period=False):
        r"""Return a copy of the basis with a new domain range, with the
            corresponding values rescaled to the new bounds.

            Args:
                domain_range (tuple, optional): Definition of the interval
                    where the basis defines a space. Defaults uses the same as
                    the original basis.
                rescale_period (bool, optional): If true the period will be
                    rescaled using the ratio between the lengths of the new
                    and old interval. Defaults to False.
        """

        rescale_basis = super().rescale(domain_range)

        if rescale_period is False:
            rescale_basis.period = self.period
        else:
            domain_rescaled = rescale_basis.domain_range[0]
            domain = self.domain_range[0]

            rescale_basis.period = (self.period *
                                    (domain_rescaled[1] - domain_rescaled[0]) /
                                    (domain[1] - domain[0]))

        return rescale_basis

    def _to_R(self):
        drange = self.domain_range[0]
        return ("create.fourier.basis(rangeval = c(" + str(drange[0]) + "," +
                str(drange[1]) + "), nbasis = " + str(self.n_basis) +
                ", period = " + str(self.period) + ")")

    def __repr__(self):
        """Representation of a Fourier basis."""
        return (f"{self.__class__.__name__}(domain_range={self.domain_range}, "
                f"n_basis={self.n_basis}, period={self.period})")

    def __eq__(self, other):
        """Equality of Basis"""
        return super().__eq__(other) and self.period == other.period


class CoefficientsTransformer(BaseEstimator, TransformerMixin):
    """
    Transformer returning the coefficients of FDataBasis objects as a matrix.

    Attributes:
        shape_ (tuple): original shape of coefficients per sample.

    Examples:
        >>> from skfda.representation.basis import (FDataBasis, Monomial,
        ...                                         CoefficientsTransformer)
        >>>
        >>> basis = Monomial(n_basis=4)
        >>> coefficients = [[0.5, 1, 2, .5], [1.5, 1, 4, .5]]
        >>> fd = FDataBasis(basis, coefficients)
        >>>
        >>> transformer = CoefficientsTransformer()
        >>> transformer.fit_transform(fd)
        array([[ 0.5,  1. ,  2. ,  0.5],
               [ 1.5,  1. ,  4. ,  0.5]])

    """

    def fit(self, X: FDataBasis, y=None):

        self.shape_ = X.coefficients.shape[1:]

        return self

    def transform(self, X, y=None):

        check_is_fitted(self, 'shape_')

        assert X.coefficients.shape[1:] == self.shape_

        coefficients = X.coefficients.copy()
        coefficients = coefficients.reshape((X.n_samples, -1))

        return coefficients
