"""Module for functional data manipulation in a basis system.

Defines functional data object in a basis function system representation and
the corresponding basis classes.

"""
from abc import ABC, abstractmethod
import copy
import warnings

import numpy as np

from ..._utils import (_list_of_arrays, _same_domain,
                       _reshape_eval_points, _evaluate_grid)


__author__ = "Miguel Carbajo Berrocal"
__email__ = "miguel.carbajo@estudiante.uam.es"

# aux functions


def _check_domain(domain_range):
    for domain in domain_range:
        if len(domain) != 2 or domain[0] >= domain[1]:
            raise ValueError(f"The interval {domain} is not well-defined.")


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

        super().__init__()

    @property
    def dim_domain(self):
        return 1

    @property
    def dim_codomain(self):
        return 1

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
    def _evaluate(self, eval_points):
        """Subclasses must override this to provide basis evaluation."""
        pass

    def evaluate(self, eval_points, *, derivative=0):
        """Evaluate Basis objects and its derivatives.

        Evaluates the basis function system or its derivatives at a list of
        given values.

        Args:
            eval_points (array_like): List of points where the basis is
                evaluated.

        Returns:
            (numpy.darray): Matrix whose rows are the values of the each
            basis function or its derivatives at the values specified in
            eval_points.

        """
        if derivative < 0:
            raise ValueError("derivative only takes non-negative values.")
        elif derivative != 0:
            warnings.warn("Parameter derivative is deprecated. Use the "
                          "derivative function instead.", DeprecationWarning)
            return self.derivative(order=derivative)(eval_points)

        eval_points = _reshape_eval_points(eval_points,
                                           aligned=True,
                                           n_samples=self.n_basis,
                                           dim_domain=self.dim_domain)

        return self._evaluate(eval_points).reshape(
            (self.n_basis, len(eval_points), self.dim_codomain))

    def __call__(self, *args, **kwargs):
        return self.evaluate(*args, **kwargs)

    def __len__(self):
        return self.n_basis

    def derivative(self, *, order=1):
        """Construct a FDataBasis object containing the derivative.

        Args:
            order (int, optional): Order of the derivative. Defaults to 1.

        Returns:
            (FDataBasis): Derivative object.

        """

        return self.to_basis().derivative(order=order)

    def _derivative_basis_and_coefs(self, coefs, order=1):
        """
        Subclasses can override this to provide derivative construction.

        A basis can provide derivative evaluation at given points
        without providing a basis representation for its derivatives,
        although is recommended to provide both if possible.

        """
        raise NotImplementedError(f"{type(self)} basis does not support "
                                  "the construction of a basis of the "
                                  "derivatives.")

    def plot(self, chart=None, **kwargs):
        """Plot the basis object or its derivatives.

        Args:
            chart (figure object, axe or list of axes, optional): figure over
                with the graphs are plotted or axis over where the graphs are
                plotted.
            **kwargs: keyword arguments to be passed to the
                fdata.plot function.

        Returns:
            fig (figure): figure object in which the graphs are plotted.

        """
        self.to_basis().plot(chart=chart, **kwargs)

    def _coordinate_nonfull(self, fdatabasis, key):
        """
        Returns a fdatagrid for the coordinate functions indexed by key.

        Subclasses can override this to provide coordinate indexing.

        The key parameter has been already validated and is an integer or
        slice in the range [0, self.dim_codomain.

        """
        raise NotImplementedError("Coordinate indexing not implemented")

    def _coordinate(self, fdatabasis, key):
        """Returns a fdatagrid for the coordinate functions indexed by key."""

        # Raises error if not in range and normalize key
        r_key = range(self.dim_codomain)[key]

        if isinstance(r_key, range) and len(r_key) == 0:
            raise IndexError("Empty number of coordinates selected")

        # Full fdatabasis case
        if (self.dim_codomain == 1 and r_key == 0) or (
                isinstance(r_key, range) and len(r_key) == self.dim_codomain):

            return fdatabasis.copy()

        else:

            return self._coordinate_nonfull(fdatabasis=fdatabasis, key=r_key)

    @abstractmethod
    def basis_of_product(self, other):
        pass

    @abstractmethod
    def rbasis_of_product(self, other):
        pass

    @staticmethod
    def default_basis_of_product(one, other):
        """Default multiplication for a pair of basis"""
        from ._bspline import BSpline

        if not _same_domain(one, other):
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

    def copy(self):
        """Basis copy"""
        return copy.deepcopy(self)

    def to_basis(self):
        from . import FDataBasis
        return FDataBasis(self.copy(), np.identity(self.n_basis))

    def _list_to_R(self, knots):
        retstring = "c("
        for i in range(0, len(knots)):
            retstring = retstring + str(knots[i]) + ", "
        return retstring[0:len(retstring) - 2] + ")"

    def _to_R(self):
        raise NotImplementedError

    def inner_product_matrix(self, other=None):
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
        from ...misc import inner_product_matrix

        if other is None or self == other:
            return self.gram_matrix()

        return inner_product_matrix(self, other)

    def _gram_matrix_numerical(self):
        """
        Compute the Gram matrix numerically.

        """
        from ...misc import inner_product_matrix

        return inner_product_matrix(self, force_numerical=True)

    def _gram_matrix(self):
        """
        Compute the Gram matrix.

        Subclasses may override this method for improving computation
        of the Gram matrix.

        """
        return self._gram_matrix_numerical()

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

        gram = getattr(self, "_gram_matrix_cached", None)

        if gram is None:
            gram = self._gram_matrix()
            self._gram_matrix_cached = gram

        return gram

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
                and _same_domain(self, other)
                and self.n_basis == other.n_basis)
