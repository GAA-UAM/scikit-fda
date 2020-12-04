"""Module for functional data manipulation in a basis system.

Defines functional data object in a basis function system representation and
the corresponding basis classes.

"""
import copy
import warnings
from abc import ABC, abstractmethod
from typing import Tuple

import numpy as np

from ..._utils import _domain_range, _reshape_eval_points, _same_domain
from . import _fdatabasis


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

    def __init__(self, *, domain_range=None, n_basis: int = 1):
        """Basis constructor.

        Args:
            domain_range (tuple or list of tuples, optional): Definition of the
                interval where the basis defines a space. Defaults to (0,1).
            n_basis: Number of functions that form the basis. Defaults to 1.

        """
        if domain_range is not None:

            domain_range = _domain_range(domain_range)

            # Some checks
            _check_domain(domain_range)

        if n_basis < 1:
            raise ValueError(
                "The number of basis has to be strictly positive.",
            )

        self._domain_range = domain_range
        self._n_basis = n_basis

        super().__init__()

    def __call__(self, *args, **kwargs) -> np.ndarray:
        """Evaluate the basis using :meth:`evaluate`."""
        return self.evaluate(*args, **kwargs)

    @property
    def dim_domain(self) -> int:
        return 1

    @property
    def dim_codomain(self) -> int:
        return 1

    @property
    def domain_range(self) -> Tuple[Tuple[float, float], ...]:
        if self._domain_range is None:
            return ((0, 1),) * self.dim_domain
        else:
            return self._domain_range

    @property
    def n_basis(self) -> int:
        return self._n_basis

    @abstractmethod
    def _evaluate(self, eval_points) -> np.ndarray:
        """Subclasses must override this to provide basis evaluation."""
        pass

    def evaluate(self, eval_points, *, derivative: int = 0) -> np.ndarray:
        """Evaluate Basis objects and its derivatives.

        Evaluates the basis function system or its derivatives at a list of
        given values.

        Args:
            eval_points (array_like): List of points where the basis is
                evaluated.

        Returns:
            Matrix whose rows are the values of the each
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

    def __len__(self) -> int:
        return self.n_basis

    def derivative(self, *, order: int = 1) -> '_fdatabasis.FDataBasis':
        """Construct a FDataBasis object containing the derivative.

        Args:
            order: Order of the derivative. Defaults to 1.

        Returns:
            Derivative object.

        """

        return self.to_basis().derivative(order=order)

    def _derivative_basis_and_coefs(self, coefs: np.ndarray, order: int = 1):
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

    def rescale(self, domain_range=None):
        r"""Return a copy of the basis with a new :term:`domain` range, with
            the corresponding values rescaled to the new bounds.

            Args:
                domain_range (tuple, optional): Definition of the interval
                    where the basis defines a space. Defaults uses the same as
                    the original basis.
        """

        return self.copy(domain_range=domain_range)

    def copy(self, domain_range=None):
        """Basis copy"""

        new_copy = copy.deepcopy(self)

        if domain_range is not None:
            domain_range = _domain_range(domain_range)

            # Some checks
            _check_domain(domain_range)

            new_copy._domain_range = domain_range

        return new_copy

    def to_basis(self) -> '_fdatabasis.FDataBasis':
        """Convert the Basis to FDatabasis.

        Returns:
            FDataBasis with this basis as its basis, and all basis functions
            as observations.

        """
        from . import FDataBasis
        return FDataBasis(self.copy(), np.identity(self.n_basis))

    def _list_to_R(self, knots):
        retstring = "c("
        for i in range(0, len(knots)):
            retstring = retstring + str(knots[i]) + ", "
        return retstring[0:len(retstring) - 2] + ")"

    def _to_R(self):
        raise NotImplementedError

    def inner_product_matrix(self, other: 'Basis' = None) -> np.array:
        r"""Return the Inner Product Matrix of a pair of basis.

        The Inner Product Matrix is defined as

        .. math::
            IP_{ij} = \langle\phi_i, \theta_j\rangle

        where :math:`\phi_i` is the ith element of the basi and
        :math:`\theta_j` is the jth element of the second basis.
        This matrix helps on the calculation of the inner product
        between objects on two basis and for the change of basis.

        Args:
            other: Basis to compute the inner product
                matrix. If not basis is given, it computes the matrix with
                itself returning the Gram Matrix

        Returns:
            Inner Product Matrix of two basis

        """
        from ...misc import inner_product_matrix

        if other is None or self == other:
            return self.gram_matrix()

        return inner_product_matrix(self, other)

    def _gram_matrix_numerical(self) -> np.array:
        """
        Compute the Gram matrix numerically.

        """
        from ...misc import inner_product_matrix

        return inner_product_matrix(self, force_numerical=True)

    def _gram_matrix(self) -> np.array:
        """
        Compute the Gram matrix.

        Subclasses may override this method for improving computation
        of the Gram matrix.

        """
        return self._gram_matrix_numerical()

    def gram_matrix(self) -> np.array:
        r"""Return the Gram Matrix of a basis

        The Gram Matrix is defined as

        .. math::
            G_{ij} = \langle\phi_i, \phi_j\rangle

        where :math:`\phi_i` is the ith element of the basis. This is a
        symmetric matrix and positive-semidefinite.

        Returns:
            Gram Matrix of the basis.

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

    def __repr__(self) -> str:
        """Representation of a Basis object."""
        return (f"{self.__class__.__name__}(domain_range={self.domain_range}, "
                f"n_basis={self.n_basis})")

    def __eq__(self, other) -> bool:
        """Equality of Basis"""
        return (type(self) == type(other)
                and _same_domain(self, other)
                and self.n_basis == other.n_basis)

    def __hash__(self) -> int:
        """Hash of Basis"""
        return hash((self.domain_range, self.n_basis))
