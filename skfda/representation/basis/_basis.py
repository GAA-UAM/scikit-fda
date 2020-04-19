"""Module for functional data manipulation in a basis system.

Defines functional data object in a basis function system representation and
the corresponding basis classes.

"""
from abc import ABC, abstractmethod
import copy

import scipy.integrate

import numpy as np

from ..._utils import _list_of_arrays


__author__ = "Miguel Carbajo Berrocal"
__email__ = "miguel.carbajo@estudiante.uam.es"

# aux functions


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

        eval_points = np.atleast_1d(eval_points)
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

    def _internal_representation(self):
        """
        Returns an internal representation of the basis.

        This representation may have several operations available that return
        objects of the same kind, and can be used to build operators in an
        analytical, but generic, way.

        """
        return NotImplemented

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
        from ._bspline import BSpline

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
        from . import FDataBasis
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
