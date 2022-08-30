"""Abstract base class for basis."""

from __future__ import annotations

import copy
import warnings
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any, Tuple, TypeVar

import numpy as np
from matplotlib.figure import Figure

from ...typing._base import DomainRange, DomainRangeLike
from ...typing._numpy import ArrayLike, NDArrayFloat

if TYPE_CHECKING:
    from ._fdatabasis import FDataBasis

T = TypeVar("T", bound='Basis')


class Basis(ABC):
    """Defines the structure of a basis of functions.

    Parameters:
        domain_range: The :term:`domain range` over which the basis can be
            evaluated.
        n_basis: number of functions in the basis.

    """

    def __init__(
        self,
        *,
        domain_range: DomainRangeLike | None = None,
        n_basis: int = 1,
    ) -> None:
        """Basis constructor."""
        from ...misc.validation import validate_domain_range

        if domain_range is not None:

            domain_range = validate_domain_range(domain_range)

        if n_basis < 1:
            raise ValueError(
                "The number of basis has to be strictly positive.",
            )

        self._domain_range = domain_range
        self._n_basis = n_basis

        super().__init__()

    def __call__(
        self,
        eval_points: ArrayLike,
        *,
        derivative: int = 0,
    ) -> NDArrayFloat:
        """Evaluate Basis objects.

        Evaluates the basis functions at a list of given values.

        Args:
            eval_points: List of points where the basis is
                evaluated.
            derivative: order of the derivative.

                .. deprecated:: 0.4
                    Use `derivative` method instead.

        Returns:
            Matrix whose rows are the values of the each
            basis function or its derivatives at the values specified in
            eval_points.

        """
        from ...misc.validation import validate_evaluation_points

        if derivative < 0:
            raise ValueError("derivative only takes non-negative values.")
        elif derivative != 0:
            warnings.warn(
                "Parameter derivative is deprecated. Use the "
                "derivative method instead.",
                DeprecationWarning,
            )
            return self.derivative(order=derivative)(eval_points)

        eval_points = validate_evaluation_points(
            eval_points,
            aligned=True,
            n_samples=self.n_basis,
            dim_domain=self.dim_domain,
        )

        return self._evaluate(eval_points).reshape(
            (self.n_basis, len(eval_points), self.dim_codomain),
        )

    @property
    def dim_domain(self) -> int:
        if self._domain_range is None:
            return 1
        return len(self._domain_range)

    @property
    def dim_codomain(self) -> int:
        return 1

    @property
    def domain_range(self) -> DomainRange:
        if self._domain_range is None:
            return ((0.0, 1.0),) * self.dim_domain

        return self._domain_range

    @property
    def n_basis(self) -> int:
        return self._n_basis

    def is_domain_range_fixed(self) -> bool:
        """
        Return wether the :term:`domain range` has been set explicitly.

        This is useful when using a basis for converting a dataset, since
        if this is not explicitly assigned it can be changed to the domain of
        the data.

        Returns:
            `True` if the domain range has been fixed. `False` otherwise.

        """
        return self._domain_range is not None

    @abstractmethod
    def _evaluate(
        self,
        eval_points: NDArrayFloat,
    ) -> NDArrayFloat:
        """
        Evaluate Basis object.

        Subclasses must override this to provide basis evaluation.

        """
        pass

    def evaluate(
        self,
        eval_points: ArrayLike,
        *,
        derivative: int = 0,
    ) -> NDArrayFloat:
        """
        Evaluate Basis objects and its derivatives.

        Evaluates the basis functions at a list of given values.

        ..  deprecated:: 0.8
            Use normal calling notation instead.

        Args:
            eval_points: List of points where the basis is
                evaluated.
            derivative: order of the derivative.

                .. deprecated:: 0.4
                    Use `derivative` method instead.

        Returns:
            Matrix whose rows are the values of the each
            basis function or its derivatives at the values specified in
            eval_points.

        """
        warnings.warn(
            "The method 'evaluate' is deprecated. "
            "Please use the normal calling notation on the basis "
            "object instead.",
            DeprecationWarning,
            stacklevel=2,
        )

        return self(
            eval_points=eval_points,
            derivative=derivative,
        )

    def __len__(self) -> int:
        return self.n_basis

    def derivative(self, *, order: int = 1) -> FDataBasis:
        """Construct a FDataBasis object containing the derivative.

        Args:
            order: Order of the derivative. Defaults to 1.

        Returns:
            Derivative object.

        """
        return self.to_basis().derivative(order=order)

    def _derivative_basis_and_coefs(
        self: T,
        coefs: NDArrayFloat,
        order: int = 1,
    ) -> Tuple[T, NDArrayFloat]:
        """
        Return basis and coefficients of the derivative.

        Args:
            coefs: Coefficients of a vector expressed in this basis.
            order: Order of the derivative.

        Returns:
            Tuple with the basis of the derivative and its coefficients.

        Subclasses can override this to provide derivative construction.

        """
        raise NotImplementedError(
            f"{type(self)} basis does not support the construction of a "
            "basis of the derivatives.",
        )

    def derivative_basis_and_coefs(
        self: T,
        coefs: NDArrayFloat,
        order: int = 1,
    ) -> Tuple[T, NDArrayFloat]:
        """
        Return basis and coefficients of the derivative.

        Args:
            coefs: Coefficients of a vector expressed in this basis.
            order: Order of the derivative.

        Returns:
            Tuple with the basis of the derivative and its coefficients.

        """
        return self._derivative_basis_and_coefs(coefs, order)

    def plot(self, *args: Any, **kwargs: Any) -> Figure:
        """Plot the basis object or its derivatives.

        Args:
            args: arguments to be passed to the
                fdata.plot function.
            kwargs: keyword arguments to be passed to the
                fdata.plot function.

        Returns:
            Figure object in which the graphs are plotted.

        """
        self.to_basis().plot(*args, **kwargs)

    def _coordinate_nonfull(
        self,
        coefs: NDArrayFloat,
        key: int | slice,
    ) -> Tuple[Basis, NDArrayFloat]:
        """
        Return a basis and coefficients for the indexed coordinate functions.

        Subclasses can override this to provide coordinate indexing.

        """
        raise NotImplementedError("Coordinate indexing not implemented")

    def coordinate_basis_and_coefs(
        self,
        coefs: NDArrayFloat,
        key: int | slice,
    ) -> Tuple[Basis, NDArrayFloat]:
        """Return a fdatabasis for the coordinate functions indexed by key."""
        # Raises error if not in range and normalize key
        r_key = range(self.dim_codomain)[key]

        if isinstance(r_key, range) and len(r_key) == 0:
            raise IndexError("Empty number of coordinates selected")

        # Full fdatabasis case
        if (
            (self.dim_codomain == 1 and r_key == 0)
            or (isinstance(r_key, range) and len(r_key) == self.dim_codomain)
        ):
            return self, np.copy(coefs)

        return self._coordinate_nonfull(
            coefs=coefs,
            key=key,
        )

    def rescale(self: T, domain_range: DomainRangeLike | None = None) -> T:
        """
        Return a copy of the basis with a new :term:`domain` range.

        Args:
            domain_range: Definition of the interval
                where the basis defines a space. Defaults uses the same as
                the original basis.

        Returns:
            Rescaled copy.

        """
        return self.copy(domain_range=domain_range)

    def copy(self: T, domain_range: DomainRangeLike | None = None) -> T:
        """Basis copy."""
        from ...misc.validation import validate_domain_range

        new_copy = copy.deepcopy(self)

        if domain_range is not None:
            domain_range = validate_domain_range(domain_range)

            new_copy._domain_range = domain_range  # noqa: WPS437

        return new_copy

    def to_basis(self) -> FDataBasis:
        """
        Convert the Basis to FDatabasis.

        Returns:
            FDataBasis with this basis as its basis, and all basis functions
            as observations.

        """
        from . import FDataBasis  # noqa: WPS442
        return FDataBasis(self.copy(), np.identity(self.n_basis))

    def _to_R(self) -> str:  # noqa: N802
        raise NotImplementedError

    def inner_product_matrix(
        self,
        other: Basis | None = None,
    ) -> NDArrayFloat:
        r"""
        Return the Inner Product Matrix of a pair of basis.

        The Inner Product Matrix is defined as

        .. math::
            I_{ij} = \langle\phi_i, \theta_j\rangle

        where :math:`\phi_i` is the ith element of the basis and
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

    def _gram_matrix_numerical(self) -> NDArrayFloat:
        """Compute the Gram matrix numerically."""
        from ...misc import inner_product_matrix

        return inner_product_matrix(self, force_numerical=True)

    def _gram_matrix(self) -> NDArrayFloat:
        """
        Compute the Gram matrix.

        Subclasses may override this method for improving computation
        of the Gram matrix.

        """
        return self._gram_matrix_numerical()

    def gram_matrix(self) -> NDArrayFloat:
        r"""
        Return the Gram Matrix of a basis.

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

    def _mul_constant(
        self: T,
        coefs: NDArrayFloat,
        other: float,
    ) -> Tuple[T, NDArrayFloat]:
        coefs = coefs.copy()
        other_array = np.atleast_2d(other).reshape(-1, 1)
        coefs *= other_array

        return self.copy(), coefs

    def __repr__(self) -> str:
        """Representation of a Basis object."""
        return (
            f"{self.__class__.__name__}("
            f"domain_range={self.domain_range}, "
            f"n_basis={self.n_basis})"
        )

    def __eq__(self, other: Any) -> bool:
        """Test equality of Basis."""
        from ..._utils import _same_domain
        return (
            isinstance(other, type(self))
            and _same_domain(self, other)
            and self.n_basis == other.n_basis
        )

    def __hash__(self) -> int:
        """Hash a Basis."""
        return hash((self.domain_range, self.n_basis))
