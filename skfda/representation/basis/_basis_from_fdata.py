"""Abstract base class for basis."""

from __future__ import annotations

from typing import Any, Tuple, TypeVar

from basis import Basis

from ...typing._numpy import NDArrayFloat
from .._functional_data import FData
from _fdatabasis import FDataBasis

T = TypeVar("T", bound='BasisFromFdata')


class BasisFromFdata(Basis):
    """Defines the structure of a basis of functions.

    Parameters:
        domain_range: The :term:`domain range` over which the basis can be
            evaluated.
        n_basis: number of functions in the basis.

    """

    def __init__(
        self,
        *,
        fdata: FData,
    ) -> None:
        """Basis constructor."""
        super().__init__(
            domain_range=fdata.domain_range,
            n_basis=fdata.n_samples,
        )

    def _evaluate(
        self,
        eval_points: NDArrayFloat,
    ) -> NDArrayFloat:
        """Evaluate Basis object."""
        return self.fdata(eval_points)

    def __len__(self) -> int:
        return self.n_basis

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
        derivated_basis = BasisFromFdata(fdata=self.fdata.derivative(order))

        return derivated_basis, coefs

        raise NotImplementedError(
            f"{type(self)} basis does not support the construction of a "
            "basis of the derivatives.",
        )

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

    def __eq__(self, other: Any) -> bool:
        """Test equality of Basis."""
        from ..._utils import _same_domain
        return (
            isinstance(other, type(self))
            and _same_domain(self, other)
            and self.fdata == other.fdata
        )

    def __hash__(self) -> int:
        """Hash a Basis."""
        return hash(self.fdata)
