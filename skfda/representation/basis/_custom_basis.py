"""Abstract base class for basis."""

from __future__ import annotations

from typing import Any, Tuple, TypeVar

import numpy as np

from ...typing._numpy import NDArrayFloat
from .._functional_data import FData
from ..grid import FDataGrid
from ._basis import Basis
from ._fdatabasis import FDataBasis

T = TypeVar("T", bound="CustomBasis")


class CustomBasis(Basis):
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
        if isinstance(fdata, FDataGrid):
            self._check_linearly_independent_grid(fdata)
        else:
            self._check_linearly_independent_basis(fdata)

        self.fdata = fdata

    def _check_linearly_independent_grid(self, fdata: FDataGrid) -> None:
        """Check if the observations in the FDataGrid linearly independent."""
        if fdata.n_samples > fdata.data_matrix.shape[1]:
            raise ValueError(
                "Too many samples in the basis. "
                "The number of samples must be less than or equal to the "
                "number of sampling points.",
            )

        new_shape = (
            fdata.data_matrix.shape[0],
            fdata.data_matrix.shape[1] * fdata.data_matrix.shape[2],
        )
        rank = np.linalg.matrix_rank(fdata.data_matrix.reshape(new_shape))

        if rank != fdata.n_samples:
            raise ValueError(
                "There are only {rank} linearly independent "
                "functions".format(
                    rank=rank,
                ),
            )

    def _check_linearly_independent_basis(self, fdata: FDataBasis) -> None:
        """Check if the observations in the FDataBasis linearly independent."""
        if fdata.n_samples > fdata.basis.n_basis:
            raise ValueError(
                "Too many samples in the basis. "
                "The number of samples must be less than or equal to the "
                "number of basis functions.",
            )
        if np.linalg.matrix_rank(fdata.coefficients) < fdata.n_samples:
            raise ValueError(
                "There are only {rank} linearly independent functions".format(
                    rank=np.linalg.matrix_rank(fdata.coefficients),
                ),
            )
        pass

    def _derivative_basis_and_coefs(
        self: T,
        coefs: NDArrayFloat,
        order: int = 1,
    ) -> Tuple[T, NDArrayFloat]:

        derivated_basis = CustomBasis(
            fdata=self.fdata.derivative(order=order),
        )

        return derivated_basis, coefs

    def _coordinate_nonfull(
        self,
        coefs: NDArrayFloat,
        key: int | slice,
    ) -> Tuple[Basis, NDArrayFloat]:
        return CustomBasis(fdata=self.fdata.coordinates[key]), coefs

    def _evaluate(
        self,
        eval_points: NDArrayFloat,
    ) -> NDArrayFloat:
        return self.fdata(eval_points)

    def __len__(self) -> int:
        return self.n_basis

    @property
    def dim_codomain(self) -> int:
        return self.fdata.dim_codomain

    def __eq__(self, other: Any) -> bool:
        from ..._utils import _same_domain

        return (
            isinstance(other, type(self))
            and _same_domain(self, other)
            and self.fdata == other.fdata
        )

    def __hash__(self) -> int:
        return hash(self.fdata)
