"""Abstract base class for basis."""

from __future__ import annotations

from typing import Any, Tuple, TypeVar

import multimethod
import numpy as np

from ...typing._numpy import NDArrayFloat
from .._functional_data import FData
from ..grid import FDataGrid
from ._basis import Basis
from ._fdatabasis import FDataBasis

T = TypeVar("T", bound="CustomBasis")


class CustomBasis(Basis):
    """Basis composed of custom functions.

    Defines a basis composed of the functions in the :class: `FData` object
    passed as argument.
    The functions must be linearly independent, otherwise
    an exception is raised.

    Parameters:
        fdata: Functions that define the basis.

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
        self._check_linearly_independent(fdata)

        self.fdata = fdata

    @multimethod.multidispatch
    def _get_coordinates_matrix(self, fdata) -> NDArrayFloat:
        raise ValueError(
            "Unexpected type of functional data object.",
        )

    @multimethod.multidispatch
    def _set_coordinates_matrix(self, fdata, matrix: np.ndarray):
        raise ValueError(
            "Unexpected type of functional data object.",
        )

    @_get_coordinates_matrix.register
    def _get_coordinates_matrix_grid(self, fdata: FDataGrid) -> NDArrayFloat:
        return fdata.data_matrix

    @_set_coordinates_matrix.register
    def _set_coordinates_matrix_grid(
        self, fdata: FDataGrid, matrix: np.ndarray,
    ):
        fdata.data_matrix = matrix

    @_get_coordinates_matrix.register
    def _get_coordinates_matrix_basis(self, fdata: FDataBasis) -> NDArrayFloat:
        return fdata.coefficients

    @_set_coordinates_matrix.register
    def _set_coordinates_matrix_basis(
        self, fdata: FDataBasis, matrix: np.ndarray,
    ):
        fdata.coefficients = matrix

    def _check_linearly_independent(self, fdata) -> None:
        """Check if the functions are linearly independent."""
        coord_matrix = self._get_coordinates_matrix(fdata)
        coord_matrix = coord_matrix.reshape(coord_matrix.shape[0], -1)
        if coord_matrix.shape[0] > coord_matrix.shape[1]:
            raise ValueError(
                "Too many samples in the basis",
            )

        rank = np.linalg.matrix_rank(coord_matrix)
        if rank != coord_matrix.shape[0]:
            raise ValueError(
                "There are only {rank} linearly independent "
                "functions".format(
                    rank=rank,
                ),
            )

    def _derivative_basis_and_coefs(
        self: T,
        coefs: NDArrayFloat,
        order: int = 1,
    ) -> Tuple[T, NDArrayFloat]:
        deriv_fdata = self.fdata.derivative(order=order)
        new_basis = None

        coord_matrix = self._get_coordinates_matrix(deriv_fdata)
        coord_matrix_reshaped = coord_matrix.reshape(
            coord_matrix.shape[0],
            -1,
        )

        q, r = np.linalg.qr(coord_matrix_reshaped.T)

        new_data = q.T.reshape(
            -1,
            *coord_matrix.shape[1:],
        )

        self._set_coordinates_matrix(deriv_fdata, new_data)

        new_basis = CustomBasis(fdata=deriv_fdata)
        coefs = coefs @ coord_matrix_reshaped @ q

        return new_basis, coefs

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
