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
    def _check_linearly_independent(self, fdata) -> None:
        """Check if the functions are linearly independent."""
        raise ValueError(
            "The basis creation functionality is not available for the "
            "type of FData object provided",
        )

    @_check_linearly_independent.register
    def _check_linearly_independent_grid(self, fdata: FDataGrid) -> None:
        """Ensure the functions in the FDataGrid are linearly independent."""
        # Flatten the last dimension of the data matrix
        flattened_shape = (
            fdata.data_matrix.shape[0],
            fdata.data_matrix.shape[1] * fdata.data_matrix.shape[2],
        )

        if fdata.n_samples > flattened_shape[1]:
            raise ValueError(
                "Too many samples in the basis. The number of samples "
                "must be less than or equal to the number of sampling points "
                "times the dimension of the codomain.",
            )
        rank = np.linalg.matrix_rank(
            fdata.data_matrix.reshape(flattened_shape),
        )

        if rank < fdata.n_samples:
            raise ValueError(
                "There are only {rank} linearly independent "
                "functions".format(
                    rank=rank,
                ),
            )

    @_check_linearly_independent.register
    def _check_linearly_independent_basis(self, fdata: FDataBasis) -> None:
        """Ensure the functions in the FDataBasis are linearly independent."""
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

    def _derivative_basis_and_coefs(
        self: T,
        coefs: NDArrayFloat,
        order: int = 1,
    ) -> Tuple[T, NDArrayFloat]:
        deriv_fdata = self.fdata.derivative(order=order)
        new_basis = None

        if isinstance(deriv_fdata, FDataBasis):
            q, r = np.linalg.qr(deriv_fdata.coefficients.T)
            new_basis = CustomBasis(fdata=deriv_fdata.copy(coefficients=q.T))
            coefs = coefs @ deriv_fdata.coefficients @ q

        else:
            flattened_shape = (
                deriv_fdata.data_matrix.shape[0],
                (
                    deriv_fdata.data_matrix.shape[1]
                    * deriv_fdata.data_matrix.shape[2]
                ),
            )

            mat = deriv_fdata.data_matrix.reshape(flattened_shape)

            q, r = np.linalg.qr(mat.T)
            new_data = q.T.reshape(
                (
                    -1,
                    deriv_fdata.data_matrix.shape[1],
                    deriv_fdata.data_matrix.shape[2],
                ),
            )
            new_basis = CustomBasis(
                fdata=deriv_fdata.copy(
                    data_matrix=new_data,
                ),
            )
            coefs = coefs @ mat @ q

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
