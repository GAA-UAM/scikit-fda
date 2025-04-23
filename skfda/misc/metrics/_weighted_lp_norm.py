"""Implementation of Weighted Lp norms."""

from collections.abc import Callable

import numpy as np

from ..._utils import nquad_vec
from ...representation import FData, FDataBasis, FDataGrid
from ...typing._base import (
    GridPointsLike,
)
from ...typing._metric import Norm
from ...typing._numpy import NDArrayFloat


class WeightedLpNorm:
    def __init__(
        self,
        p: float,
        vector_norm: Norm[NDArrayFloat] | float | None = None,
        lp_weight: (
            Callable[[GridPointsLike], NDArrayFloat] | float | None
        ) = None,
    ) -> None:

        # Checks that the lp normed is well defined
        if not np.isinf(p) and p < 1:
            msg = f"p (={p}) must be equal or greater than 1."
            raise ValueError(msg)

        self.p = p
        self.vector_norm = vector_norm
        self.lp_weight = lp_weight

    def __repr__(self) -> str:
        return f"{type(self).__name__}(p={self.p}, vector_norm={self.vector_norm})"

    def __call__(self, vector: NDArrayFloat | FData) -> NDArrayFloat:
        """Compute the Lp norm of a functional data object."""
        from .. import weighted_inner_product

        if isinstance(vector, np.ndarray):
            if isinstance(self.lp_weight, (float, int)):
                vector = vector * self.lp_weight
            return np.linalg.norm(  # type: ignore[no-any-return]
                vector,
                ord=self.p,
                axis=-1,
            )

        vector_norm = self.vector_norm
        lp_weight = self.lp_weight

        if vector_norm is None:
            vector_norm = self.p
        if lp_weight is None:
            lp_weight = 1.0

        # Special case, the inner product is heavily optimized  TODO
        """ if self.p == vector_norm == 2:
            return np.sqrt(weighted_inner_product(vector, vector)) """

        if isinstance(vector, FDataBasis):
            domain = vector.basis.domain_range
            call = vector

            def integrand(*args: NDArrayFloat) -> NDArrayFloat:
                f_args = np.asarray(args)

                try:
                    f1 = call(f_args)[:, 0, :]
                except Exception:  # noqa: BLE001
                    f1 = call(f_args)
                weight = (
                    lp_weight
                    if isinstance(lp_weight, (float, int))
                    else lp_weight(f_args)
                )
                return np.asarray(
                    np.power(np.abs(f1), self.p) * weight,
                    dtype=np.float64,
                )

            integral = nquad_vec(
                integrand,
                domain,
            )

            res = (np.sum(integral, axis=-1)) ** (1 / self.p)

        elif isinstance(vector, FDataGrid):
            data_matrix = vector.data_matrix

            if isinstance(vector_norm, (float, int)):
                data_matrix = np.linalg.norm(
                    vector.data_matrix,
                    ord=vector_norm,
                    axis=-1,
                    keepdims=True,
                )
            else:
                original_shape = data_matrix.shape
                data_matrix = data_matrix.reshape(-1, original_shape[-1])
                data_matrix = vector_norm(data_matrix)
                data_matrix = data_matrix.reshape(original_shape[:-1] + (1,))

            data_matrix = (
                data_matrix * lp_weight
                if isinstance(lp_weight, (float, int))
                else lp_weight(vector.grid_points) * data_matrix
            )

            if np.isinf(self.p):
                res = np.max(
                    data_matrix,
                    axis=tuple(range(1, data_matrix.ndim)),
                )

            else:
                integrand = vector.copy(
                    data_matrix=data_matrix**self.p,
                    coordinate_names=(None,),
                )
                # Computes the norm, approximating the integral with Simpson's
                # rule.
                res = integrand.integrate().ravel() ** (1 / self.p)
        else:
            msg = f"LpNorm not implemented for type {type(vector)}"
            raise NotImplementedError(msg)

        if len(res) == 1:
            return res[0]  # type: ignore[no-any-return]

        return res  # type: ignore[no-any-return]


def weighted_lp_norm(
    vector: NDArrayFloat | FData,
    *,
    p: float,
    vector_norm: Norm[NDArrayFloat] | float | None = None,
    lp_weight: Callable[[GridPointsLike], NDArrayFloat] | float | None = None,
) -> NDArrayFloat:
    return WeightedLpNorm(p=p, vector_norm=vector_norm, lp_weight=lp_weight)(
        vector,
    )
