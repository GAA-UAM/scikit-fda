"""Implementation of Weighted Lp norms."""

import math
from builtins import isinstance
from typing import Union

import numpy as np
import scipy.integrate # type: ignore[import-untyped]
from typing_extensions import Final
from typing import Union, Callable


from ...representation import FData, FDataBasis, FDataGrid
from ...typing._metric import Norm
from ...typing._numpy import NDArrayFloat


class WeightedLpNorm:
    def __init__(
        self,
        p: float,
        vector_norm: Union[Norm[NDArrayFloat], float, None] = None,
        lp_weight: Union[
            Callable[[NDArrayFloat], NDArrayFloat],
            float,
            None,
        ] = None,
    ) -> None:

        # Checks that the lp normed is well defined
        if not np.isinf(p) and p < 1:
            raise ValueError(f"p (={p}) must be equal or greater than 1.")

        self.p = p
        self.vector_norm = vector_norm
        self.lp_weight = lp_weight

    def __repr__(self) -> str:
        return f"{type(self).__name__}(" f"p={self.p}, vector_norm={self.vector_norm})"

    def __call__(self, vector: Union[NDArrayFloat, FData]) -> NDArrayFloat:
        """Compute the Lp norm of a functional data object."""
        from .. import inner_product

        if isinstance(vector, np.ndarray):
            if isinstance(self.lp_weight, (float, int)):
                vector = vector * self.lp_weight
            return np.linalg.norm(  # type: ignore[no-any-return]
                vector,
                ord=self.p,
                axis=-1,
            )

        vector_norm = self.vector_norm

        if vector_norm is None:
            vector_norm = self.p
        if self.lp_weight is None:
            self.lp_weight = 1.0

        # Special case, the inner product is heavily optimized
        if self.p == vector_norm == 2:
            return np.sqrt(inner_product(vector, vector))

        if isinstance(vector, FDataBasis):
            if self.p != 2:
                raise NotImplementedError

            start, end = vector.domain_range[0]
            integral = scipy.integrate.quad_vec(
                lambda x: (
                    self.lp_weight * np.power(np.abs(vector(x)), self.p)
                    if isinstance(self.lp_weight, (float, int))
                    else self.lp_weight(x) * np.power(np.abs(vector(x)), self.p)
                ),
                start,
                end,
            )
            res = np.sqrt(integral[0]).flatten()

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
                data_matrix * self.lp_weight
                if isinstance(self.lp_weight, (float, int))
                else self.lp_weight(vector.grid_points[0]) * data_matrix
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
            raise NotImplementedError(
                f"LpNorm not implemented for type {type(vector)}",
            )

        if len(res) == 1:
            return res[0]  # type: ignore[no-any-return]

        return res  # type: ignore[no-any-return]


def weighted_lp_norm(
    vector: Union[NDArrayFloat, FData],
    *,
    p: float,
    vector_norm: Union[Norm[NDArrayFloat], float, None] = None,
    lp_weight: Union[
        Callable[[NDArrayFloat], NDArrayFloat],
        float,
        None,
    ] = None,
) -> NDArrayFloat:
    return WeightedLpNorm(p=p, vector_norm=vector_norm, lp_weight=lp_weight)(vector)
