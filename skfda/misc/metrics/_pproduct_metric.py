"""Implementation of Lp norms."""

import math
from builtins import isinstance
from typing import Union, Callable, List

import numpy as np
from typing_extensions import Final

from ...representation import FData, FDataBasis, FDataGrid
from ...typing._metric import Norm
from ...typing._numpy import NDArrayFloat


class PProductMetric(Norm):
    def __init__(
        self,
        p: float,
        norms: Union[List[Norm], Norm, None] = None,
        weights: Union[
            NDArrayFloat,
            float,
            None,
        ] = None,
    ) -> None:

        # Checks that the lp normed is well defined
        if not np.isinf(p) and p < 1:
            raise ValueError(f"p (={p}) must be equal or greater than 1.")

        self.p = p
        self.norms = norms
        self.weights = weights

    def __repr__(self) -> str:
        return f"{type(self).__name__}(" f"p={self.p}, measure={self.weights}"

    def __call__(self, vector: Union[NDArrayFloat, FData]) -> NDArrayFloat:
        """Compute the Lp norm of a functional data object."""
        from ..metrics import l2_norm

        weights = self.weights if self.weights else 1.0
        if isinstance(vector, np.ndarray):
            if isinstance(weights, (float, int)):
                vector = vector * weights
            return np.linalg.norm(  # type: ignore[no-any-return]
                vector,
                ord=self.p,
                axis=-1,
            )

        # Special case, the inner product is heavily optimized: TODO: Is it interesting to optimize the inner product with weights and ...
        """ if self.p == 2:
            return np.sqrt(weighted_inner_product(vector, vector, weights)) """

        D = vector.dim_codomain

        norms = self.norms if self.norms else [l2_norm]*D
        if isinstance(norms, Norm):
            norms = [norms] * D
        elif isinstance(norms, list):
            if len(norms) != D:
                raise ValueError(
                    f"Number of norms ({len(norms)}) does not match the number of dimensions ({D}).",
                )
        if isinstance(weights, (float, int)):
            weights = np.full(D, weights)
        elif isinstance(weights, np.ndarray):
            if len(weights) != D:
                raise ValueError(
                    f"Number of weights ({len(weights)}) does not match the number of dimensions ({D}).",
                )

        if isinstance(vector, FDataBasis):
            raise NotImplementedError

        elif isinstance(vector, FDataGrid):
            data_matrix = vector.data_matrix
            # apply each norm to each of the dimensions and sum them
            values = np.array([norms[i](FDataGrid(data_matrix=data_matrix[:,:,i], grid_points=vector.grid_points)) for i in range(D)])
            res = np.sum(
                np.power(values, self.p) * weights,
                axis=0,
            )
        else:
            raise NotImplementedError(
                f"WeightedLpNorm not implemented for type {type(vector)}",
            )

        if len(res) == 1:
            return res[0]  # type: ignore[no-any-return]

        return res  # type: ignore[no-any-return]


def pproduct_metric(
    vector: Union[NDArrayFloat, FData],
    *,
    p: float,
    norms: Union[List[Norm], Norm, None] = None,
    weights: Union[
        NDArrayFloat,
        float,
        None,
    ] = None,
) -> NDArrayFloat:
    return PProductMetric(p=p, norms=norms, weights=weights)(vector)
