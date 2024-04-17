from __future__ import annotations

from typing import Callable, Iterable
from functools import singledispatch
import numpy as np


from ..misc.validation import validate_random_state
from ..representation import FData, FDataGrid, FDataBasis, FDataIrregular
from ..typing._base import RandomState, RandomStateLike
from ..typing._numpy import NDArrayFloat


def _irregular_sample_from_callable(
    funcs: Iterable[Callable[[NDArrayFloat], NDArrayFloat]],
    points_matrix: NDArrayFloat,
) -> FDataIrregular:
    """Sample from a list of functions at irregular points.

    Args:
        funcs: List of functions to sample.
        points_matrix: of shape (n_funcs, n_points_per_function). Points where
            to measure each function sample.
    """
    assert points_matrix.ndim == 2
    n_points_per_curve = points_matrix.shape[1]
    total_n_points = points_matrix.shape[0] * n_points_per_curve
    return FDataIrregular(
        points=points_matrix.reshape(-1),
        start_indices=np.array(range(0, total_n_points, n_points_per_curve)),
        values=np.concatenate([
            func(func_points).reshape(-1)
            for func, func_points in zip(funcs, points_matrix)
        ]),
    )


def irregular_sample(
    fdata: FDataGrid | FDataBasis,
    n_points_per_curve: int,
    random_state: RandomStateLike = None,
) -> FDataIrregular:
    """Irregularly sample from a FDataGrid or FDataBasis object.

    Only implemented for 1D domains and codomains. The points are selected at
    random (uniformly) from the domain of the input object.

    Args:
        fdata: Functional data object to sample from.
        n_points_per_curve: Number of points to sample per curve.
    """
    if fdata.dim_domain != 1 or fdata.dim_codomain != 1:
        raise NotImplementedError(
            "Only implemented for 1D domains and codomains.",
        )

    random_state = validate_random_state(random_state)

    points_matrix = _irregular_sample_points_matrix(
        fdata,
        n_points_per_curve=n_points_per_curve,
        random_state=random_state,
    )
    return _irregular_sample_from_callable(
        funcs=fdata,
        points_matrix=points_matrix,
    )


@singledispatch
def _irregular_sample_points_matrix(
    fdata: FDataGrid | FDataBasis,
    n_points_per_curve: int,
    random_state: RandomState,
) -> NDArrayFloat:
    raise NotImplementedError(
        "Only implemented for FDataGrid and FDataBasis.",
    )


@_irregular_sample_points_matrix.register
def _irregular_sample_points_matrix_fdatagrid(
    fdata: FDataGrid,
    n_points_per_curve: int,
    random_state: RandomState,
) -> NDArrayFloat:
    return random_state.choice(
        fdata.grid_points[0],  # This only works for 1D domains
        size=(fdata.n_samples, n_points_per_curve),
        replace=True,
    )


@_irregular_sample_points_matrix.register
def _irregular_sample_points_matrix_fdatabasis(
    fdata: FDataBasis,
    n_points_per_curve: int,
    random_state: RandomState,
) -> NDArrayFloat:
    return random_state.uniform(
        *fdata.domain_range[0],  # This only works for 1D domains
        size=(fdata.n_samples, n_points_per_curve),
    )
