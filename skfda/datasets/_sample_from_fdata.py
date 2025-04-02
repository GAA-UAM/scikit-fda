from __future__ import annotations

from functools import singledispatch
from typing import List, Tuple

import numpy as np

from .._utils import _cartesian_product, _to_grid_points
from ..misc.validation import validate_random_state
from ..representation import FDataBasis, FDataGrid, FDataIrregular
from ..typing._base import RandomState, RandomStateLike
from ..typing._numpy import NDArrayFloat, NDArrayInt


def irregular_sample(
    fdata: FDataBasis | FDataGrid | FDataIrregular,
    n_points_per_curve: int | NDArrayInt,
    random_state: RandomStateLike = None,
) -> FDataIrregular:
    """Irregularly sample from a FDataGrid or FDataBasis object.

    The points are selected at random (uniformly) from the domain of the input
    object.
    If the input is an FDataGrid or an FDataIrregular, the points are selected
    uniformly from the finite grid points of the object. If the input is an
    FDataBasis, the points are selected from the rectangular domain of the
    with a uniform (continuous) distribution.

    Args:
        fdata: Functional data object to sample from.
        n_points_per_curve: Number of points to sample per curve. If fdata is
            an FDataGrid or an FDataIrregular and a sample has less points than
            specified in n_points_per_curve, the sample will have the same
            number of points as before.
        random_state: Random state to control the random number generation.
    """
    random_state = validate_random_state(random_state)
    if isinstance(n_points_per_curve, int):
        n_points_per_curve = np.full(fdata.n_samples, n_points_per_curve)

    points_list, start_indices = (
        _irregular_sample_points_list(
            fdata,
            n_points_per_curve=n_points_per_curve,
            random_state=random_state,
        )
    )

    return FDataIrregular(
        points=np.concatenate(points_list),
        start_indices=start_indices,
        values=np.concatenate([
            func(func_points)[0, :, :]
            for func, func_points in zip(fdata, points_list)
        ]),
    )


def _start_indices(n_points_per_curve: NDArrayInt) -> NDArrayInt:
    return np.cumsum(
        np.concatenate([
            np.zeros(1, dtype=int),
            n_points_per_curve[:-1],
        ]),
    )


@singledispatch
def _irregular_sample_points_list(
    fdata: FDataBasis | FDataGrid | FDataIrregular,
    n_points_per_curve: NDArrayInt,
    random_state: RandomState,
) -> Tuple[List[NDArrayFloat], NDArrayInt]:
    """Return a list of points and the start indices for each curve.

    The points are selected at random (uniformly) from the domain of the input.

    Returns:
        points_list: List of points for each curve.
        start_indices: Start indices for each curve.
    """
    raise NotImplementedError(
        "Only implemented for FDataBasis, FDataGrid and FDataIrregular.",
    )


@_irregular_sample_points_list.register
def _irregular_sample_points_matrix_fdatagrid(
    fdata: FDataGrid,
    n_points_per_curve: NDArrayInt,
    random_state: RandomState,
) -> Tuple[List[NDArrayFloat], NDArrayInt]:
    all_points_single_function = _cartesian_product(
        _to_grid_points(fdata.grid_points),
    )
    flat_points = np.tile(
        all_points_single_function, (fdata.n_samples, 1),
    )
    n_points_per_curve = np.minimum(
        n_points_per_curve,
        len(flat_points),
    )
    return (
        [
            random_state.permutation(flat_points)[:n_points]
            for n_points in n_points_per_curve
        ],
        _start_indices(n_points_per_curve),
    )


@_irregular_sample_points_list.register
def _irregular_sample_points_matrix_fdatairregular(
    fdata: FDataIrregular,
    n_points_per_curve: NDArrayInt,
    random_state: RandomState,
) -> Tuple[List[NDArrayFloat], NDArrayInt]:
    original_n_points_per_curve = np.diff(
        np.concatenate([fdata.start_indices, [len(fdata.points)]]),
    )
    n_points_per_curve = np.minimum(
        n_points_per_curve,
        original_n_points_per_curve,
    )
    return (
        [
            random_state.permutation(curve_points)[
                :min(n_points, len(curve_points)),
            ]
            for n_points, curve_points in zip(
                n_points_per_curve,
                np.split(fdata.points, fdata.start_indices[1:]),
            )
        ],
        _start_indices(n_points_per_curve),
    )


@_irregular_sample_points_list.register
def _irregular_sample_points_matrix_fdatabasis(
    fdata: FDataBasis,
    n_points_per_curve: NDArrayInt,
    random_state: RandomState,
) -> Tuple[List[NDArrayFloat], NDArrayInt]:
    len_points = np.sum(n_points_per_curve)
    separate_coordinate_points = [
        random_state.uniform(*domain_range_coordinate, size=(len_points))
        for domain_range_coordinate in fdata.domain_range
    ]
    start_indices = _start_indices(n_points_per_curve)
    points = np.stack(separate_coordinate_points, axis=1)
    return (np.split(points, start_indices[1:]), start_indices)
