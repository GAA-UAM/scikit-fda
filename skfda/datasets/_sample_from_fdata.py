from __future__ import annotations

from typing import List, Tuple
from functools import singledispatch
import numpy as np


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

    Only implemented for 1D domains and codomains. The points are selected at
    random (uniformly) from the domain of the input object.

    Args:
        fdata: Functional data object to sample from.
        n_points_per_curve: Number of points to sample per curve. If fdata is
        an FDataGrid or an FDataIrregular and a sample has less points than
        specified in n_points_per_curve, the sample will have the same number
        of points as before.
        random_state: Random state to control the random number generation.
    """
    if fdata.dim_domain != 1 or fdata.dim_codomain != 1:
        raise NotImplementedError(
            "Only implemented for 1D domains and codomains.",
        )

    random_state = validate_random_state(random_state)
    if isinstance(n_points_per_curve, int):
        n_points_per_curve = np.full(fdata.n_samples, n_points_per_curve)

    points_list, n_points_per_curve = (
        _irregular_sample_points_list(
            fdata,
            n_points_per_curve=n_points_per_curve,
            random_state=random_state,
        )
    )
    values = np.concatenate([
        func(func_points).reshape(-1)
        for func, func_points in zip(fdata, points_list)
    ])
    return FDataIrregular(
        points=np.concatenate(points_list),
        start_indices=np.cumsum(
            np.concatenate([
                np.zeros(1, dtype=int),
                n_points_per_curve[:-1],
            ]),
        ),
        values=np.concatenate([
            func(func_points).reshape(-1)
            for func, func_points in zip(fdata, points_list)
        ]),
    )


@singledispatch
def _irregular_sample_points_list(
    fdata: FDataBasis | FDataGrid | FDataIrregular,
    n_points_per_curve: NDArrayInt,
    random_state: RandomState,
) -> Tuple[List[NDArrayFloat], NDArrayInt]:
    raise NotImplementedError(
        "Only implemented for FDataGrid and FDataBasis.",
    )


@_irregular_sample_points_list.register
def _irregular_sample_points_matrix_fdatagrid(
    fdata: FDataGrid,
    n_points_per_curve: NDArrayInt,
    random_state: RandomState,
) -> List[NDArrayFloat]:
    # This only works for 1D domains
    n_points_per_curve = np.minimum(
        n_points_per_curve,
        len(fdata.grid_points[0]),
    )
    return [
        random_state.choice(
            fdata.grid_points[0].reshape(-1),
            size=n_points,
            replace=False,
        )
        for n_points in n_points_per_curve
    ], n_points_per_curve


@_irregular_sample_points_list.register
def _irregular_sample_points_matrix_fdatairregular(
    fdata: FDataIrregular,
    n_points_per_curve: NDArrayInt,
    random_state: RandomState,
) -> List[NDArrayFloat]:
    # This only works for 1D domains
    original_n_points_per_curve = np.diff(
        np.concatenate([fdata.start_indices, [len(fdata.points)]]),
    )
    n_points_per_curve = np.minimum(
        n_points_per_curve,
        original_n_points_per_curve,
    )
    return [
        random_state.choice(
            curve_points.reshape(-1),
            size=min(n_points, len(curve_points)),
            replace=False,
        )
        for n_points, curve_points in zip(
            n_points_per_curve,
            np.split(fdata.points, fdata.start_indices[1:]),
        )
    ], n_points_per_curve


@_irregular_sample_points_list.register
def _irregular_sample_points_matrix_fdatabasis(
    fdata: FDataBasis,
    n_points_per_curve: NDArrayInt,
    random_state: RandomState,
) -> List[NDArrayFloat]:
    # This only works for 1D domains
    return [
        random_state.uniform(
            *fdata.domain_range[0],
            size=(n_points),
        )
        for n_points in n_points_per_curve
    ], n_points_per_curve
