from functools import singledispatch
from typing import Any, Callable, TypeVar

import numpy as np

from ....representation import FDataGrid
from ....typing._numpy import NDArrayAny, NDArrayFloat

dtype_bound = np.number
X_T = TypeVar("X_T")
dtype_X_T = TypeVar("dtype_X_T", bound="dtype_bound[Any]")
dtype_y_T = TypeVar("dtype_y_T", bound="dtype_bound[Any]")

depX_T = TypeVar("depX_T", bound=NDArrayAny)
depy_T = TypeVar("depy_T", bound=NDArrayAny)

_DependenceMeasure = Callable[
    [depX_T, depy_T],
    NDArrayFloat,
]


@singledispatch
def _compute_dependence(
    X: X_T,
    y: np.typing.NDArray[dtype_y_T],
    *,
    dependence_measure: _DependenceMeasure[
        np.typing.NDArray[dtype_X_T],
        np.typing.NDArray[dtype_y_T],
    ],
) -> X_T:
    """
    Compute dependence between points and target.

    Computes the dependence of each point in each trajectory in X with the
    corresponding class label in Y.

    """
    from dcor import rowwise

    assert isinstance(X, np.ndarray)
    X_ndarray = X

    # Shape without number of samples and codomain dimension
    input_shape = X_ndarray.shape[1:-1]

    # Move n_samples to the end
    # The shape is now input_shape + n_samples + n_output
    X_ndarray = np.moveaxis(X_ndarray, 0, -2)

    # Join input in a list for rowwise
    X_ndarray = X_ndarray.reshape(-1, X_ndarray.shape[-2], X_ndarray.shape[-1])

    if y.ndim == 1:
        y = np.atleast_2d(y).T

    Y = np.array([y] * len(X_ndarray))

    dependence_results = rowwise(  # type: ignore[no-untyped-call]
        dependence_measure,
        X_ndarray,
        Y,
    )

    return dependence_results.reshape(  # type: ignore[no-any-return]
        input_shape,
    )


@_compute_dependence.register
def _compute_dependence_fdatagrid(
    X: FDataGrid,
    y: np.typing.NDArray[dtype_y_T],
    *,
    dependence_measure: _DependenceMeasure[
        np.typing.NDArray[dtype_X_T],
        np.typing.NDArray[dtype_y_T],
    ],
) -> FDataGrid:

    return X.copy(
        data_matrix=_compute_dependence(
            X.data_matrix,
            y,
            dependence_measure=dependence_measure,
        ),
        coordinate_names=("relevance",),
        sample_names=("relevance function",),
    )
