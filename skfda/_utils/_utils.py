"""Module with generic methods."""

from __future__ import annotations

import functools
import numbers
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Iterable,
    List,
    Optional,
    Sequence,
    Sized,
    Tuple,
    Type,
    TypeVar,
    Union,
    cast,
    overload,
)

import numpy as np
import scipy.integrate
from pandas.api.indexers import check_array_indexer
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.multiclass import check_classification_targets
from typing_extensions import Literal, ParamSpec, Protocol

from ..typing._base import GridPoints, GridPointsLike
from ..typing._numpy import NDArrayAny, NDArrayFloat, NDArrayInt, NDArrayStr
from ._sklearn_adapter import BaseEstimator

ArrayDTypeT = TypeVar("ArrayDTypeT", bound="np.generic")

if TYPE_CHECKING:
    from ..representation import FData, FDataBasis, FDataGrid
    from ..representation.basis import Basis
    from ..representation.extrapolation import ExtrapolationLike

    T = TypeVar("T", bound=FData)

    Input = TypeVar("Input", bound=Union[FData, NDArrayFloat])
    Output = TypeVar("Output", bound=Union[FData, NDArrayFloat])
    Target = TypeVar("Target", bound=NDArrayInt)

    AcceptedExtrapolation = Union[ExtrapolationLike, None, Literal["default"]]


_MapAcceptableSelf = TypeVar(
    "_MapAcceptableSelf",
    bound="_MapAcceptable",
)


class _MapAcceptable(Protocol, Sized):

    def __getitem__(
        self: _MapAcceptableSelf,
        __key: Union[slice, NDArrayInt],  # noqa: WPS112
    ) -> _MapAcceptableSelf:
        pass

    @property
    def nbytes(self) -> int:
        pass


_MapAcceptableT = TypeVar(
    "_MapAcceptableT",
    bound=_MapAcceptable,
    contravariant=True,
)
MapFunctionT = TypeVar("MapFunctionT", covariant=True)
P = ParamSpec("P")


class _MapFunction(Protocol[_MapAcceptableT, P, MapFunctionT]):
    """Protocol for functions that can be mapped over several arrays."""

    def __call__(
        self,
        *args: _MapAcceptableT,
        **kwargs: P.kwargs,
    ) -> MapFunctionT:
        pass


class _PairwiseFunction(Protocol[_MapAcceptableT, P, MapFunctionT]):
    """Protocol for pairwise array functions."""

    def __call__(
        self,
        __arg1: _MapAcceptableT,  # noqa: WPS112
        __arg2: _MapAcceptableT,  # noqa: WPS112
        **kwargs: P.kwargs,  # type: ignore[name-defined]
    ) -> MapFunctionT:
        pass


def _to_grid(
    X: FData,
    y: FData,
    eval_points: Optional[NDArrayFloat] = None,
) -> Tuple[FDataGrid, FDataGrid]:
    """Transform a pair of FDatas in grids to perform calculations."""
    from .. import FDataGrid
    x_is_grid = isinstance(X, FDataGrid)
    y_is_grid = isinstance(y, FDataGrid)

    if eval_points is not None:
        X = X.to_grid(eval_points)
        y = y.to_grid(eval_points)
    elif x_is_grid and not y_is_grid:
        y = y.to_grid(X.grid_points[0])
    elif not x_is_grid and y_is_grid:
        X = X.to_grid(y.grid_points[0])
    elif not x_is_grid and not y_is_grid:
        X = X.to_grid()
        y = y.to_grid()

    return X, y


def _same_domain(fd: Union[Basis, FData], fd2: Union[Basis, FData]) -> bool:
    """Check if the domain range of two objects is the same."""
    return np.array_equal(fd.domain_range, fd2.domain_range)


def nquad_vec(
    func: Callable[[NDArrayFloat], NDArrayFloat],
    ranges: Sequence[Tuple[float, float]],
) -> NDArrayFloat:
    """Perform multiple integration of vector valued functions."""
    initial_depth = len(ranges) - 1

    def integrate(*args: Any, depth: int) -> NDArrayFloat:  # noqa: WPS430

        if depth == 0:
            f = functools.partial(func, *args)
        else:
            f = functools.partial(integrate, *args, depth=depth - 1)

        return scipy.integrate.quad_vec(  # type: ignore[no-any-return]
            f,
            *ranges[initial_depth - depth],
        )[0]

    return integrate(depth=initial_depth)


def _map_in_batches(
    function: _MapFunction[_MapAcceptableT, P, np.typing.NDArray[ArrayDTypeT]],
    arguments: Tuple[_MapAcceptableT, ...],
    indexes: Tuple[NDArrayInt, ...],
    memory_per_batch: Optional[int] = None,
    *args: P.args,  # Should be empty
    **kwargs: P.kwargs,
) -> np.typing.NDArray[ArrayDTypeT]:
    """
    Map a function over samples of FData or ndarray tuples efficiently.

    This function prevents a large set of indexes to use all available
    memory and hang the PC.

    """
    if memory_per_batch is None:
        # 256MB is not too big
        memory_per_batch = 256 * 1024 * 1024  # noqa: WPS432

    memory_per_element = sum(a.nbytes // len(a) for a in arguments)
    n_elements_per_batch_allowed = memory_per_batch // memory_per_element
    if n_elements_per_batch_allowed < 1:
        raise ValueError("Too few memory allowed for the operation")

    n_indexes = len(indexes[0])

    assert all(n_indexes == len(i) for i in indexes)

    batches: List[np.typing.NDArray[ArrayDTypeT]] = []

    for pos in range(0, n_indexes, n_elements_per_batch_allowed):
        batch_args = tuple(
            a[i[pos:pos + n_elements_per_batch_allowed]]
            for a, i in zip(arguments, indexes)
        )

        batches.append(function(*batch_args, **kwargs))

    return np.concatenate(batches, axis=0)


def _pairwise_symmetric(
    function: _PairwiseFunction[
        _MapAcceptableT,
        P,
        np.typing.NDArray[ArrayDTypeT],
    ],
    arg1: _MapAcceptableT,
    arg2: Optional[_MapAcceptableT] = None,
    memory_per_batch: Optional[int] = None,
    *args: P.args,  # Should be empty
    **kwargs: P.kwargs,
) -> np.typing.NDArray[ArrayDTypeT]:
    """Compute pairwise a commutative function."""
    def map_function(
        *args: _MapAcceptableT,
        **kwargs: P.kwargs,
    ) -> np.typing.NDArray[ArrayDTypeT]:
        """Just to keep Mypy happy."""
        return function(args[0], args[1], **kwargs)

    dim1 = len(arg1)
    if arg2 is None or arg2 is arg1:
        triu_indices = np.triu_indices(dim1)

        triang_vec = _map_in_batches(
            map_function,
            (arg1, arg1),
            triu_indices,
            memory_per_batch,
            **kwargs,  # type: ignore[arg-type]
        )

        matrix = np.empty((dim1, dim1), dtype=triang_vec.dtype)

        # Set upper matrix
        matrix[triu_indices] = triang_vec

        # Set lower matrix
        matrix[(triu_indices[1], triu_indices[0])] = triang_vec

        return matrix

    dim2 = len(arg2)
    indices = np.indices((dim1, dim2))

    vec = _map_in_batches(
        map_function,
        (arg1, arg2),
        (indices[0].ravel(), indices[1].ravel()),
        memory_per_batch=memory_per_batch,
        **kwargs,  # type: ignore[arg-type]
    )

    return np.reshape(vec, (dim1, dim2))


def _int_to_real(array: Union[NDArrayInt, NDArrayFloat]) -> NDArrayFloat:
    """Convert integer arrays to floating point."""
    if np.issubdtype(array.dtype, np.integer):
        return array.astype(np.float64)

    assert np.issubdtype(array.dtype, np.floating)
    return cast(NDArrayFloat, array)


def _check_array_key(array: NDArrayAny, key: Any) -> Any:
    """Check a getitem key."""
    key = check_array_indexer(array, key)
    if isinstance(key, tuple):
        non_ellipsis = [i for i in key if i is not Ellipsis]
        if len(non_ellipsis) > 1:
            raise KeyError(key)
        key = non_ellipsis[0]

    if isinstance(key, numbers.Integral):  # To accept also numpy ints
        key = int(key)
        if key < 0:
            key = len(array) + key

        if not 0 <= key < len(array):
            raise IndexError("index out of bounds")

        return slice(key, key + 1)

    return key


def _check_estimator(estimator: Type[BaseEstimator]) -> None:
    from sklearn.utils.estimator_checks import (
        check_get_params_invariance,
        check_set_params,
    )

    name = estimator.__name__
    instance = estimator()
    check_get_params_invariance(name, instance)
    check_set_params(name, instance)


def _classifier_get_classes(
    y: NDArrayStr | NDArrayInt,
) -> Tuple[NDArrayStr | NDArrayInt, NDArrayInt]:

    check_classification_targets(y)

    le = LabelEncoder()
    y_ind = le.fit_transform(y)

    classes = le.classes_

    if classes.size < 2:
        raise ValueError(
            f'The number of classes has to be greater than'
            f'one; got {classes.size} class',
        )
    return classes, y_ind


def function_to_fdatabasis(
    f: Callable[[NDArrayFloat], NDArrayFloat],
    new_basis: Basis,
) -> FDataBasis:
    """Express a math function as a FDataBasis with a given basis.

    Args:
        f: math function.
        new_basis: the basis of the output.

    Returns:
        FDataBasis: FDataBasis with calculated coefficients and the new
        basis.
    """
    from .. import FDataBasis  # noqa: WPS442
    from ..misc._math import inner_product_matrix

    if isinstance(f, FDataBasis) and f.basis == new_basis:
        return f.copy()

    inner_prod = inner_product_matrix(
        new_basis,
        f,
        _domain_range=new_basis.domain_range,
    )

    gram_matrix = new_basis.gram_matrix()

    coefs = np.linalg.solve(gram_matrix, inner_prod)

    return FDataBasis(new_basis, coefs.T)
