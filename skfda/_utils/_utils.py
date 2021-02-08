"""Module with generic methods."""

from __future__ import annotations

import functools
import numbers
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    List,
    Optional,
    Sequence,
    Tuple,
    TypeVar,
    Union,
    cast,
)

import numpy as np
import scipy.integrate
from numpy import ndarray
from pandas.api.indexers import check_array_indexer
from sklearn.base import clone

from ..representation._typing import (
    DomainRange,
    DomainRangeLike,
    GridPoints,
    GridPointsLike,
)
from ..representation.evaluator import Evaluator

RandomStateLike = Optional[Union[int, np.random.RandomState]]

if TYPE_CHECKING:
    from ..exploratory.depth import Depth
    from ..representation import FData
    from ..representation.basis import Basis
    from ..representation.grid import FDataGrid

T = TypeVar("T", contravariant=True)


class _FDataCallable():

    def __init__(self, function, *, domain_range, n_samples=1):

        self.function = function
        self.domain_range = domain_range
        self.n_samples = n_samples

    def __call__(self, *args, **kwargs):

        return self.function(*args, **kwargs)

    def __len__(self):

        return self.n_samples

    def __getitem__(self, key):

        def new_function(*args, **kwargs):
            return self.function(*args, **kwargs)[key]

        tmp = np.empty(self.n_samples)
        new_nsamples = len(tmp[key])

        return _FDataCallable(
            new_function,
            domain_range=self.domain_range,
            n_samples=new_nsamples,
        )


def check_is_univariate(fd):
    """Check if an FData is univariate and raises an error.

    Args:
        fd (:class:`~skfda.FData`): Functional object to check if is
            univariate.

    Raises:
        ValueError: If it is not univariate, i.e., `fd.dim_domain != 1` or
            `fd.dim_codomain != 1`.

    """
    if fd.dim_domain != 1 or fd.dim_codomain != 1:
        raise ValueError(
            f'The functional data must be univariate, i.e., '
            f'with dim_domain=1 '
            f'{"" if fd.dim_domain == 1 else f"(now {fd.dim_domain}) "}'
            f'and dim_codomain=1 '
            f'{"" if fd.dim_codomain == 1 else f"(now {fd.dim_codomain})"}',
        )


def _to_grid(X, y, eval_points=None):
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


def _to_grid_points(grid_points_like: GridPointsLike) -> GridPoints:
    """Convert to grid points.

    If the original list is one-dimensional (e.g. [1, 2, 3]), return list to
    array (in this case [array([1, 2, 3])]).

    If the original list is two-dimensional (e.g. [[1, 2, 3], [4, 5]]), return
    a list containing other one-dimensional arrays (in this case
    [array([1, 2, 3]), array([4, 5])]).

    In any other case the behaviour is unespecified.

    """
    unidimensional = False

    try:
        iter(grid_points_like)
    except TypeError:
        grid_points_like = [grid_points_like]

    try:
        iter(grid_points_like[0])
    except TypeError:
        unidimensional = True

    if unidimensional:
        return (_int_to_real(np.asarray(grid_points_like)),)

    return tuple(_int_to_real(np.asarray(i)) for i in grid_points_like)


def _to_domain_range(sequence: DomainRangeLike) -> DomainRange:
    """Convert sequence to a proper domain range."""
    seq_aux = cast(
        Sequence[Sequence[float]],
        (sequence,) if isinstance(sequence[0], numbers.Real) else sequence,
    )

    tuple_aux = tuple(tuple(s) for s in seq_aux)

    if not all(len(s) == 2 and s[0] <= s[1] for s in tuple_aux):
        raise ValueError(
            "Domain intervals should have 2 bounds for "
            "dimension: (lower, upper).",
        )

    return cast(DomainRange, tuple_aux)


def _to_array_maybe_ragged(array, *, row_shape=None):
    """
    Convert to an array where each element may or may not be of equal length.

    If each element is of equal length the array is multidimensional.
    Otherwise it is a ragged array.

    """
    def convert_row(row):
        r = np.array(row)

        if row_shape is not None:
            r = r.reshape(row_shape)

        return r

    array_list = [convert_row(a) for a in array]
    shapes = [a.shape for a in array_list]

    if all(s == shapes[0] for s in shapes):
        return np.array(array_list)

    res = np.empty(len(array_list), dtype=np.object_)

    for i, a in enumerate(array_list):
        res[i] = a

    return res


def _cartesian_product(axes, flatten=True, return_shape=False):
    """Compute the cartesian product of the axes.

    Computes the cartesian product of the axes and returns a numpy array of
    1 dimension with all the possible combinations, for an arbitrary number of
    dimensions.

    Args:
        axes (array_like): List with axes.
        flatten: Boolean flag. True if the product is flattened.
        return_shape: Boolean flag. True if the shape is returned.

    Returns:
        (np.ndarray): Numpy 2-D array with all the possible combinations.
        The entry (i,j) represent the j-th coordinate of the i-th point.

    Examples:
        >>> from skfda._utils import _cartesian_product
        >>> axes = [[0,1],[2,3]]
        >>> _cartesian_product(axes)
        array([[0, 2],
               [0, 3],
               [1, 2],
               [1, 3]])

        >>> axes = [[0,1],[2,3],[4]]
        >>> _cartesian_product(axes)
        array([[0, 2, 4],
               [0, 3, 4],
               [1, 2, 4],
               [1, 3, 4]])

        >>> axes = [[0,1]]
        >>> _cartesian_product(axes)
        array([[0],
               [1]])
    """
    cartesian = np.stack(np.meshgrid(*axes, indexing='ij'), -1)

    shape = cartesian.shape

    if flatten:
        cartesian = cartesian.reshape(-1, len(axes))

    if return_shape:
        return cartesian, shape

    return cartesian


def _same_domain(fd: Union[Basis, FData], fd2: Union[Basis, FData]) -> bool:
    """Check if the domain range of two objects is the same."""
    return np.array_equal(fd.domain_range, fd2.domain_range)


def _reshape_eval_points(
    eval_points: np.ndarray,
    *,
    aligned: bool,
    n_samples: int,
    dim_domain: int,
) -> np.ndarray:
    """Convert and reshape the eval_points to ndarray.

    Args:
        eval_points: Evaluation points to be reshaped.
        aligned: Boolean flag. True if all the samples
            will be evaluated at the same evaluation_points.
        n_samples: Number of observations.
        dim_domain: Dimension of the domain.

    Returns:
        Numpy array with the eval_points, if
        evaluation_aligned is True with shape `number of evaluation points`
        x `dim_domain`. If the points are not aligned the shape of the
        points will be `n_samples` x `number of evaluation points`
        x `dim_domain`.

    """
    if aligned:
        eval_points = np.asarray(eval_points)
    else:
        eval_points = _to_array_maybe_ragged(
            eval_points, row_shape=(-1, dim_domain),
        )

    # Case evaluation of a single value, i.e., f(0)
    # Only allowed for aligned evaluation
    if aligned and (
        eval_points.shape == (dim_domain,)
        or (eval_points.ndim == 0 and dim_domain == 1)
    ):
        eval_points = np.array([eval_points])

    if aligned:  # Samples evaluated at same eval points

        eval_points = eval_points.reshape((
            eval_points.shape[0],
            dim_domain,
        ))

    else:  # Different eval_points for each sample

        if eval_points.shape[0] != n_samples:
            raise ValueError(
                f'eval_points should be a list '
                f'of length {n_samples} with the '
                f'evaluation points for each sample.',
            )

    return eval_points


def _one_grid_to_points(axes, *, dim_domain):
    """
    Convert a list of ndarrays, one per domain dimension, in the points.

    Returns also the shape containing the information of how each point
    is formed.
    """
    axes = _to_grid_points(axes)

    if len(axes) != dim_domain:
        raise ValueError(
            f'Length of axes should be {dim_domain}',
        )

    cartesian, shape = _cartesian_product(axes, return_shape=True)

    # Drop domain size dimension, as it is not needed to reshape the output
    shape = shape[:-1]

    return cartesian, shape


def _evaluate_grid(
    axes: Sequence[np.ndarray],
    *,
    evaluate_method: Any,
    n_samples: int,
    dim_domain: int,
    dim_codomain: int,
    extrapolation: Optional[Union[str, Evaluator]] = None,
    aligned: bool = True,
) -> np.ndarray:
    """Evaluate the functional object in the cartesian grid.

    This method is called internally by :meth:`evaluate` when the argument
    `grid` is True.

    Evaluates the functional object in the grid generated by the cartesian
    product of the axes. The length of the list of axes should be equal
    than the domain dimension of the object.

    If the list of axes has lengths :math:`n_1, n_2, ..., n_m`, where
    :math:`m` is equal than the dimension of the domain, the result of the
    evaluation in the grid will be a matrix with :math:`m+1` dimensions and
    shape :math:`n_{samples} x n_1 x n_2 x ... x n_m`.

    If `aligned` is false each sample is evaluated in a
    different grid, and the list of axes should contain a list of axes for
    each sample.

    If the domain dimension is 1, the result of the behaviour of the
    evaluation will be the same than :meth:`evaluate` without the grid
    option, but with worst performance.

    Args:
        axes: List of axes to generated the grid where the
            object will be evaluated.
        extrapolation: Controls the
            extrapolation mode for elements outside the domain range. By
            default it is used the mode defined during the instance of the
            object.
        aligned: If False evaluates each sample
            in a different grid.
        evaluate_method: method to use to evaluate the points
        n_samples: number of samples
        dim_domain: dimension of the domain
        dim_codomain: dimensions of the codomain

    Returns:
        Numpy array with dim_domain + 1 dimensions with
            the result of the evaluation.

    Raises:
        ValueError: If there are a different number of axes than the domain
            dimension.

    """
    # Compute intersection points and resulting shapes
    if aligned:

        eval_points, shape = _one_grid_to_points(axes, dim_domain=dim_domain)

    else:

        axes = list(axes)

        if len(axes) != n_samples:
            raise ValueError(
                "A list of axis per sample should be provided",
            )

        eval_points, shape = zip(
            *[_one_grid_to_points(a, dim_domain=dim_domain) for a in axes],
        )

    eval_points = _to_array_maybe_ragged(eval_points)

    # Evaluate the points
    res = evaluate_method(
        eval_points,
        extrapolation=extrapolation,
        aligned=aligned,
    )

    # Reshape the result
    if aligned:

        res = res.reshape(
            [n_samples] + list(shape) + [dim_codomain],
        )

    else:

        res = _to_array_maybe_ragged([
            r.reshape(list(s) + [dim_codomain])
            for r, s in zip(res, shape)
        ])

    return res


def nquad_vec(
    func: Callable[[np.ndarray], np.ndarray],
    ranges: Sequence[Tuple[float, float]],
) -> np.ndarray:
    """Perform multiple integration of vector valued functions."""
    initial_depth = len(ranges) - 1

    def integrate(*args: Any, depth: int) -> np.ndarray:  # noqa: WPS430

        if depth == 0:
            f = functools.partial(func, *args)
        else:
            f = functools.partial(integrate, *args, depth=depth - 1)

        return scipy.integrate.quad_vec(f, *ranges[initial_depth - depth])[0]

    return integrate(depth=initial_depth)


def _map_in_batches(
    function: Callable[..., np.ndarray],
    arguments: Tuple[Union[FData, np.ndarray], ...],
    indexes: Tuple[np.ndarray, ...],
    memory_per_batch: Optional[int] = None,
    **kwargs: Any,
) -> np.ndarray:
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

    batches: List[np.ndarray] = []

    for pos in range(0, n_indexes, n_elements_per_batch_allowed):
        batch_args = tuple(
            a[i[pos:pos + n_elements_per_batch_allowed]]
            for a, i in zip(arguments, indexes)
        )

        batches.append(function(*batch_args, **kwargs))

    return np.concatenate(batches, axis=0)


def _pairwise_symmetric(
    function: Callable[..., np.ndarray],
    arg1: Union[FData, np.ndarray],
    arg2: Optional[Union[FData, np.ndarray]] = None,
    memory_per_batch: Optional[int] = None,
    **kwargs: Any,
) -> np.ndarray:
    """Compute pairwise a commutative function."""
    dim1 = len(arg1)
    if arg2 is None or arg2 is arg1:
        indices = np.triu_indices(dim1)

        matrix = np.empty((dim1, dim1))

        triang_vec = _map_in_batches(
            function,
            (arg1, arg1),
            indices,
            memory_per_batch=memory_per_batch,
            **kwargs,
        )

        # Set upper matrix
        matrix[indices] = triang_vec

        # Set lower matrix
        matrix[(indices[1], indices[0])] = triang_vec

        return matrix

    dim2 = len(arg2)
    indices = np.indices((dim1, dim2))

    vec = _map_in_batches(
        function,
        (arg1, arg2),
        (indices[0].ravel(), indices[1].ravel()),
        memory_per_batch=memory_per_batch,
        **kwargs,
    )

    return vec.reshape((dim1, dim2))


def _int_to_real(array: np.ndarray) -> np.ndarray:
    """Convert integer arrays to floating point."""
    return array + 0.0


def _check_array_key(array, key):
    """Check a getitem key."""
    key = check_array_indexer(array, key)

    if isinstance(key, numbers.Integral):  # To accept also numpy ints
        key = int(key)
        key = range(len(array))[key]

        return slice(key, key + 1)

    return key


def _check_estimator(estimator):
    from sklearn.utils.estimator_checks import (
        check_get_params_invariance,
        check_set_params,
    )

    name = estimator.__name__
    instance = estimator()
    check_get_params_invariance(name, instance)
    check_set_params(name, instance)


def _classifier_get_classes(y: ndarray) -> Tuple[ndarray, ndarray]:
    from sklearn.preprocessing import LabelEncoder
    from sklearn.utils.multiclass import check_classification_targets

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


def _classifier_get_depth_methods(
    classes: ndarray,
    X: FDataGrid,
    y_ind: ndarray,
    depth_methods: Sequence[Depth[T]],
) -> Sequence[Depth[T]]:
    return [
        clone(depth_method).fit(X[y_ind == cur_class])
        for cur_class in range(classes.size)
        for depth_method in depth_methods
    ]


def _classifier_fit_depth_methods(
    X: FDataGrid,
    y: ndarray,
    depth_methods: Sequence[Depth[T]],
) -> Tuple[ndarray, Sequence[Depth[T]]]:
    classes, y_ind = _classifier_get_classes(y)

    class_depth_methods_ = _classifier_get_depth_methods(
        classes, X, y_ind, depth_methods,
    )

    return classes, class_depth_methods_
