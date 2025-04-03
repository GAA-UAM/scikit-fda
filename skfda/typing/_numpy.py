"""NumPy aliases for compatibility."""

from typing import Any, Union

import numpy as np

try:  # noqa: WPS113
    from numpy.typing import ArrayLike as ArrayLike, DTypeLike as DTypeLike
except ImportError:
    ArrayLike = np.ndarray  # type:ignore[misc] # noqa: WPS440
    DTypeLike = np.dtype  # type:ignore[misc]

try:  # noqa: WPS229
    from numpy.typing import NDArray
    NDArrayAny = NDArray[Any]
    NDArrayInt = NDArray[np.int_]
    NDArrayFloat = NDArray[np.float64]
    NDArrayReal = NDArray[Union[np.float64, np.int_]]
    NDArrayBool = NDArray[np.bool_]
    NDArrayStr = NDArray[np.str_]
    NDArrayObject = NDArray[np.object_]
except ImportError:
    NDArray = np.ndarray  # type:ignore[misc] # noqa: WPS440
    NDArrayAny = np.ndarray  # type:ignore[misc]
    NDArrayInt = np.ndarray  # type:ignore[misc]
    NDArrayFloat = np.ndarray  # type:ignore[misc]
    NDArrayReal = np.ndarray  # type:ignore[misc]
    NDArrayBool = np.ndarray  # type:ignore[misc]
    NDArrayStr = np.ndarray  # type:ignore[misc]
    NDArrayObject = np.ndarray  # type:ignore[misc]
