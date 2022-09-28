"""Methods to solve least squares problems."""
from __future__ import annotations

from typing import Callable, Optional, Union

import numpy as np
import scipy.linalg
from typing_extensions import Final, Literal

from ..typing._numpy import NDArrayFloat

LstsqMethodCallable = Callable[[np.ndarray, np.ndarray], np.ndarray]
LstsqMethodName = Literal["cholesky", "qr", "svd"]
LstsqMethod = Union[LstsqMethodCallable, LstsqMethodName]


def lstsq_cholesky(
    coefs: NDArrayFloat,
    result: NDArrayFloat,
) -> NDArrayFloat:
    """Solve OLS problem using a Cholesky decomposition."""
    left = coefs.T @ coefs
    right = coefs.T @ result
    return scipy.linalg.solve(  # type: ignore[no-any-return]
        left,
        right,
        assume_a="pos",
    )


def lstsq_qr(
    coefs: NDArrayFloat,
    result: NDArrayFloat,
) -> NDArrayFloat:
    """Solve OLS problem using a QR decomposition."""
    return scipy.linalg.lstsq(  # type: ignore[no-any-return]
        coefs,
        result,
        lapack_driver="gelsy",
    )[0]


def lstsq_svd(
    coefs: NDArrayFloat,
    result: NDArrayFloat,
) -> NDArrayFloat:
    """Solve OLS problem using a SVD decomposition."""
    return scipy.linalg.lstsq(  # type: ignore[no-any-return]
        coefs,
        result,
        lapack_driver="gelsd",
    )[0]


method_dict: Final = {
    "cholesky": lstsq_cholesky,
    "qr": lstsq_qr,
    "svd": lstsq_svd,
}


def _get_lstsq_method(
    method: LstsqMethod,
) -> LstsqMethodCallable:
    """Convert method string to method if necessary."""
    return method if callable(method) else method_dict[method]


def solve_regularized_weighted_lstsq(
    coefs: NDArrayFloat,
    result: NDArrayFloat,
    *,
    weights: Optional[NDArrayFloat] = None,
    penalty_matrix: Optional[NDArrayFloat] = None,
    lstsq_method: LstsqMethod = lstsq_svd,
) -> NDArrayFloat:
    """
    Solve a regularized and weighted least squares problem.

    If weights is a 1-D array it is converted to 2-D array with weights on the
    diagonal.

    If the penalty matrix is not ``None`` and nonzero, there
    is a closed solution. Otherwise the problem can be reduced
    to a least squares problem.

    """
    lstsq_method = _get_lstsq_method(lstsq_method)

    if lstsq_method is not lstsq_cholesky and (
        penalty_matrix is None
    ):
        # Weighted least squares case
        if weights is not None:

            if weights.ndim == 1:
                weights_chol = np.diag(np.sqrt(weights))
                coefs = weights_chol * coefs
                result = weights_chol * result
            else:
                weights_chol = scipy.linalg.cholesky(weights)
                coefs = weights_chol @ coefs
                result = weights_chol @ result

        return lstsq_method(coefs, result)

    # Cholesky case (always used for the regularized case)
    if weights is None:
        left = coefs.T @ coefs
        right = coefs.T @ result
    else:
        left = coefs.T @ weights @ coefs
        right = coefs.T @ weights @ result

    if penalty_matrix is not None:
        left += penalty_matrix

    return scipy.linalg.solve(  # type: ignore[no-any-return]
        left,
        right,
        assume_a="pos",
    )
