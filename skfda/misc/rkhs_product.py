"""RKHS inner product of two FData objects."""
from __future__ import annotations

from typing import Callable, Optional

import multimethod
import numpy as np
import scipy.linalg

from ..misc.covariances import EmpiricalBasis
from ..representation import FData, FDataBasis, FDataGrid
from ..representation.basis import Basis, TensorBasis
from ..typing._numpy import NDArrayFloat
from ._math import inner_product
from .validation import check_fdata_dimensions


def _broadcast_samples(
    fdata1: FData,
    fdata2: FData,
) -> tuple[FData, FData]:
    """Broadcast samples of two FData objects.

    Args:
        fdata1: First FData object.
        fdata2: Second FData object.

    Returns:
        Tuple of FData objects with the same number of samples.

    Raises:
        ValueError: If the number of samples is not the same or if the number
    """
    if fdata1.n_samples == 1:
        fdata1 = fdata1.repeat(fdata2.n_samples)
    elif fdata2.n_samples == 1:
        fdata2 = fdata2.repeat(fdata1.n_samples)
    elif fdata1.n_samples != fdata2.n_samples:
        raise ValueError(
            "Invalid number of samples for functional data objects:"
            f"{fdata1.n_samples} != {fdata2.n_samples}.",
        )
    return fdata1, fdata2


def _get_coeff_matrix(
    cov_function: Callable[[NDArrayFloat, NDArrayFloat], NDArrayFloat],
    basis1: Basis,
    basis2: Basis,
) -> NDArrayFloat:
    """Return the matrix of coefficients of a function in a tensor basis.

    Args:
        cov_function: Covariance function as a callable. It is expected to
            receive two arrays, s and t, and return the corresponding
            covariance matrix.
        basis1: First basis.
        basis2: Second basis.

    Returns:
        Matrix of coefficients of the covariance function in the tensor
        basis formed by the two given bases.
    """
    # In order to use inner_product, the callable must follow the
    # same convention as the evaluation on FDataGrids, that is, it
    # is expected to receive a single bidimensional point as input
    def cov_function_pointwise(  # noqa: WPS430
        x: NDArrayFloat,
    ) -> NDArrayFloat:
        t = np.array([x[0]])
        s = np.array([x[1]])
        return cov_function(t, s)[..., np.newaxis]

    tensor_basis = TensorBasis([basis1, basis2])

    # Solving the system yields the coefficients of the covariance
    # as a vector that is reshaped to form a matrix
    return np.linalg.solve(
        tensor_basis.gram_matrix(),
        inner_product(
            cov_function_pointwise,
            tensor_basis,
            _domain_range=tensor_basis.domain_range,
        ).T,
    ).reshape(basis1.n_basis, basis2.n_basis)


@multimethod.multidispatch
def rkhs_inner_product(
    fdata1: FData,
    fdata2: FData,
    *,
    cov_function: Callable[[NDArrayFloat, NDArrayFloat], NDArrayFloat],
) -> NDArrayFloat:
    """RKHS inner product of two FData objects.

    Args:
        fdata1: First FData object.
        fdata2: Second FData object.
        cov_function: Covariance function as a callable.

    Returns:
        Matrix of the RKHS inner product between paired samples of
        fdata1 and fdata2.
    """
    raise NotImplementedError(
        "RKHS inner product not implemented for the given types.",
    )


@rkhs_inner_product.register(FDataGrid, FDataGrid)
def rkhs_inner_product_fdatagrid(
    fdata1: FDataGrid,
    fdata2: FDataGrid,
    *,
    cov_function: Callable[[NDArrayFloat, NDArrayFloat], NDArrayFloat],
    cond: Optional[float] = None,
) -> NDArrayFloat:
    r"""RKHS inner product of two univariate FDataGrid objects.

    This is the most general method for calculating the RKHS inner product
    using a discretization of the domain. The RKHS inner product is calculated
    as the inner product between the square root inverse of the covariance
    operator applied to the functions.
    Assuming the existence of such inverse, using the self-adjointness of the
    covariance operator, the RKHS inner product can be calculated as:
        \langle f, \mathcal{K}^{-1} g \rangle
    When discretizing a common domain, this is equivalent to doing the matrix
    product between the discretized functions and the inverse of the
    covariance matrix.
    In case of having different discretization grids, the left inverse of the
    transposed covariace matrix is used instead.

    If one of the FDataGrid terms consists of only one sample, it is repeated
    to match the number of samples of the other term.

    Args:
        fdata1: First FDataGrid object.
        fdata2: Second FDataGrid object.
        cov_function: Covariance function as a callable. It is expected to
            receive two arrays, s and t, and return the corresponding
            covariance matrix.
        cond: Cutoff for small singular values of the covariance matrix.
            Default uses scipy default.

    Returns:
        Matrix of the RKHS inner product between paired samples of
        fdata1 and fdata2.
    """
    # Check univariate and apply broadcasting
    check_fdata_dimensions(fdata1, dim_domain=1, dim_codomain=1)
    check_fdata_dimensions(fdata2, dim_domain=1, dim_codomain=1)
    fdata1, fdata2 = _broadcast_samples(fdata1, fdata2)

    data_matrix_1 = fdata1.data_matrix
    data_matrix_2 = fdata2.data_matrix
    grid_points_1 = fdata1.grid_points[0]
    grid_points_2 = fdata2.grid_points[0]
    cov_matrix_1_2 = cov_function(grid_points_1, grid_points_2)

    # Calculate the inverse operator applied to fdata2
    if np.array_equal(grid_points_1, grid_points_2):
        inv_fd2_matrix = np.linalg.solve(
            cov_matrix_1_2,
            data_matrix_2,
        )
    else:
        inv_fd2_matrix = np.asarray(
            scipy.linalg.lstsq(
                cov_matrix_1_2.T,
                data_matrix_2[..., 0].T,
                cond=cond,
            )[0],
        ).T[..., np.newaxis]

    products = (
        np.transpose(data_matrix_1, axes=(0, 2, 1))
        @ inv_fd2_matrix
    )
    # Remove redundant dimensions
    return products.reshape(products.shape[0])


@rkhs_inner_product.register(FDataBasis, FDataBasis)
def rkhs_inner_product_fdatabasis(
    fdata1: FDataBasis,
    fdata2: FDataBasis,
    *,
    cov_function: Callable[[NDArrayFloat, NDArrayFloat], NDArrayFloat],
) -> NDArrayFloat:
    """RKHS inner product of two FDataBasis objects.

    In case of using a basis expression, the RKHS inner product can be
    computed obtaining first a basis representation of the covariance
    function in the tensor basis. Then, the inverse operator applied to
    the second term is calculated.
    In case of using a common basis, that step is done by solving the
    system given by the inverse of the matrix of coefficients of the
    covariance function in the tensor basis formed by the two given bases.
    In case of using different bases, the left inverse of the transposed
    matrix of coefficients of the covariance function is used instead.
    Finally, the inner product between each pair of samples is calculated.

    In case of knowing the matrix of coefficients of the covariance function
    in the tensor basis formed by the two given bases, it can be passed as
    an argument to avoid the calculation of it.

    If one of the FDataBasis terms consists of only one sample, it is repeated
    to match the number of samples of the other term.

    Args:
        fdata1: First FDataBasis object.
        fdata2: Second FDataBasis object.
        cov_function: Covariance function as a callable. It is expected to
            receive two arrays, s and t, and return the corresponding
            covariance matrix.

    Returns:
        Matrix of the RKHS inner product between paired samples of
        fdata1 and fdata2.
    """
    # Check univariate and apply broadcasting
    check_fdata_dimensions(fdata1, dim_domain=1, dim_codomain=1)
    check_fdata_dimensions(fdata2, dim_domain=1, dim_codomain=1)
    fdata1, fdata2 = _broadcast_samples(fdata1, fdata2)

    if isinstance(cov_function, EmpiricalBasis):
        cov_coeff_matrix = cov_function.coeff_matrix
    else:
        # Express the covariance function in the tensor basis
        # NOTE: The alternative is to convert to FDatagrid the two FDataBasis
        cov_coeff_matrix = _get_coeff_matrix(
            cov_function,
            fdata1.basis,
            fdata2.basis,
        )

    if fdata1.basis == fdata2.basis:
        inv_fd2_coefs = np.linalg.solve(
            cov_coeff_matrix,
            fdata2.coefficients.T,
        )
    else:
        inv_fd2_coefs = np.linalg.lstsq(
            cov_coeff_matrix.T,
            fdata2.coefficients.T,
            rcond=None,
        )[0]

    # Following einsum is equivalent to doing the matrix multiplication
    # and then taking the diagonal of the result
    return np.einsum(  # type: ignore[no-any-return]
        "ij,ji->i",
        fdata1.coefficients,
        inv_fd2_coefs,
    )
