"""Test the RKHS inner product method."""
from typing import Any

import numpy as np
import pytest

from skfda._utils import constants
from skfda.misc.rkhs_product import rkhs_inner_product
from skfda.representation import FDataBasis, FDataGrid
from skfda.representation.basis import (
    Basis,
    BSplineBasis,
    FourierBasis,
    TensorBasis,
)
from skfda.typing._numpy import NDArrayFloat

N_BASIS = 5
DOMAIN_RANGE = (0, 1)


@pytest.fixture(
    params=[
        FourierBasis,
        BSplineBasis,
    ],
    ids=[
        "FourierBasis",
        "BSplineBasis",
    ],
)
def basis_2(
    request: Any,
) -> Any:
    """Fixture for classes to test."""
    return request.param(
        n_basis=N_BASIS,
        domain_range=DOMAIN_RANGE,
    )


def test_rkhs_inner_product_grid() -> None:
    """Test the RKHS product for specific FDataGrid.

    The tested functions are:
        x(t) = t
        y(t) = t/3 + 1/2
        K(t,s) = t*s + 1

    The expected result of the RKHS product is:
        <x,y>_K = 1/3
    """
    grid_points = np.linspace(
        *DOMAIN_RANGE,
        constants.N_POINTS_FINE_MESH,
    )
    x = FDataGrid(
        [grid_points],
        grid_points,
    )
    y = (x / 3) + (1 / 2)

    def cov_function(  # noqa: WPS430
        t: NDArrayFloat,
        s: NDArrayFloat,
    ) -> NDArrayFloat:
        return np.outer(t, s) + 1

    result = rkhs_inner_product(
        fdata1=x,
        fdata2=y,
        cov_function=cov_function,
    )
    expected_result = [1 / 3]
    np.testing.assert_allclose(
        result,
        expected_result,
        rtol=1e-13,
    )

    # Test the product is symmetric
    result = rkhs_inner_product(
        fdata1=y,
        fdata2=x,
        cov_function=cov_function,
    )
    np.testing.assert_allclose(
        result,
        expected_result,
        rtol=1e-13,
    )


def test_rkhs_inner_product_basis(
    basis_2: Basis,
) -> None:
    """Test the RKHS product for specific FDataBasis.

    The tested functions are:
        x(t) = sin(2*pi*t)
        y(t) = phi_0(t) + phi_2(t)
        K(t,s) = Phi(t)^T B Phi(s)
    Where Phi = {phi_0, phi_1, phi_2} is the Fourier basis and B is a
    specific symmetric definite-positive matrix.
    The inverse of the operator applied to y is phi_2(t).

    The expected result of the RKHS product is:
        <x,y>_K = 0
    """
    grid_points = np.linspace(
        *DOMAIN_RANGE,
        constants.N_POINTS_FINE_MESH,
    )
    x = FDataGrid(
        [np.sin(2 * np.pi * grid_points)],
        grid_points,
    )

    # Define y and cov_function in term of basis_1
    basis_1 = FourierBasis(
        n_basis=3,
        domain_range=DOMAIN_RANGE,
    )
    y = FDataBasis(
        basis=basis_1,
        coefficients=[[1, 0, 1]],
    )
    kernel = FDataBasis(
        basis=TensorBasis([basis_1, basis_1]),
        coefficients=np.array([
            [3, 1, 1],
            [1, 2, 0],
            [1, 0, 1],
        ]).flatten(),
    )

    def cov_function(  # noqa: WPS430
        t: NDArrayFloat,
        s: NDArrayFloat,
    ) -> NDArrayFloat:
        return kernel([s, t], grid=True)[0, ..., 0]

    x_basis = x.to_basis(basis_1)
    y_basis = y.to_basis(basis_2)

    result = rkhs_inner_product(
        fdata1=x_basis,
        fdata2=y_basis,
        cov_function=cov_function,
    )
    expected_result = [0]
    np.testing.assert_allclose(
        result,
        expected_result,
        atol=1e-4,
    )

    # Test the product is symmetric
    result = rkhs_inner_product(
        fdata1=y_basis,
        fdata2=x_basis,
        cov_function=cov_function,
    )
    np.testing.assert_allclose(
        result,
        expected_result,
        atol=1e-4,
    )


def test_rkhs_inner_product_brownian_bridge() -> None:
    r"""Test the RKHS product for specific FDataGrid.

    The tested functions are:
        x(t) = 1-(2t-1)^2
        y(t) = sin(pi*t)
        K(t,s) = sigma^2 * (min(t,s) - t*s)
    Where sigma is a constant.

    The precise formula is:
        <f,g>_K = 1/sigma^2 * \int_0^1 f'(t)g'(t)dt
    Assuming that the domain is [0,1] and the functions
    are in the RKHS of the Brownian bridge.
    See ramos___2020_gaussian.pdf for more details.

    The expected result of the RKHS product is:
        <x,y>_K = 16/pi
        for sigma = 1
    """
    grid_points = np.linspace(
        *DOMAIN_RANGE,
        constants.N_POINTS_FINE_MESH,
    )
    # Remove axis 0 to avoid singular matrix
    grid_points = grid_points[1:-1]
    x = FDataGrid(
        [1 - (2 * grid_points - 1)**2],
        grid_points,
    )
    y = FDataGrid(
        [np.sin(np.pi * grid_points)],
        grid_points,
    )
    sigma = 1

    def cov_function(  # noqa: WPS430
        t: NDArrayFloat,
        s: NDArrayFloat,
    ) -> NDArrayFloat:
        # Matrix with the minimum of each pair of values
        min_values = np.amin(
            np.transpose(
                np.meshgrid(t, s),
                (2, 1, 0),
            ),
            axis=2,
        )
        return sigma**2 * (  # type: ignore[no-any-return]
            min_values - np.outer(t, s)
        )

    result = rkhs_inner_product(
        fdata1=x,
        fdata2=y,
        cov_function=cov_function,
    )
    expected_result = [16 / np.pi]
    np.testing.assert_allclose(
        result,
        expected_result,
        rtol=1e-5,
    )
