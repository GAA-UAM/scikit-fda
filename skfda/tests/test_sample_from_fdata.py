from typing import Any

import numpy as np
import pytest

from skfda import FData, FDataBasis, FDataGrid, FDataIrregular
from skfda.datasets import irregular_sample
from skfda.representation.basis import (
    FourierBasis,
    MonomialBasis,
    TensorBasis,
    VectorValuedBasis,
)

random_state = np.random.RandomState(23486974)


def _assert_equivalent(fdata: FData, fdatairregular: FDataIrregular) -> None:
    points = np.split(fdatairregular.points, fdatairregular.start_indices[1:])
    assert len(points) == len(fdatairregular) == len(fdata)
    for fun_points, original, irregular in zip(points, fdata, fdatairregular):
        np.testing.assert_allclose(
            irregular.values, original(fun_points)[0],
            # irregular(fun_points), original(fun_points),
        )
        # The commented line above should be used but evaluation of
        # FDataIrregular is not working for multidimensional domain


@pytest.fixture
def fdatabasis_1dimensional() -> FDataBasis:
    basis = MonomialBasis(n_basis=4, domain_range=(0, 1))
    return FDataBasis(
        basis=basis,
        coefficients=random_state.randn(15, basis.n_basis),
    )


@pytest.fixture
def fdatabasis_multidimensional() -> FDataBasis:
    """3-dimensional domain and 2-dimensional codomain"""
    basis_momonial1 = MonomialBasis(n_basis=3, domain_range=(-3, 3))
    basis_fourier1 = FourierBasis(n_basis=3, domain_range=(-3, 3))
    basis_monomial2 = MonomialBasis(n_basis=2, domain_range=(0, 1))
    basis_fourier2 = FourierBasis(n_basis=5, domain_range=(0, 1))

    tensor_basis1 = TensorBasis([basis_momonial1, basis_monomial2])
    tensor_basis2 = TensorBasis([basis_fourier1, basis_fourier2])

    basis = VectorValuedBasis([tensor_basis1, tensor_basis2, tensor_basis1])
    return FDataBasis(
        basis=basis,
        coefficients=random_state.randn(15, basis.n_basis),
    )


@pytest.fixture
def fdatabasis_2dimensional_domain() -> FDataBasis:
    basis_fourier = FourierBasis(n_basis=5, domain_range=(-3, 3))
    basis_monomial = MonomialBasis(n_basis=4, domain_range=(0, 1))
    basis = TensorBasis([basis_fourier, basis_monomial])
    return FDataBasis(
        basis=basis,
        coefficients=random_state.randn(15, basis.n_basis),
    )


@pytest.fixture
def fdatagrid_1dimensional() -> FDataGrid:
    return FDataGrid(
        data_matrix=random_state.randn(14, 50),
        grid_points=np.linspace(0, 100, 50),
    )


@pytest.fixture
def fdatagrid_multidimensional() -> FDataGrid:
    """3-dimensional domain and 5-dimensional codomain"""
    return FDataGrid(
        data_matrix=random_state.randn(14, 10, 5, 7, 5),
        grid_points=[
            np.linspace(0, 100, 10),
            np.linspace(-20, 20, 5),
            np.linspace(-20, 20, 7),
        ],
    )


@pytest.fixture
def fdatairregular_1dimensional() -> FDataIrregular:
    start_indices = np.concatenate([
        [0], np.cumsum(random_state.randint(2, 5, 17)),
    ])
    return FDataIrregular(
        points=random_state.randn(100),
        values=random_state.randn(100),
        start_indices=start_indices,
    )


@pytest.fixture
def fdatairregular_multidimensional() -> FDataIrregular:
    start_indices = np.concatenate([
        [0], np.cumsum(random_state.randint(2, 5, 17)),
    ])
    return FDataIrregular(
        points=random_state.randn(100, 1),  # TODO: Change to multidimensional
        # domain when evaluation of FDataIrregular is working for
        # multidimensional domains.
        values=random_state.randn(100, 5),
        start_indices=start_indices,
    )


@pytest.mark.parametrize(
    "fdata_fixture",
    [
        "fdatabasis_1dimensional",
        "fdatagrid_1dimensional",
        "fdatairregular_1dimensional",
        "fdatabasis_2dimensional_domain",
        "fdatabasis_multidimensional",
        "fdatagrid_multidimensional",
        "fdatairregular_multidimensional",
    ],
)
def test_irregular_sample(
    fdata_fixture: str, request: Any
) -> None:
    """Test the irregular sample function.

    The test checks that the number of points per curve after irregular
    sampling is as expected. The test is done for different dimensions in
    domain and codomain, as well as for FDataBasis, FDataGrid and
    FDataIrregular.
    """
    fdata: FDataBasis | FDataGrid | FDataIrregular = (
        request.getfixturevalue(fdata_fixture)
    )
    n_points_per_curve = random_state.randint(1, 15, fdata.n_samples)
    fdatairregular = irregular_sample(
        fdata,
        n_points_per_curve=n_points_per_curve,
        random_state=random_state,
    )

    got_points_per_curve = np.diff(
        np.append(fdatairregular.start_indices, [len(fdatairregular.points)]),
    )
    if isinstance(fdata, FDataBasis):
        assert all(got_points_per_curve == n_points_per_curve)
    else:
        assert all(got_points_per_curve <= n_points_per_curve)

    assert fdatairregular.values.shape == (
        sum(got_points_per_curve), fdata.dim_codomain,
    )

    # The values of the irregular sample should not contain NaNs
    # because the original datasets do not contain NaNs in their values
    assert np.sum(np.isnan(fdatairregular.values)) == 0

    _assert_equivalent(fdata, fdatairregular)
