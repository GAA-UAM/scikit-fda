"""Tests of BasisOfFData."""

import numpy as np
import pytest

from skfda.representation.basis import CustomBasis, FDataBasis, FourierBasis
from skfda.representation.grid import FDataGrid


def test_grid():
    """Test a datagrid toy example."""
    grid_points = np.array([0, 1, 2])
    sample = FDataGrid(
        data_matrix=np.array([[1, 2, 3], [4, 5, 6]]),
        grid_points=np.array([[0, 1, 2]]),
    )

    data_basis = FDataBasis(
        basis=CustomBasis(fdata=sample),
        coefficients=np.array([[1, 0], [0, 1], [1, 1]]),
    )

    evaluated = [
        np.array([1, 2, 3]),
        np.array([4, 5, 6]),
        np.array([5, 7, 9]),
    ]

    np.testing.assert_equal(data_basis(grid_points)[..., 0], evaluated)


def test_basis():
    """Test a databasis toy example."""
    basis = FourierBasis(n_basis=3)
    coeficients = np.array(
        [
            [2, 1, 0],
            [3, 1, 0],
            [1, 2, 9],
        ],
    )

    data_basis = FDataBasis(
        basis=CustomBasis(
            fdata=FDataBasis(basis=basis, coefficients=coeficients),
        ),
        coefficients=coeficients,
    )

    combined_basis = FDataBasis(
        basis=basis,
        coefficients=coeficients @ coeficients,
    )

    eval_points = np.linspace(0, 1, 100)
    np.testing.assert_almost_equal(
        data_basis(eval_points),
        combined_basis(eval_points),
    )


def test_not_linearly_independent_too_many():
    """
    Test that a non linearly independent basis raises an error.

    In this case, the number of samples is greater than the number
    of sampling points or base functions in the underlying base.
    """
    sample = FDataGrid(
        data_matrix=np.array(
            [[1, 2, 3], [2, 4, 6], [15, 4, -2], [1, 28, 0]],
        ),
        grid_points=np.array([[0, 1, 2]]),
    )

    with pytest.raises(ValueError):
        CustomBasis(fdata=sample)

    sample = FDataBasis(
        basis=FourierBasis(n_basis=3),
        coefficients=np.array(
            [
                [0, 1, 0],
                [1, 0, 0],
                [0, 0, 1],
                [1, 1, 1],
            ],
        ),
    )

    with pytest.raises(ValueError):
        CustomBasis(fdata=sample)


def test_not_linearly_independent_range():
    """
    Test that a non linearly independent basis raises an error.

    In this case, the number of samples is valid but the given
    samples are not linearly independent.
    """
    sample = FDataGrid(
        data_matrix=np.array([[1, 2, 3], [2, 4, 6]]),
        grid_points=np.array([[0, 1, 2]]),
    )

    with pytest.raises(ValueError):
        CustomBasis(fdata=sample)

    sample = FDataBasis(
        basis=FourierBasis(n_basis=3),
        coefficients=np.array(
            [
                [2, 1, 0],
                [3, 1, 0],
                [1, 2, 0],
            ],
        ),
    )

    with pytest.raises(ValueError):
        CustomBasis(fdata=sample)


def test_derivative_grid():
    """Test the derivative of a basis constructed from a FDataGrid."""
    base_functions = FDataGrid(
        data_matrix=np.array([[1, 2, 3], [1, 1, 5]]),
        grid_points=np.array([[0, 1, 2]]),
    )
    # The derivative of the first function is always 1
    # The derivative of the second function is 0 and then 4

    basis = CustomBasis(fdata=base_functions)

    coefs = np.array([[1, 0], [0, 1], [1, 1]])
    derivate_basis, derivative_coefs = basis.derivative_basis_and_coefs(
        coefs=coefs,
        order=1,
    )

    derivative = FDataBasis(
        basis=derivate_basis,
        coefficients=derivative_coefs,
    )

    eval_points = np.array([0.5, 1.5])

    # Derivative of the original functions sampled in the given points
    # and multiplied by the coefs
    derivative_evaluated = np.array([[1, 1], [0, 4], [1, 5]])
    np.testing.assert_allclose(
        derivative(eval_points)[..., 0],
        derivative_evaluated,
        atol=1e-15,
    )


def test_derivative_basis():
    """Test the derivative of a basis constructed from a FDataBasis."""
    basis_coef = np.array(
        [
            [2, 1, 0, 10, 0],
            [3, 1, 99, 15, 99],
            [0, 2, 9, 22, 0],
        ],
    )

    base_functions = FDataBasis(
        basis=FourierBasis(n_basis=5),
        coefficients=basis_coef,
    )

    basis = CustomBasis(fdata=base_functions)

    coefs = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1], [2, 1, 5]])
    derivate_basis, derivative_coefs = basis.derivative_basis_and_coefs(
        coefs=coefs,
        order=1,
    )

    derivative = FDataBasis(
        basis=derivate_basis,
        coefficients=derivative_coefs,
    )

    eval_points = np.linspace(0, 1, 10)

    # Derivative of the original functions
    basis_derivative, coefs_derivative = FourierBasis(
        n_basis=5,
    ).derivative_basis_and_coefs(coefs=coefs @ basis_coef)

    derivative_on_the_basis = FDataBasis(
        basis=basis_derivative,
        coefficients=coefs_derivative,
    )
    np.testing.assert_almost_equal(
        derivative(eval_points),
        derivative_on_the_basis(eval_points),
    )


def test_multivariate_codomain():
    """Test basis from a multivariate function."""
    points = np.array([0, 1, 2])
    base_functions = FDataGrid(
        data_matrix=np.array(
            [
                [[0, 0, 1], [0, 0, 2], [0, 0, 3]],
                [[1, 0, 0], [2, 0, 0], [3, 0, 0]],
            ],
        ),
        grid_points=points,
    )
    base = CustomBasis(fdata=base_functions)

    coefs = np.array([[1, 0], [0, 1], [1, 1]])

    functions = FDataBasis(
        basis=base,
        coefficients=coefs,
    )

    expected_data = np.array(
        [
            [[0, 0, 1], [0, 0, 2], [0, 0, 3]],
            [[1, 0, 0], [2, 0, 0], [3, 0, 0]],
            [[1, 0, 1], [2, 0, 2], [3, 0, 3]],
        ],
    )
    np.testing.assert_equal(functions(points), expected_data)


def test_multivariate_codomain_linearly_dependent():
    """Test basis from multivariate linearly dependent functions."""
    points = np.array([0, 1, 2])
    base_functions = FDataGrid(
        data_matrix=np.array(
            [
                [[0, 0, 1], [0, 0, 2], [0, 0, 3]],
                [[1, 0, 0], [2, 0, 0], [3, 0, 0]],
                [[1, 0, 1], [2, 0, 2], [3, 0, 3]],
            ],
        ),
        grid_points=points,
    )
    # The third function is the sum of the first two

    with pytest.raises(ValueError):
        CustomBasis(fdata=base_functions)


def test_evaluate_derivative():
    """Test the evaluation of the derivative of a DataBasis."""
    grid_points = np.array([[0, 1, 2]])
    base_functions = FDataGrid(
        data_matrix=np.array([[1, 2, 3], [1, 3, 5]]),
        grid_points=grid_points,
    )
    basis = CustomBasis(fdata=base_functions)

    coefs = np.array([[1, 0], [0, 1], [1, 2]])

    functions = FDataBasis(
        basis=basis,
        coefficients=coefs,
    )
    deriv = functions.derivative(order=1)
    derivate_evalated = deriv(np.array([0.5, 1.5]))
    np.testing.assert_allclose(
        derivate_evalated[..., 0],
        np.array([[1, 1], [2, 2], [5, 5]]),
        rtol=1e-15,
    )


def test_coordinates():
    """Test the coordinates of a basis."""
    grid_points = np.array([[0, 1]])
    base_functions = FDataGrid(
        data_matrix=np.array([
            [[1, 2, 3], [1, 3, 5]],
            [[4, 5, 6], [7, 8, 9]],
        ]),
        grid_points=grid_points,
    )
    basis = CustomBasis(fdata=base_functions)

    coefs = np.array([[1, 0], [0, 1], [1, 2]])

    functions = FDataBasis(
        basis=basis,
        coefficients=coefs,
    )

    # First coordinate at 0
    np.testing.assert_allclose(
        functions.coordinates[0](0),
        np.array([[[1]], [[4]], [[9]]]),
    )

    # Second two coordinates at 1
    np.testing.assert_allclose(
        functions.coordinates[1:3](1),
        np.array([[[3, 5]], [[8, 9]], [[16 + 3, 18 + 5]]]),
    )


def test_equality():
    """Test the equality of two basis."""
    grid_points = np.array([[0, 1, 2]])
    base_functions = FDataGrid(
        data_matrix=np.array([[1, 2, 3], [1, 3, 5]]),
        grid_points=grid_points,
    )
    basis = CustomBasis(fdata=base_functions)

    other_base_functions = FDataGrid(
        data_matrix=np.array([[1, 2, 3], [1, 3, 5]]),
        grid_points=grid_points,
    )

    other_basis = CustomBasis(fdata=other_base_functions)

    assert basis == other_basis

    different_base_functions = FDataGrid(
        data_matrix=np.array([[1, 2, 3], [1, 8, 5]]),
        grid_points=grid_points,
    )

    different_basis = CustomBasis(fdata=different_base_functions)

    assert basis != different_basis
