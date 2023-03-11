"""Test regularization methods."""
from __future__ import annotations

import unittest
import warnings
from typing import Callable, Optional, Sequence, Union

import numpy as np
from sklearn.datasets import make_regression
from sklearn.linear_model import Ridge
from sklearn.model_selection._split import train_test_split

import skfda
from skfda.misc.operators import (
    Identity,
    LinearDifferentialOperator,
    gram_matrix,
)
from skfda.misc.operators._linear_differential_operator import (
    _monomial_evaluate_constant_linear_diff_op,
)
from skfda.misc.operators._operators import gram_matrix_numerical
from skfda.misc.regularization import L2Regularization
from skfda.ml.regression import LinearRegression
from skfda.representation.basis import (
    Basis,
    BSplineBasis,
    ConstantBasis,
    FourierBasis,
    MonomialBasis,
)

LinearDifferentialOperatorInput = Union[
    int,
    Sequence[
        Union[
            float,
            Callable[
                [np.typing.NDArray[np.float_]],
                np.typing.NDArray[np.float_],
            ],
        ],
    ],
    None,
]


class TestLinearDifferentialOperatorRegularization(unittest.TestCase):
    """Test linear differential operator penalty with different bases."""

    def _test_penalty(
        self,
        basis: Basis,
        linear_diff_op: LinearDifferentialOperatorInput,
        atol: float = 0,
        result: Optional[np.typing.NDArray[np.float_]] = None,
    ) -> None:

        operator = LinearDifferentialOperator(linear_diff_op)

        penalty = gram_matrix(operator, basis)
        numerical_penalty = gram_matrix_numerical(operator, basis)

        np.testing.assert_allclose(
            penalty,
            numerical_penalty,
            atol=atol,
        )

        if result is not None:
            np.testing.assert_allclose(
                penalty,
                result,
                atol=atol,
            )

    def test_constant_penalty(self) -> None:
        """Test penalty for Constant basis."""
        basis = ConstantBasis(domain_range=(0, 3))

        res = np.array([[12]])

        self._test_penalty(basis, linear_diff_op=[2, 3, 4], result=res)

    def test_monomial_linear_diff_op(self) -> None:
        """Test directly the penalty for Monomial basis."""
        n_basis = 5

        basis = MonomialBasis(n_basis=n_basis)

        linear_diff_op = [3]
        res = np.array([
            [0, 0, 0, 0, 3],
            [0, 0, 0, 3, 0],
            [0, 0, 3, 0, 0],
            [0, 3, 0, 0, 0],
            [3, 0, 0, 0, 0],
        ])

        np.testing.assert_allclose(
            _monomial_evaluate_constant_linear_diff_op(
                basis,
                np.array(linear_diff_op),
            ),
            res,
        )

        linear_diff_op = [3, 2]
        res = np.array([
            [0, 0, 0, 0, 3],
            [0, 0, 0, 3, 2],
            [0, 0, 3, 4, 0],
            [0, 3, 6, 0, 0],
            [3, 8, 0, 0, 0],
        ])

        np.testing.assert_allclose(
            _monomial_evaluate_constant_linear_diff_op(
                basis,
                np.array(linear_diff_op),
            ),
            res,
        )

        linear_diff_op = [3, 0, 5]
        res = np.array([
            [0, 0, 0, 0, 3],
            [0, 0, 0, 3, 0],
            [0, 0, 3, 0, 10],
            [0, 3, 0, 30, 0],
            [3, 0, 60, 0, 0],
        ])

        np.testing.assert_allclose(
            _monomial_evaluate_constant_linear_diff_op(
                basis,
                np.array(linear_diff_op),
            ),
            res,
        )

    def test_monomial_penalty(self) -> None:
        """Test penalty for Monomial basis."""
        basis = MonomialBasis(n_basis=5, domain_range=(0, 3))

        # Theorethical result
        res = np.array([
            [0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0],
            [0, 0, 12, 54, 216],
            [0, 0, 54, 324, 1458],
            [0, 0, 216, 1458, 6998.4],
        ])

        self._test_penalty(basis, linear_diff_op=2, result=res)

        basis = MonomialBasis(n_basis=8, domain_range=(1, 5))

        self._test_penalty(basis, linear_diff_op=[1, 2, 3])
        self._test_penalty(basis, linear_diff_op=7)
        self._test_penalty(basis, linear_diff_op=0)
        self._test_penalty(basis, linear_diff_op=1)
        self._test_penalty(basis, linear_diff_op=27)

    def test_fourier_penalty(self) -> None:
        """Test penalty for Fourier basis."""
        basis = FourierBasis(n_basis=5)

        res = np.array([
            [0, 0, 0, 0, 0],
            [0, 1558.55, 0, 0, 0],
            [0, 0, 1558.55, 0, 0],
            [0, 0, 0, 24936.73, 0],
            [0, 0, 0, 0, 24936.73],
        ])

        # Those comparisons require atol as there are zeros involved
        self._test_penalty(basis, linear_diff_op=2, atol=0.01, result=res)

        basis = FourierBasis(n_basis=9, domain_range=(1, 5))
        self._test_penalty(basis, linear_diff_op=[1, 2, 3], atol=1e-7)
        self._test_penalty(basis, linear_diff_op=[2, 3, 0.1, 1], atol=1e-7)
        self._test_penalty(basis, linear_diff_op=0, atol=1e-7)
        self._test_penalty(basis, linear_diff_op=1, atol=1e-7)
        self._test_penalty(basis, linear_diff_op=3, atol=1e-7)

    def test_bspline_penalty(self) -> None:
        """Test penalty for BSpline basis."""
        basis = BSplineBasis(n_basis=5)

        res = np.array([
            [96, -132, 24, 12, 0],
            [-132, 192, -48, -24, 12],
            [24, -48, 48, -48, 24],
            [12, -24, -48, 192, -132],
            [0, 12, 24, -132, 96],
        ])

        self._test_penalty(basis, linear_diff_op=2, result=res)

        basis = BSplineBasis(n_basis=9, domain_range=(1, 5))
        self._test_penalty(basis, linear_diff_op=[1, 2, 3])
        self._test_penalty(basis, linear_diff_op=[2, 3, 0.1, 1])
        self._test_penalty(basis, linear_diff_op=0)
        self._test_penalty(basis, linear_diff_op=1)
        self._test_penalty(basis, linear_diff_op=3)
        self._test_penalty(basis, linear_diff_op=4)

        basis = BSplineBasis(n_basis=16, order=8)
        self._test_penalty(basis, linear_diff_op=0, atol=1e-7)

    def test_bspline_penalty_special_case(self) -> None:
        """Test for behavior like in issue #185."""
        basis = BSplineBasis(n_basis=5)

        res = np.array([
            [1152, -2016, 1152, -288, 0],
            [-2016, 3600, -2304, 1008, -288],
            [1152, -2304, 2304, -2304, 1152],
            [-288, 1008, -2304, 3600, -2016],
            [0, -288, 1152, -2016, 1152],
        ])

        operator = LinearDifferentialOperator(basis.order - 1)
        penalty = gram_matrix(operator, basis)
        numerical_penalty = gram_matrix_numerical(operator, basis)

        np.testing.assert_allclose(
            penalty,
            res,
        )

        np.testing.assert_allclose(
            numerical_penalty,
            res,
        )


class TestDefaultL2Regularization(unittest.TestCase):
    """Test default value of L2 regularization."""

    def test_basis_default(self) -> None:
        """Test that in basis smoothing."""
        data_matrix = np.linspace([0, 1, 2, 3], [1, 2, 3, 4], 100)

        fd = skfda.FDataGrid(data_matrix.T)

        smoother = skfda.preprocessing.smoothing.BasisSmoother(
            basis=skfda.representation.basis.BSplineBasis(
                n_basis=10,
                domain_range=fd.domain_range,
            ),
            regularization=L2Regularization(),
        )

        smoother2 = skfda.preprocessing.smoothing.BasisSmoother(
            basis=skfda.representation.basis.BSplineBasis(
                n_basis=10,
                domain_range=fd.domain_range,
            ),
            regularization=L2Regularization(
                Identity(),
            ),
        )

        fd_basis = smoother.fit_transform(fd)
        fd_basis2 = smoother2.fit_transform(fd)

        np.testing.assert_allclose(
            fd_basis.data_matrix,
            fd_basis2.data_matrix,
        )


class TestEndpointsDifferenceRegularization(unittest.TestCase):
    """Test regularization with a callable."""

    def test_basis_conversion(self) -> None:
        """Test that in basis smoothing."""
        data_matrix = np.linspace([0, 1, 2, 3], [1, 2, 3, 4], 100)

        fd = skfda.FDataGrid(data_matrix.T)

        smoother = skfda.preprocessing.smoothing.BasisSmoother(
            basis=skfda.representation.basis.BSplineBasis(
                n_basis=10,
                domain_range=fd.domain_range,
            ),
            regularization=L2Regularization(
                lambda x: x(1)[:, 0] - x(0)[:, 0],
            ),
            smoothing_parameter=10000,
        )

        fd_basis = smoother.fit_transform(fd)

        np.testing.assert_allclose(
            fd_basis(0),
            fd_basis(1),
            atol=0.001,
        )


class TestL2Regularization(unittest.TestCase):
    """Test the L2 regularization."""

    def test_multivariate(self) -> None:
        """Test that it works with multivariate inputs."""

        def ignore_scalar_warning() -> None:  # noqa: WPS430
            warnings.filterwarnings(
                "ignore",
                category=UserWarning,
                message="All the covariates are scalar.",
            )

        X, y = make_regression(
            n_samples=20,
            n_features=10,
            random_state=1,
            bias=3.5,
        )

        X_train, X_test, y_train, _ = train_test_split(
            X,
            y,
            random_state=2,
        )

        for regularization_parameter in (0, 1, 10, 100):

            with self.subTest(
                regularization_parameter=regularization_parameter,
            ):

                sklearn_l2 = Ridge(alpha=regularization_parameter)
                skfda_l2 = LinearRegression(
                    regularization=L2Regularization(
                        regularization_parameter=regularization_parameter,
                    ),
                )

                sklearn_l2.fit(X_train, y_train)
                with warnings.catch_warnings():
                    ignore_scalar_warning()
                    skfda_l2.fit(X_train, y_train)

                sklearn_y_pred = sklearn_l2.predict(X_test)
                with warnings.catch_warnings():
                    ignore_scalar_warning()
                    skfda_y_pred = skfda_l2.predict(X_test)

                np.testing.assert_allclose(
                    sklearn_l2.coef_,
                    skfda_l2.coef_[0],
                )

                np.testing.assert_allclose(
                    sklearn_l2.intercept_,
                    skfda_l2.intercept_,
                )

                np.testing.assert_allclose(
                    sklearn_y_pred,
                    skfda_y_pred,
                )
