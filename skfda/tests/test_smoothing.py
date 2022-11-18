"""Test smoothing methods."""
import unittest
from typing import Tuple

import numpy as np
import sklearn
from typing_extensions import Literal

import skfda
import skfda.preprocessing.smoothing as smoothing
import skfda.preprocessing.smoothing.validation as validation
from skfda._utils import _check_estimator
from skfda.datasets import fetch_weather
from skfda.misc.hat_matrix import (
    HatMatrix,
    KNeighborsHatMatrix,
    LocalLinearRegressionHatMatrix,
    NadarayaWatsonHatMatrix,
)
from skfda.misc.operators import LinearDifferentialOperator
from skfda.misc.regularization import L2Regularization
from skfda.preprocessing.smoothing import KernelSmoother
from skfda.representation.basis import BSplineBasis, MonomialBasis
from skfda.representation.grid import FDataGrid


class TestSklearnEstimators(unittest.TestCase):
    """Test for sklearn estimators."""

    def test_kernel_smoothing(self) -> None:
        """Test if estimator adheres to scikit-learn conventions."""
        _check_estimator(KernelSmoother)


class _LinearSmootherLeaveOneOutScorerAlternative:
    """Alternative implementation of the LinearSmootherLeaveOneOutScorer."""

    def __call__(
        self,
        estimator: KernelSmoother,
        X: FDataGrid,
        y: FDataGrid,
    ) -> float:
        """Calculate Leave-One-Out score."""
        estimator_clone = sklearn.base.clone(estimator)

        estimator_clone._cv = True  # noqa: WPS437
        y_est = estimator_clone.fit_transform(X)

        return float(
            -np.mean((y.data_matrix[..., 0] - y_est.data_matrix[..., 0])**2),
        )


class TestLeaveOneOut(unittest.TestCase):
    """Tests of Leave-One-Out score for kernel smoothing."""

    def _test_generic(
        self,
        estimator: KernelSmoother,
        smoothing_param_name: str = 'kernel_estimator__bandwidth',
    ) -> None:
        loo_scorer = validation.LinearSmootherLeaveOneOutScorer()
        loo_scorer_alt = _LinearSmootherLeaveOneOutScorerAlternative()

        x = np.linspace(-2, 2, 5)
        fd = skfda.FDataGrid(x ** 2, x)

        grid = validation.SmoothingParameterSearch(
            estimator,
            [2, 3],
            param_name=smoothing_param_name,
            scoring=loo_scorer,
        )

        grid.fit(fd)
        score = np.array(grid.cv_results_['mean_test_score'])

        grid_alt = validation.SmoothingParameterSearch(
            estimator,
            [2, 3],
            param_name=smoothing_param_name,
            scoring=loo_scorer_alt,
        )

        grid_alt.fit(fd)
        score_alt = np.array(grid_alt.cv_results_['mean_test_score'])

        np.testing.assert_array_almost_equal(score, score_alt)

    def test_nadaraya_watson(self) -> None:
        """Test Leave-One-Out with Nadaraya Watson method."""
        self._test_generic(KernelSmoother(
            kernel_estimator=NadarayaWatsonHatMatrix(),
        ),
        )

    def test_local_linear_regression(self) -> None:
        """Test Leave-One-Out with Local Linear Regression method."""
        self._test_generic(KernelSmoother(
            kernel_estimator=LocalLinearRegressionHatMatrix(),
        ),
        )

    def test_knn(self) -> None:
        """Test Leave-One-Out with KNNeighbours method."""
        self._test_generic(
            KernelSmoother(
                kernel_estimator=KNeighborsHatMatrix(),
            ),
            smoothing_param_name='kernel_estimator__n_neighbors',
        )


class TestKernelSmoother(unittest.TestCase):
    """Test Kernel Smoother.

    Comparison of results with fda.usc R library
    """

    def _test_hat_matrix(
        self,
        kernel_estimator: HatMatrix,
    ) -> np.typing.NDArray[np.float_]:
        return KernelSmoother(  # noqa: WPS437
            kernel_estimator=kernel_estimator,
        )._hat_matrix(
            input_points=[[1, 2, 3, 4, 5]],
            output_points=[[1, 2, 3, 4, 5]],
        )

    def test_nw(self) -> None:
        """Comparison of NW hat matrix with the one obtained from fda.usc."""
        hat_matrix = self._test_hat_matrix(
            NadarayaWatsonHatMatrix(bandwidth=10),
        )
        hat_matrix_r = [
            [0.206001865, 0.204974427, 0.201922755, 0.196937264, 0.190163689],
            [0.201982911, 0.202995354, 0.201982911, 0.198975777, 0.194063047],
            [0.198003042, 0.200995474, 0.202002968, 0.200995474, 0.198003042],
            [0.194063047, 0.198975777, 0.201982911, 0.202995354, 0.201982911],
            [0.190163689, 0.196937264, 0.201922755, 0.204974427, 0.206001865],
        ]
        np.testing.assert_allclose(hat_matrix, hat_matrix_r)

    def test_llr(self) -> None:
        """Test LLR."""
        hat_matrix = self._test_hat_matrix(
            LocalLinearRegressionHatMatrix(bandwidth=10),
        )

        # For a straight line the estimated results should coincide with
        # the real values
        # r(x) = 3x + 2
        np.testing.assert_allclose(
            np.dot(hat_matrix, [5, 8, 11, 14, 17]),
            [5, 8, 11, 14, 17],
        )

    def test_knn(self) -> None:
        """Comparison of KNN hat matrix with the one obtained from fda.usc."""
        hat_matrix = self._test_hat_matrix(
            KNeighborsHatMatrix(n_neighbors=2),
        )

        hat_matrix_r = [
            [0.500000000, 0.500000000, 0.000000000, 0.000000000, 0.000000000],
            [0.333333333, 0.333333333, 0.333333333, 0.000000000, 0.000000000],
            [0.000000000, 0.333333333, 0.333333333, 0.333333333, 0.000000000],
            [0.000000000, 0.000000000, 0.333333333, 0.333333333, 0.333333333],
            [0.000000000, 0.000000000, 0.000000000, 0.500000000, 0.500000000],
        ]
        np.testing.assert_allclose(hat_matrix, hat_matrix_r)


class TestBasisSmoother(unittest.TestCase):
    """Test Basis Smoother."""

    def test_cholesky(self) -> None:
        """Test Basis Smoother using BSpline basis and Cholesky method."""
        t = np.linspace(0, 1, 5)
        x = np.sin(2 * np.pi * t) + np.cos(2 * np.pi * t)
        basis = BSplineBasis((0, 1), n_basis=5)

        fd = FDataGrid(data_matrix=x, grid_points=t)

        smoother = smoothing.BasisSmoother(
            basis=basis,
            smoothing_parameter=10,
            regularization=L2Regularization(
                LinearDifferentialOperator(2),
            ),
            method='cholesky',
            return_basis=True,
        )

        fd_basis = smoother.fit_transform(fd)

        np.testing.assert_array_almost_equal(
            fd_basis.coefficients.round(2),
            np.array([[0.6, 0.47, 0.2, -0.07, -0.2]]),
        )

    def test_qr(self) -> None:
        """Test Basis Smoother using BSpline basis and QR method."""
        t = np.linspace(0, 1, 5)
        x = np.sin(2 * np.pi * t) + np.cos(2 * np.pi * t)
        basis = BSplineBasis((0, 1), n_basis=5)

        fd = FDataGrid(data_matrix=x, grid_points=t)

        smoother = smoothing.BasisSmoother(
            basis=basis,
            smoothing_parameter=10,
            regularization=L2Regularization(
                LinearDifferentialOperator(2),
            ),
            method='qr',
            return_basis=True,
        )

        fd_basis = smoother.fit_transform(fd)

        np.testing.assert_array_almost_equal(
            fd_basis.coefficients.round(2),
            np.array([[0.6, 0.47, 0.2, -0.07, -0.2]]),
        )

    def test_monomial_smoothing(self) -> None:
        """Test Basis Smoother using Monomial basis."""
        # It does not have much sense to apply smoothing in this basic case
        # where the fit is very good but its just for testing purposes
        t = np.linspace(0, 1, 5)
        x = np.sin(2 * np.pi * t) + np.cos(2 * np.pi * t)
        basis = MonomialBasis(n_basis=4)

        fd = FDataGrid(data_matrix=x, grid_points=t)

        smoother = smoothing.BasisSmoother(
            basis=basis,
            smoothing_parameter=1,
            regularization=L2Regularization(
                LinearDifferentialOperator(2),
            ),
            return_basis=True,
        )

        fd_basis = smoother.fit_transform(fd)

        # These results where extracted from the R package fda
        np.testing.assert_array_almost_equal(
            fd_basis.coefficients.round(2),
            np.array([[0.61, -0.88, 0.06, 0.02]]),
        )

    def test_vector_valued_smoothing(self) -> None:
        """Test Basis Smoother for vector values functions."""
        X, _ = fetch_weather(return_X_y=True)

        basis_dim = skfda.representation.basis.FourierBasis(
            n_basis=7, domain_range=X.domain_range,
        )

        basis = skfda.representation.basis.VectorValuedBasis(
            [basis_dim] * 2,
        )

        method_set: Tuple[Literal['cholesky', 'qr', 'svd'], ...] = (
            'cholesky',
            'qr',
            'svd',
        )
        for method in method_set:
            with self.subTest(method=method):

                basis_smoother = smoothing.BasisSmoother(
                    basis,
                    regularization=L2Regularization(
                        LinearDifferentialOperator(2),
                    ),
                    return_basis=True,
                    smoothing_parameter=1,
                    method=method,
                )

                basis_smoother_dim = smoothing.BasisSmoother(
                    basis_dim,
                    regularization=L2Regularization(
                        LinearDifferentialOperator(2),
                    ),
                    return_basis=True,
                    smoothing_parameter=1,
                    method=method,
                )

                X_basis = basis_smoother.fit_transform(X)

                self.assertEqual(X_basis.dim_codomain, 2)

                self.assertEqual(X_basis.coordinates[0].basis, basis_dim)
                np.testing.assert_allclose(
                    X_basis.coordinates[0].coefficients,
                    basis_smoother_dim.fit_transform(
                        X.coordinates[0],
                    ).coefficients,
                )

                self.assertEqual(X_basis.coordinates[1].basis, basis_dim)
                np.testing.assert_allclose(
                    X_basis.coordinates[1].coefficients,
                    basis_smoother_dim.fit_transform(
                        X.coordinates[1],
                    ).coefficients,
                )
