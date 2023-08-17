"""Test kernel regression method."""
import unittest
from typing import Callable, Optional, Tuple

import numpy as np
import sklearn.model_selection

from skfda import FData
from skfda.datasets import fetch_tecator
from skfda.misc.hat_matrix import (
    KNeighborsHatMatrix,
    LocalLinearRegressionHatMatrix,
    NadarayaWatsonHatMatrix,
)
from skfda.misc.kernels import normal, uniform
from skfda.misc.metrics import l2_distance
from skfda.ml.regression import KernelRegression
from skfda.representation.basis import FDataBasis, FourierBasis, MonomialBasis
from skfda.representation.grid import FDataGrid

FloatArray = np.typing.NDArray[np.float_]


def _nw_alt(
    fd_train: FData,
    fd_test: FData,
    y_train: FloatArray,
    *,
    bandwidth: float,
    kernel: Optional[Callable[[FloatArray], FloatArray]] = None,
) -> FloatArray:
    if kernel is None:
        kernel = normal

    y = np.zeros(fd_test.n_samples)
    for i in range(fd_test.n_samples):
        w = kernel(l2_distance(fd_train, fd_test[i]) / bandwidth)
        y[i] = (w @ y_train) / sum(w)

    return y


def _knn_alt(
    fd_train: FData,
    fd_test: FData,
    y_train: FloatArray,
    *,
    bandwidth: int,
    kernel: Optional[Callable[[FloatArray], FloatArray]] = None,
) -> FloatArray:
    if kernel is None:
        kernel = uniform

    y = np.zeros(fd_test.n_samples)

    for i in range(fd_test.n_samples):
        d = l2_distance(fd_train, fd_test[i])
        tol = np.finfo(np.float64).eps
        h = sorted(d)[bandwidth - 1] + tol
        w = kernel(d / h)
        y[i] = (w @ y_train) / sum(w)

    return y


def _llr_alt(
    fd_train: FDataBasis,
    fd_test: FDataBasis,
    y_train: FloatArray,
    *,
    bandwidth: float,
    kernel: Optional[Callable[[FloatArray], FloatArray]] = None,
) -> FloatArray:
    if kernel is None:
        kernel = normal

    y = np.zeros(fd_test.n_samples)

    for i in range(fd_test.n_samples):
        d = l2_distance(fd_train, fd_test[i])
        W = np.diag(kernel(d / bandwidth))

        C = np.concatenate(
            (
                np.ones(fd_train.n_samples)[:, np.newaxis],
                (fd_train - fd_test[i]).coefficients,
            ),
            axis=1,
        )

        M = np.linalg.inv(np.linalg.multi_dot([C.T, W, C]))
        y[i] = np.linalg.multi_dot([M, C.T, W, y_train])[0]

    return y


def _create_data_basis(
) -> Tuple[FDataBasis, FDataBasis, FloatArray]:
    X, y = fetch_tecator(return_X_y=True, as_frame=True)
    fd = X.iloc[:, 0].values
    fat = y['fat'].values

    basis = FourierBasis(
        n_basis=10,
        domain_range=fd.domain_range,
    )

    fd_basis = fd.to_basis(basis=basis)

    fd_train, fd_test, y_train, _ = sklearn.model_selection.train_test_split(
        fd_basis,
        fat,
        test_size=0.2,
        random_state=10,
    )
    return fd_train, fd_test, y_train


def _create_data_grid(
) -> Tuple[FDataGrid, FDataGrid, FloatArray]:
    X, y = fetch_tecator(return_X_y=True, as_frame=True)
    fd = X.iloc[:, 0].values
    fat = y['fat'].values

    fd_train, fd_test, y_train, _ = sklearn.model_selection.train_test_split(
        fd,
        fat,
        test_size=0.2,
        random_state=10,
    )

    return fd_train, fd_test, y_train


def _create_data_r(
) -> Tuple[FDataGrid, FDataGrid, FloatArray]:
    X, y = fetch_tecator(return_X_y=True, as_frame=True)
    fd = X.iloc[:, 0].values
    fat = y['fat'].values

    return fd[:100], fd[100:110], fat[:100]


class TestKernelRegression(unittest.TestCase):
    """Test Nadaraya-Watson, KNNeighbours and LocalLinearRegression methods."""

    def test_nadaraya_watson(self) -> None:
        """Test Nadaraya-Watson method."""
        # Creating data
        fd_train_basis, fd_test_basis, y_train_basis = _create_data_basis()
        fd_train_grid, fd_test_grid, y_train_grid = _create_data_grid()

        # Test NW method with basis representation and bandwidth=1
        nw_basis = KernelRegression[FDataBasis, np.typing.NDArray[np.float_]](
            kernel_estimator=NadarayaWatsonHatMatrix(bandwidth=1),
        )
        nw_basis.fit(fd_train_basis, y_train_basis)
        y_basis = nw_basis.predict(fd_test_basis)

        np.testing.assert_allclose(
            _nw_alt(
                fd_train_basis,
                fd_test_basis,
                y_train_basis,
                bandwidth=1,
            ),
            y_basis,
        )

        # Test NW method with grid representation and bandwidth=1
        nw_grid = KernelRegression[FDataGrid, np.typing.NDArray[np.float_]](
            kernel_estimator=NadarayaWatsonHatMatrix(bandwidth=1),
        )
        nw_grid.fit(fd_train_grid, y_train_grid)
        y_grid = nw_grid.predict(fd_test_grid)

        np.testing.assert_allclose(
            _nw_alt(
                fd_train_grid,
                fd_test_grid,
                y_train_grid,
                bandwidth=1,
            ),
            y_grid,
        )

    def test_knn(self) -> None:
        """Test K-Nearest Neighbours method."""
        # Creating data
        fd_train_basis, fd_test_basis, y_train_basis = _create_data_basis()
        fd_train_grid, fd_test_grid, y_train_grid = _create_data_grid()

        # Test KNN method with basis representation, n_neighbours=3 and
        # uniform kernel
        knn_basis = KernelRegression[FDataBasis, np.typing.NDArray[np.float_]](
            kernel_estimator=KNeighborsHatMatrix(n_neighbors=3),
        )
        knn_basis.fit(fd_train_basis, y_train_basis)
        y_basis = knn_basis.predict(fd_test_basis)

        np.testing.assert_allclose(
            _knn_alt(
                fd_train_basis,
                fd_test_basis,
                y_train_basis,
                bandwidth=3,
            ),
            y_basis,
        )

        # Test KNN method with grid representation, n_neighbours=3 and
        # uniform kernel
        knn_grid = KernelRegression[FDataGrid, np.typing.NDArray[np.float_]](
            kernel_estimator=KNeighborsHatMatrix(n_neighbors=3),
        )
        knn_grid.fit(fd_train_grid, y_train_grid)
        y_grid = knn_grid.predict(fd_test_grid)

        np.testing.assert_allclose(
            _knn_alt(
                fd_train_grid,
                fd_test_grid,
                y_train_grid,
                bandwidth=3,
            ),
            y_grid,
        )

        # Test KNN method with basis representation, n_neighbours=10 and
        # normal kernel
        knn_basis = KernelRegression(
            kernel_estimator=KNeighborsHatMatrix(
                n_neighbors=10,
                kernel=normal,
            ),
        )
        knn_basis.fit(fd_train_basis, y_train_basis)
        y_basis = knn_basis.predict(fd_test_basis)

        np.testing.assert_allclose(
            _knn_alt(
                fd_train_basis,
                fd_test_basis,
                y_train_basis,
                bandwidth=10,
                kernel=normal,
            ),
            y_basis,
        )

    def test_llr(self) -> None:
        """Test Local Linear Regression method."""
        # Creating data
        fd_train_basis, fd_test_basis, y_train_basis = _create_data_basis()

        llr_basis = KernelRegression[FDataBasis, np.typing.NDArray[np.float_]](
            kernel_estimator=LocalLinearRegressionHatMatrix(bandwidth=1),
        )
        llr_basis.fit(fd_train_basis, y_train_basis)
        y_basis = llr_basis.predict(fd_test_basis)

        np.testing.assert_allclose(
            _llr_alt(
                fd_train_basis,
                fd_test_basis,
                y_train_basis,
                bandwidth=1,
            ),
            y_basis,
        )

    def test_nw_r(self) -> None:
        """Comparison of NW's results with results from fda.usc."""
        X_train, X_test, y_train = _create_data_r()

        nw = KernelRegression[FDataGrid, np.typing.NDArray[np.float_]](
            kernel_estimator=NadarayaWatsonHatMatrix(bandwidth=1),
        )
        nw.fit(X_train, y_train)

        y = nw.predict(X_test)
        result_R = [
            18.245093,
            22.976695,
            9.429236,
            16.852003,
            16.568529,
            8.520466,
            14.943808,
            15.344949,
            8.646862,
            16.576900,
        ]

        np.testing.assert_almost_equal(y, result_R, decimal=3)

    def test_knn_r(self) -> None:
        """Comparison of NW's results with results from fda.usc."""
        X_train, X_test, y_train = _create_data_r()

        knn = KernelRegression[FDataGrid, np.typing.NDArray[np.float_]](
            kernel_estimator=KNeighborsHatMatrix(n_neighbors=3),
        )
        knn.fit(X_train, y_train)

        y = knn.predict(X_test)
        result_R = [
            20.400000,
            24.166667,
            10.900000,
            20.466667,
            16.900000,
            5.433333,
            14.400000,
            11.966667,
            9.033333,
            19.633333,
        ]

        np.testing.assert_almost_equal(y, result_R, decimal=6)


class TestNonOthonormalBasisLLR(unittest.TestCase):
    """Test LocalLinearRegression method with non orthonormal basis."""

    def test_llr_non_orthonormal(self) -> None:
        """Test LocalLinearRegression with monomial basis."""
        coef1 = [[1, 5, 8], [4, 6, 6], [9, 4, 1]]
        coef2 = [[6, 3, 5]]
        basis = MonomialBasis(n_basis=3, domain_range=(0, 3))

        X_train = FDataBasis(coefficients=coef1, basis=basis)
        X = FDataBasis(coefficients=coef2, basis=basis)
        y_train = np.array([8, 6, 1])

        llr = LocalLinearRegressionHatMatrix(
            bandwidth=100,
            kernel=uniform,
        )
        kr = KernelRegression[FDataBasis, np.typing.NDArray[np.float_]](
            kernel_estimator=llr,
        )
        kr.fit(X_train, y_train)
        np.testing.assert_almost_equal(kr.predict(X), 4.35735166)
