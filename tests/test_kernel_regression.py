"""Test kernel regression method."""
import unittest
from typing import Tuple

import numpy as np
import sklearn

import skfda
from skfda.misc.hat_matrix import (
    KNeighborsHatMatrix,
    LocalLinearRegressionHatMatrix,
    NadarayaWatsonHatMatrix,
)
from skfda.misc.kernels import normal, uniform
from skfda.misc.metrics import l2_distance
from skfda.ml.regression._kernel_regression import KernelRegression
from skfda.representation.basis import FDataBasis
from skfda.representation.grid import FDataGrid


def _nw_alt(fd_train, fd_test, y_train, *, bandwidth, kernel=None):
    if kernel is None:
        kernel = normal

    y = np.zeros(fd_test.n_samples)
    for i in range(fd_test.n_samples):
        w = kernel(l2_distance(fd_train, fd_test[i]) / bandwidth)
        y[i] = (w @ y_train) / sum(w)

    return y


def _knn_alt(fd_train, fd_test, y_train, *, bandwidth, kernel=None):
    if kernel is None:
        kernel = uniform

    y = np.zeros(fd_test.n_samples)

    for i in range(fd_test.n_samples):
        d = l2_distance(fd_train, fd_test[i])
        h = sorted(d)[bandwidth - 1] + 1.0e-15
        w = kernel(d / h)
        y[i] = (w @ y_train) / sum(w)

    return y


def _llr_alt(fd_train, fd_test, y_train, *, bandwidth, kernel=None):
    if kernel is None:
        kernel = normal

    y = np.zeros(fd_test.n_samples)

    for i in range(fd_test.n_samples):
        d = l2_distance(fd_train, fd_test[i])
        W = np.diag(kernel(d / bandwidth))

        C = np.concatenate(
            (
                (np.ones(fd_train.n_samples))[:, np.newaxis],
                (fd_train - fd_test[i]).coefficients,
            ),
            axis=1,
        )

        M = np.linalg.inv(np.linalg.multi_dot([C.T, W, C]))
        y[i] = np.linalg.multi_dot([M, C.T, W, y_train])[0]

    return y


def _create_data_basis() -> Tuple[FDataBasis, FDataBasis, np.ndarray]:
    X, y = skfda.datasets.fetch_tecator(return_X_y=True, as_frame=True)
    fd = X.iloc[:, 0].values
    fat = y['fat'].values

    basis = skfda.representation.basis.BSpline(
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


def _create_data_grid() -> Tuple[FDataGrid, FDataGrid, np.ndarray]:
    X, y = skfda.datasets.fetch_tecator(return_X_y=True, as_frame=True)
    fd = X.iloc[:, 0].values
    fat = y['fat'].values

    fd_train, fd_test, y_train, _ = sklearn.model_selection.train_test_split(
        fd,
        fat,
        test_size=0.2,
        random_state=10,
    )

    return fd_train, fd_test, y_train


class TestKernelRegression(unittest.TestCase):
    """Test Nadaraya-Watson, KNNeighbours and LocalLinearRegression methods."""

    def test_nadaraya_watson(self) -> None:
        """Test Nadaraya-Watson method."""
        # Creating data
        fd_train_basis, fd_test_basis, y_train_basis = _create_data_basis()
        fd_train_grid, fd_test_grid, y_train_grid = _create_data_grid()

        # Test NW method with basis representation and bandwidth=1
        nw_basis = KernelRegression(
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
        nw_grid = KernelRegression(
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
        knn_basis = KernelRegression(
            kernel_estimator=KNeighborsHatMatrix(bandwidth=3),
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
        knn_grid = KernelRegression(
            kernel_estimator=KNeighborsHatMatrix(bandwidth=3),
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
            kernel_estimator=KNeighborsHatMatrix(bandwidth=10, kernel=normal),
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

        llr_basis = KernelRegression(
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
