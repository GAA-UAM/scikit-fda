"""Test the FPLSRegression class."""
import os

import numpy as np
import pytest
import scipy
from sklearn.cross_decomposition import PLSRegression

from skfda.datasets import fetch_tecator
from skfda.misc.operators import LinearDifferentialOperator
from skfda.misc.regularization import L2Regularization
from skfda.ml.regression import FPLSRegression
from skfda.representation import FData
from skfda.representation.basis import BSplineBasis
from skfda.tests.test_fpls import LatentVariablesModel
from skfda.typing._numpy import NDArrayFloat


class TestFPLSRegression(LatentVariablesModel):
    """Test the FPLSRegression class."""

    def test_sklearn(self) -> None:
        """Compare results with sklearn."""
        # Load the data
        X, y = fetch_tecator(
            return_X_y=True,
        )

        # Fit the model
        fplsr = FPLSRegression[NDArrayFloat, NDArrayFloat](
            n_components=5,
            _integration_weights_X=np.ones(len(X.grid_points[0])),
        )
        fplsr.fit(X, y)

        sklearnpls = PLSRegression(n_components=5, scale=False)
        sklearnpls.fit(X.data_matrix[..., 0], y)

        rtol = 3e-5
        atol = 1e-6
        # Compare the results
        np.testing.assert_allclose(
            fplsr.coef_.flatten(),
            sklearnpls.coef_.T.flatten(),
            rtol=rtol,
            atol=atol,
        )

        # Compare predictions
        np.testing.assert_allclose(
            fplsr.predict(X).flatten(),
            sklearnpls.predict(X.data_matrix[..., 0]).flatten(),
            rtol=rtol,
            atol=atol,
        )

        # Check that the rotations are correc
        np.testing.assert_allclose(
            np.abs(fplsr.fpls_.x_rotations_matrix_).flatten(),
            np.abs(sklearnpls.x_rotations_).flatten(),
            rtol=rtol,
            atol=atol,
        )

    def test_fda_usc_no_reg(self) -> None:
        """
        Test a multivariate regression with no regularization.

        Replication Code:
            data(tecator)
            x <- tecator$absorp.fdata
            y <- tecator$y
            res2 <- fdata2pls(x, y, ncomp=5)
            for (i in res2$l) {
                c <- res2$loading.weigths$data[i, ]
                c <- c / c(sqrt(crossprod(c)))
                print(
                    paste(
                        round(c, 8),
                        collapse = ", "
                    ) # components
                )
            }
        """
        # Results from fda.usc:
        path = os.path.join(
            f"{__file__[:-3]}_data",  # Trim .py ending
            "test_fda_usc_no_reg_data.npy",
        )
        with open(path, "rb") as f:
            r_results = np.load(f, allow_pickle=False)

        signs = np.array([1, -1, 1, -1, 1])

        r_results *= signs

        X, y = fetch_tecator(
            return_X_y=True,
        )

        plsReg = FPLSRegression[FData, NDArrayFloat](
            n_components=5,
            _integration_weights_X=np.ones(len(X.grid_points[0])),
        )
        plsReg.fit(X, y)

        W = plsReg.fpls_.x_weights_
        np.testing.assert_allclose(W, r_results, atol=1e-8)

    def test_fda_usc_reg(self) -> None:
        """
        Test the FPLSRegression with regularization against fda.usc.

        Replication Code:
            data(tecator)
            x=tecator$absorp.fdata
            y=tecator$y
            res2=fdata2pls(x,y,ncomp=5,P=c(0,0,1),lambda = 1000)
            for(i in res2$l){
                c = res2$loading.weigths$data[i,]
                c = c/ c(sqrt(crossprod(c)))
                print(
                    paste(
                        round(c, 8),
                        collapse=", "
                    ) # components
                )
            }
        """
        X, y = fetch_tecator(
            return_X_y=True,
        )

        pen_order = 2
        reg_param = 1000
        # This factor compensates for the fact that the difference
        # matrices in fda.usc are scaled by the mean of the deltas
        # between grid points. This diference is introduced in
        # the function D.penalty (fdata2pc.R:479) in fda.usc.

        grid_points = X[0].grid_points[0]
        grid_step = np.mean(np.diff(grid_points))
        factor1 = grid_step ** (2 * pen_order - 1)

        reg_param *= factor1

        regularization = L2Regularization(
            LinearDifferentialOperator(pen_order),
            regularization_parameter=reg_param,
        )

        # Fit the model
        fplsr = FPLSRegression[FData, NDArrayFloat](
            n_components=5,
            regularization_X=regularization,
            _integration_weights_X=np.ones(len(X.grid_points[0])),
        )
        fplsr.fit(X, y)

        path = os.path.join(
            f"{__file__[:-3]}_data",  # Trim .py ending
            "test_fda_usc_reg_data.npy",
        )
        with open(path, "rb") as f:
            r_results = np.load(f, allow_pickle=False)

        signs = np.array([1, -1, 1, -1, 1])

        w_mat = fplsr.fpls_.x_weights_ @ np.diag(signs)

        # We use a different quadrature, so the results cannot
        # be the same
        np.testing.assert_allclose(w_mat, r_results, atol=1e-2)

    def test_basis_vs_grid(self) -> None:
        """Test that the results are the same in basis and grid."""
        X, y = fetch_tecator(
            return_X_y=True,
        )

        original_grid_points = X.grid_points[0]
        # Express the data in a FDataBasis
        X = X.to_basis(BSplineBasis(n_basis=20))

        # Fit the model
        fplsr = FPLSRegression[FData, NDArrayFloat](n_components=5)
        fplsr.fit(X, y)
        basis_components = fplsr.fpls_.x_rotations_(original_grid_points)

        # Go back to grid
        new_grid_points = np.linspace(
            original_grid_points[0],
            original_grid_points[-1],
            1000,
        )
        X = X.to_grid(grid_points=new_grid_points)

        # Get the weights for the Simpson's rule
        identity = np.eye(len(X.grid_points[0]))
        ss_weights = scipy.integrate.simpson(identity, x=X.grid_points[0])

        # Fit the model with weights
        fplsr = FPLSRegression(
            n_components=5,
            _integration_weights_X=ss_weights,
        )
        fplsr.fit(X, y)

        grid_components = fplsr.fpls_.x_rotations_(original_grid_points)

        np.testing.assert_allclose(
            basis_components,
            grid_components,
            rtol=3e-3,
        )

    @pytest.mark.parametrize("y_features", [1, 5])
    def test_multivariate_regression(self, y_features: int) -> None:
        """Test the multivariate regression.

        Consider both scalar and multivariate responses.
        """
        self.create_latent_variables(n_latent=5, n_samples=100)

        # Check that the model is able to recover the latent variables
        # if it has enough components
        y_observed, y_rotations = self.create_observed_multivariate_variable(
            n_features=y_features,
        )
        X_observed, X_rotations = self.create_observed_multivariate_variable(
            n_features=10,
        )

        plsreg = FPLSRegression[NDArrayFloat, NDArrayFloat](n_components=5)
        plsreg.fit(X_observed, y_observed)

        minimum_score = 0.99
        assert plsreg.score(X_observed, y_observed) > minimum_score

    @pytest.mark.parametrize("discretized", [True, False])
    @pytest.mark.parametrize("n_features", [1, 5])
    @pytest.mark.parametrize("noise_std", [0, 1])
    def test_simple_regresion(
        self,
        discretized: bool,
        n_features: int,
        noise_std: float,
    ) -> None:
        """Test multivariate regressor and functional response."""
        self.create_latent_variables(n_latent=5, n_samples=100)

        # Check that the model is able to recover the latent variables if
        # it has enough components
        y_observed, y_rotations = self.create_observed_multivariate_variable(
            n_features=n_features,
            noise=noise_std,
        )
        X_observed, X_rotations = self.create_observed_functional_variable(
            discretized=discretized,
            noise=noise_std,
        )

        plsreg = FPLSRegression[FData, NDArrayFloat](n_components=5)
        plsreg.fit(X_observed, y_observed)

        minimum_score = 0.99 if noise_std == 0 else 0.98
        assert plsreg.score(X_observed, y_observed) > minimum_score

    @pytest.mark.parametrize("discretized_observed", [True, False])
    @pytest.mark.parametrize("discretized_response", [True, False])
    @pytest.mark.parametrize("noise_std", [0, 1])
    def test_simple_regresion_dataset_functional(
        self,
        discretized_observed: bool,
        discretized_response: bool,
        noise_std: float,
    ) -> None:
        """Test multivariate regressor and functional response."""
        self.create_latent_variables(n_latent=5, n_samples=100)

        # Check that the model is able to recover the latent variables if
        # it has enough components
        X_observed, X_rotations = self.create_observed_functional_variable(
            discretized=discretized_observed,
            noise=noise_std,
        )
        y_observed, y_rotations = self.create_observed_functional_variable(
            discretized=discretized_response,
            noise=noise_std,
        )

        plsreg = FPLSRegression[FData, FData](n_components=5)
        plsreg.fit(X_observed, y_observed)
        minimum_score = 0.99 if noise_std == 0 else 0.98
        assert plsreg.score(X_observed, y_observed) > minimum_score
