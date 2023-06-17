"""Test the FPLS class."""
import os

import numpy as np
import pytest
import scipy
from sklearn.cross_decomposition import PLSCanonical

from skfda.datasets import fetch_tecator
from skfda.misc.operators import LinearDifferentialOperator
from skfda.misc.regularization import L2Regularization
from skfda.preprocessing.dim_reduction import FPLS
from skfda.representation.basis import BSplineBasis, FDataBasis

class LatentVariablesModel:
    """Simulate model driven by latent variables."""

    def create_latent_variables(self, n_latent, n_samples):
        """Create latent variables for testing."""
        self.rng = np.random.RandomState(0)
        self.n_latent = n_latent
        self.n_samples = n_samples
        self.grid_points = np.linspace(0, 1, 100)

        # Get the means of the latent variables
        latent_means = self.rng.uniform(low=1, high=10, size=(n_latent))

        # Sample the variables
        self.latent_samples = np.array(
            [
                self.rng.normal(loc=mean, scale=1, size=n_samples)
                for mean in latent_means
            ],
        ).T

    def create_observed_multivariate_variable(self, n_features, noise=0):
        """Create observed multivariate variable for testing."""
        rotations = self.rng.uniform(
            low=0,
            high=10,
            size=(n_features, self.n_latent),
        )

        observed_values = self.latent_samples @ rotations.T
        observed_noise = noise * self.rng.normal(
            size=(self.n_samples, n_features),
        )
        observed_values += observed_noise

        return observed_values, rotations

    def create_observed_functional_variable(self, noise=0, discretized=False):
        """Create observed functional variable for testing."""
        n_basis = 20
        basis = BSplineBasis(n_basis=n_basis)

        rotation_coef = self.rng.uniform(
            low=0,
            high=10,
            size=(self.n_latent, n_basis),
        )
        rotation = FDataBasis(basis, rotation_coef)

        observed_coef = self.latent_samples @ rotation_coef
        observed_func = FDataBasis(basis, observed_coef)

        if discretized:
            observed_func = observed_func.to_grid(
                grid_points=self.grid_points,
            )

            func_noise = noise * self.rng.normal(size=(self.n_samples, 100))
            observed_func.data_matrix[..., 0] += func_noise

        else:
            observed_func.coefficients += noise * self.rng.normal(
                size=(self.n_samples, n_basis),
            )

        return observed_func, rotation
 

class TestFPLS(LatentVariablesModel):
    """Test the FPLS class."""

    def test_sklearn(self):
        """Compare results with sklearn."""
        # Load the data
        X, y = fetch_tecator(
            return_X_y=True,
        )

        integration_weights = np.ones(len(X.grid_points[0]))

        # Fit the model
        fpls = FPLS(n_components=3,
                    integration_weights_X=integration_weights,
        )
        fpls.fit(X, y)

        sklearnpls = PLSCanonical(n_components=3, scale=False)
        sklearnpls.fit(X.data_matrix[..., 0], y)

        rtol = 2e-4
        atol = 1e-6

        # Check that the rotations are correct
        np.testing.assert_allclose(
            np.abs(fpls.x_rotations_).flatten(),
            np.abs(sklearnpls.x_rotations_).flatten(),
            rtol=rtol,
            atol=atol,
        )

    
    def test_basis_vs_grid(self,):
        """Test that the results are the same in basis and grid."""

        n_components = 5
        self.create_latent_variables(n_latent=n_components, n_samples=100)

        # Create the observed variable as basis variables
        X_observed, X_rotations = self.create_observed_functional_variable(
            discretized=False,
            noise=0,
        )
        y_observed, y_rotations = self.create_observed_functional_variable(
            discretized=False,
            noise=0,
        )

        # Fit the model
        fpls = FPLS(n_components=n_components)
        fpls.fit(X_observed, y_observed)

        # Convert the observed variables to grid
        grid_points = np.linspace(0, 1, 2000)
        X_observed_grid = X_observed.to_grid(
            grid_points=grid_points,
        )

        y_observed_grid = y_observed.to_grid(
            grid_points=grid_points,
        )

        # Fit the model with the grid variables
        fpls_grid = FPLS(n_components=n_components)
        fpls_grid.fit(X_observed_grid, y_observed_grid)


        # Check that the results are the same
        np.testing.assert_allclose(
            np.abs(fpls.components_x_(self.grid_points)),
            np.abs(fpls_grid.components_x_(self.grid_points)),
            rtol=5e-3,
        )

        np.testing.assert_allclose(
            np.abs(fpls.components_y_(self.grid_points)),
            np.abs(fpls_grid.components_y_(self.grid_points)),
            rtol=5e-3,
        )


        

