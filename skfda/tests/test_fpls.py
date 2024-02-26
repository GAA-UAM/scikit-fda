"""Test the FPLS class."""

from typing import Tuple

import numpy as np
import pytest
from sklearn.cross_decomposition import PLSCanonical

from skfda.datasets import fetch_tecator
from skfda.preprocessing.dim_reduction import FPLS
from skfda.representation import FData, FDataGrid
from skfda.representation.basis import BSplineBasis, FDataBasis
from skfda.typing._numpy import NDArrayFloat


class LatentVariablesModel:
    """Simulate model driven by latent variables."""

    def create_latent_variables(self, n_latent: int, n_samples: int) -> None:
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

    def create_observed_multivariate_variable(
        self,
        n_features: int,
        noise: float = 0,
    ) -> Tuple[NDArrayFloat, NDArrayFloat]:
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

    def create_observed_functional_variable(
        self,
        noise: float = 0,
        discretized: bool = False,
    ) -> Tuple[FData, FData]:
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

    def test_sklearn(self) -> None:
        """Compare results with sklearn."""
        # Load the data
        X, y = fetch_tecator(
            return_X_y=True,
        )

        integration_weights = np.ones(len(X.grid_points[0]))

        # Fit the model
        fpls = FPLS[FDataGrid, NDArrayFloat](
            n_components=3,
            _integration_weights_X=integration_weights,
        )
        fpls.fit(X, y)

        sklearnpls = PLSCanonical(n_components=3, scale=False)
        sklearnpls.fit(X.data_matrix[..., 0], y)

        rtol = 2e-4
        atol = 1e-6

        # Check that the rotations are correct
        np.testing.assert_allclose(
            np.abs(fpls.x_rotations_matrix_).flatten(),
            np.abs(sklearnpls.x_rotations_).flatten(),
            rtol=rtol,
            atol=atol,
        )

        # Check the transformation of X
        np.testing.assert_allclose(
            fpls.transform(X, y),
            sklearnpls.transform(X.data_matrix[..., 0], y),
            rtol=rtol,
            atol=1e-5,
        )

        comp_x, comp_y = fpls.transform(X, y)

        fpls_inv_x, fpls_inv_y = fpls.inverse_transform(comp_x, comp_y)
        sklearnpls_inv_x, sklearnpls_inv_y = sklearnpls.inverse_transform(
            comp_x,
            comp_y,
        )
        # Check the inverse transformation of X
        np.testing.assert_allclose(
            fpls_inv_x.data_matrix.flatten(),
            sklearnpls_inv_x.flatten(),
            rtol=rtol,
            atol=atol,
        )

        # Check the inverse transformation of y
        np.testing.assert_allclose(
            fpls_inv_y.flatten(),
            sklearnpls_inv_y.flatten(),
            rtol=rtol,
            atol=atol,
        )

    # Ignoring WPS210: Found too many local variables
    def test_basis_vs_grid(  # noqa: WPS210
        self,
    ) -> None:
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
        fpls = FPLS[FData, FData](n_components=n_components)
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
        fpls_grid = FPLS[FData, FData](n_components=n_components)
        fpls_grid.fit(X_observed_grid, y_observed_grid)

        # Check that the results are the same
        np.testing.assert_allclose(
            np.abs(fpls.x_rotations_(self.grid_points)),
            np.abs(fpls_grid.x_rotations_(self.grid_points)),
            rtol=5e-3,
        )

        np.testing.assert_allclose(
            np.abs(fpls.y_rotations_(self.grid_points)),
            np.abs(fpls_grid.y_rotations_(self.grid_points)),
            rtol=5e-3,
        )

        # Check that the transform is the same
        np.testing.assert_allclose(
            np.abs(fpls.transform(X_observed, y_observed)),
            np.abs(fpls_grid.transform(X_observed_grid, y_observed_grid)),
            rtol=5e-3,
        )
        # Check the inverse transform
        comp_x, comp_y = fpls.transform(X_observed, y_observed)

        fpls_inv_x, fpls_inv_y = fpls.inverse_transform(comp_x, comp_y)
        fpls_grid_x, fpsl_grid_y = fpls_grid.inverse_transform(
            comp_x,
            comp_y,
        )
        # Check the inverse transformation of X
        np.testing.assert_allclose(
            fpls_inv_x(grid_points),
            fpls_grid_x(grid_points),
            rtol=7e-2,
        )

        # Check the inverse transformation of y
        np.testing.assert_allclose(
            fpls_inv_y(grid_points),
            fpsl_grid_y(grid_points),
            rtol=0.13,
        )

    def _generate_random_matrix_by_rank(self, n_samples, n_features, rank):

        random_data = np.random.random(n_samples * rank).reshape(
            n_samples,
            rank,
        )
        if rank == n_features:
            return random_data
        repeated_data = np.array([random_data[:, 0]] * (n_features - rank)).T

        return np.concatenate(
            [random_data, repeated_data],
            axis=1,
        )

    @pytest.mark.parametrize("rank", [1, 2, 5])
    def test_collinear_matrix(
        self,
        rank: int,
    ) -> None:
        """Check that the behaviour is correct with collinear matrices."""
        n_samples = 100
        n_features = 10
        np.random.seed(0)

        X = self._generate_random_matrix_by_rank(
            n_samples=n_samples,
            n_features=n_features,
            rank=rank,
        )
        y = self._generate_random_matrix_by_rank(
            n_samples=n_samples,
            n_features=n_features,
            rank=rank,
        )

        fpls = FPLS(n_components=rank)
        fpls.fit(X, y)

        fpls = FPLS(n_components=5)

        # Check that a warning is raised when the rank is lower than the
        # number of components
        if rank < 5:
            with pytest.warns(UserWarning):
                fpls.fit(X, y)
        else:
            fpls.fit(X, y)

        # Check that only as many components as rank are returned
        assert fpls.x_weights_.shape == (n_features, rank)

        # Check that the waring is not raised when the number of components
        # is not set
        fpls = FPLS().fit(X, y)
        # Check that only as many components as rank are returned
        assert fpls.x_weights_.shape == (n_features, rank)

    def test_number_of_components(
        self,
    ) -> None:
        """Check error when number of components is too large."""
        n_samples = 100
        n_features = 10
        np.random.seed(0)

        X = self._generate_random_matrix_by_rank(
            n_samples=n_samples,
            n_features=n_features,
            rank=n_features,
        )
        y = self._generate_random_matrix_by_rank(
            n_samples=n_samples,
            n_features=n_features,
            rank=n_features,
        )

        with pytest.raises(ValueError):
            FPLS(n_components=n_features + 1).fit(X, y)
