import sklearn.utils
import numpy as np
from .. import covariances
from ..grid import FDataGrid


def make_gaussian_process(n_samples: int=100, n_features: int=100, *,
                          start: float=0., stop: float=1.,
                          mean=0, covariance=None, noise_sigma: float=0.,
                          random_state=None):
    """Generate Gaussian process trajectories.

        Args:
            n_samples: The total number of trajectories.
            n_features: The total number of trajectories.

        Returns:
            :obj:FDataGrid: Returns a FDataGrid object comprising all the
            trajectories.

    """

    random_state = sklearn.utils.check_random_state(random_state)

    x = np.linspace(start, stop, n_features)

    if covariance is None:
        covariance = covariances.Brownian()

    cov = covariances._execute_covariance(covariance, x, x)

    # Para evitar problemas numéricos
    cov += np.eye(n_features) * 1.0e-15

    mu = np.zeros(n_features)

    y = random_state.multivariate_normal(mu, cov, n_samples)

    if noise_sigma:
        noise = np.random.normal(scale=noise_sigma,
                                 size=(n_samples, n_features))

        y += noise

    return FDataGrid(sample_points=x, data_matrix=y)
