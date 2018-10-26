import sklearn.utils
import numpy as np
from .. import covariances
from ..grid import FDataGrid


def make_gaussian_process(n_samples: int=100, n_features: int=100, *,
                          start: float=0., stop: float=1.,
                          mean=0, cov=None, noise: float=0.,
                          random_state=None):
    """Generate Gaussian process trajectories.

        Args:
            n_samples: The total number of trajectories.
            n_features: The total number of trajectories.
            start: Starting point of the trajectories.
            stop: Ending point of the trajectories.
            mean: The mean function of the process. Can be a callable accepting
                  a vector with the locations, or a vector with length
                  ``n_features``.
            cov: The covariance function of the process. Can be a
                  callable accepting two vectors with the locations, or a
                  matrix with size ``n_features`` x ``n_features``.
            noise: Standard deviation of Gaussian noise added to the data.
            random_state: Random state.

        Returns:
            :class:`FDataGrid` object comprising all the trajectories.

    """

    random_state = sklearn.utils.check_random_state(random_state)

    x = np.linspace(start, stop, n_features)

    if cov is None:
        cov = covariances.Brownian()

    covariance = covariances._execute_covariance(cov, x, x)

    # To prevent numerical problems
    covariance += np.eye(n_features) * 1.0e-15

    mu = np.zeros(n_features)
    if callable(mean):
        mean = mean(x)
    mu += mean

    y = random_state.multivariate_normal(mu, covariance, n_samples)

    if noise:
        noise_sample = np.random.normal(scale=noise,
                                        size=(n_samples, n_features))

        y += noise_sample

    return FDataGrid(sample_points=x, data_matrix=y)
