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

    if noise:
        covariance += np.eye(n_features) * noise ** 2

    mu = np.zeros(n_features)
    if callable(mean):
        mean = mean(x)
    mu += mean

    y = random_state.multivariate_normal(mu, covariance, n_samples)

    return FDataGrid(sample_points=x, data_matrix=y)


def make_sinusoidal_process(n_samples: int=15, n_features: int=100, *,
                            start: float=0., stop: float=1., period: float=1.,
                            phase_mean: float=0., phase_std: float=.6,
                            amplitude_mean: float=1., amplitude_std: float=.05,
                            error_std: float=.2, random_state=None):

    """Generate sinusoidal proccess.

    Each sample :math:`x_i(t)` is generated as:

    .. math::

        x_i(t) = \\alpha_i \sin(\omega t + \phi_i) + \epsilon_i(t)


    where :math:`\omega=\\frac{2 \pi}{\\text{period}}`. Amplitudes
    :math:`\\alpha_i` and phases :math:`\phi_i` are normally distributed.
    :math:`\epsilon_i(t)` is a gaussian white noise process.

    Args:
        n_samples: Total number of samples.
        n_features: Points per sample.
        start: Starting point of the samples.
        stop: Ending point of the samples.
        period: Period of the sine function.
        phase_mean: Mean of the phase.
        phase_std: Standard deviation of the phase.
        amplitude_mean: Mean of the amplitude.
        amplitude_std: Standard deviation of the amplitude.
        error_std: Standard deviation of the gaussian Noise.
        random_state: Random state.

    Returns:
        :class:`FDataGrid` object comprising all the samples.

    """

    sklearn.utils.check_random_state(random_state)

    t = np.linspace(start, stop, n_features)

    alpha = np.diag(np.random.normal(amplitude_mean, amplitude_std, n_samples))

    phi = np.outer(np.random.normal(phase_mean, phase_std,  n_samples),
                   np.ones(n_features))

    error = np.random.normal(0, error_std, (n_samples, n_features))


    y = alpha @ np.sin((2*np.pi/period)*t + phi) + error

    return FDataGrid(sample_points=t, data_matrix=y)
