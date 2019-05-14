import sklearn.utils
import numpy as np
from scipy.stats import multivariate_normal
import scipy.integrate
from ..misc import covariances
from .. import FDataGrid
from ..representation.interpolation import SplineInterpolator
from ..preprocessing.registration import normalize_warping


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

    r"""Generate sinusoidal proccess.

    Each sample :math:`x_i(t)` is generated as:

    .. math::

        x_i(t) = \alpha_i \sin(\omega t + \phi_i) + \epsilon_i(t)


    where :math:`\omega=\frac{2 \pi}{\text{period}}`. Amplitudes
    :math:`\alpha_i` and phases :math:`\phi_i` are normally distributed.
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

    random_state = sklearn.utils.check_random_state(random_state)

    t = np.linspace(start, stop, n_features)

    alpha = np.diag(random_state.normal(amplitude_mean, amplitude_std,
                                        n_samples))

    phi = np.outer(random_state.normal(phase_mean, phase_std,  n_samples),
                   np.ones(n_features))

    error = random_state.normal(0, error_std, (n_samples, n_features))


    y = alpha @ np.sin((2*np.pi/period)*t + phi) + error

    return FDataGrid(sample_points=t, data_matrix=y)


def make_multimodal_landmarks(n_samples: int=15, *, n_modes: int=1,
                              ndim_domain: int=1, ndim_image: int = 1,
                              start: float=-1, stop: float=1, std: float=.05,
                              random_state=None):
    """Generate landmarks points.

    Used by :func:`make_multimodal_samples` to generate the location of the
    landmarks.

    Generates a matrix containing the landmarks or locations of the modes of the
    samples generates by :func:`make_multimodal_samples`.

    If the same random state is used when generating the landmarks and
    multimodal samples, these will correspond to the position of the modes of
    the multimodal samples.

    Args:
        n_samples: Total number of samples.
        n_modes: Number of modes of each sample.
        ndim_domain: Number of dimensions of the domain.
        ndim_image: Number of dimensions of the image
        start: Starting point of the samples. In multidimensional objects the
            starting point of the axis.
        stop: Ending point of the samples. In multidimensional objects the
            ending point of the axis.
        std: Standard deviation of the variation of the modes location.
        random_state: Random state.

    Returns:
        :class:`np.ndarray` with the location of the modes, where the component
        (i,j,k) corresponds to the mode k of the image dimension j of the sample
        i.
    """

    random_state = sklearn.utils.check_random_state(random_state)

    modes_location = np.linspace(start, stop, n_modes + 2)[1:-1]
    modes_location = np.repeat(modes_location[:, np.newaxis], ndim_domain,
                               axis=1)


    variation = random_state.multivariate_normal((0,) * ndim_domain,
                                                 std * np.eye(ndim_domain),
                                                 size=(n_samples,
                                                       ndim_image,
                                                       n_modes))

    return modes_location + variation


def make_multimodal_samples(n_samples: int=15, *, n_modes: int=1,
                            points_per_dim: int=100, ndim_domain: int=1,
                            ndim_image: int=1, start: float=-1, stop: float=1.,
                            std: float=.05, mode_std: float=.02,
                            noise: float=.0, modes_location=None,
                            random_state=None):

    """Generate multimodal samples.

    Each sample :math:`x_i(t)` is proportional to a gaussian mixture, generated
    as the sum of multiple pdf of multivariate normal distributions with
    different means.

    .. math::

        x_i(t) \\propto \\sum_{n=1}^{\\text{n_modes}} \\exp \\left (
        {-\\frac{1}{2\\sigma} (t-\\mu_n)^T \\mathbb{1} (t-\\mu_n)} \\right )

    Where :math:`\\mu_n=\\text{mode_location}_n+\\epsilon` and :math:`\\epsilon`
    is normally distributed, with mean :math:`\\mathbb{0}` and standard
    deviation given by the parameter `std`.

    Args:
        n_samples: Total number of samples.
        n_modes: Number of modes of each sample.
        points_per_dim: Points per sample. If the object is multidimensional
            indicates the number of points for each dimension in the domain.
            The sample will have :math:
            `\\text{points_per_dim}^\\text{ndim_domain}` points of
            discretization.
        ndim_domain: Number of dimensions of the domain.
        ndim_image: Number of dimensions of the image
        start: Starting point of the samples. In multidimensional objects the
            starting point of each axis.
        stop: Ending point of the samples. In multidimensional objects the
            ending point of each axis.
        std: Standard deviation of the variation of the modes location.
        mode_std: Standard deviation :math:`\\sigma` of each mode.
        noise: Standard deviation of Gaussian noise added to the data.
        modes_location:  List of coordinates of each mode.
        random_state: Random state.

    Returns:
        :class:`FDataGrid` object comprising all the samples.
    """

    random_state = sklearn.utils.check_random_state(random_state)


    if modes_location is None:

        location = make_multimodal_landmarks(n_samples=n_samples,
                                             n_modes=n_modes,
                                             ndim_domain=ndim_domain,
                                             ndim_image=ndim_image,
                                             start=start,
                                             stop=stop,
                                             std=std,
                                             random_state=random_state)

    else:
        location = np.asarray(modes_location)

        shape = (n_samples, ndim_image, n_modes, ndim_domain)
        location = location.reshape(shape)


    axis = np.linspace(start, stop, points_per_dim)

    if ndim_domain == 1:
        sample_points = axis
        evaluation_grid = axis
    else:
        sample_points = np.repeat(axis[:, np.newaxis], ndim_domain, axis=1).T

        meshgrid = np.meshgrid(*sample_points)

        evaluation_grid = np.empty(meshgrid[0].shape + (ndim_domain,))

        for i in range(ndim_domain):
            evaluation_grid[..., i] = meshgrid[i]

    # Data matrix of the grid
    shape = (n_samples,) + ndim_domain * (points_per_dim,) + (ndim_image,)
    data_matrix = np.zeros(shape)

    # Covariance matrix of the samples
    cov = mode_std * np.eye(ndim_domain)

    for i in range(n_samples):
        for j in range(ndim_image):
            for k in range(n_modes):
                data_matrix[i,...,j] += multivariate_normal.pdf(evaluation_grid,
                                                                mean=location[i,j,k],
                                                                cov=cov)

    # Constant to make modes value aprox. 1
    data_matrix *= (2*np.pi*mode_std)**(ndim_domain/2)

    data_matrix += random_state.normal(0, noise, size=data_matrix.shape)

    return FDataGrid(sample_points=sample_points, data_matrix=data_matrix)


def make_random_warping(n_samples: int=15, n_features: int=100, *,
                        start: float=0., stop: float=1., sigma: float=1.,
                        shape_parameter: float=50, n_random: int=4,
                        random_state=None):
    r"""Generate random warping functions.

     Let :math:`v(t)` be a randomly generated function defined in :math:`[0,1]`

     .. math::
        v(t) = \sum_{j=0}^{N} a_j \sin(\frac{2 \pi j}{K}t) + b_j
        \cos(\frac{2 \pi j}{K}t)

    where :math:`a_j, b_j \sim N(0, \sigma)`.

    The random warping it is constructed making an exponential map to
    :math:`\Gamma`.

     .. math::
        \gamma(t) = \int_0^t \left ( \frac{\sin(\|v\|)}{\|v\|} v(s) +
        \cos(\|v\|) \right )^2 ds

    An affine traslation it is used to define the warping in :math:`[a,b]`.

    The smoothing and shape of the warpings can be controlling changing
    :math:`N`, :math:`\sigma` and :math:`K= 1 + \text{shape_parameter}`.


    Args:
        n_samples: Total number of samples. Defaults 15.
        n_features: The total number of trajectories. Defaults 100.
        start: Starting point of the samples. Defaults 1.
        stop: Ending point of the samples. Defaults 0.
        sigma: Parameter to control the variance of the samples. Defaults 1.
        shape_parameter: Parameter to control the shape of the warpings.
            Should be a positive value. When the shape parameter goes to
            infinity the warpings generated are :math:`\gamma_{id}`.
            Defaults to 50.
        n_random: Number of random sines and cosines to be sum to construct
            the warpings.
        random_state: Random state.

    Returns:
        :class:`FDataGrid` object comprising all the samples.

     """
     # Based on the original implementation of J. D. Tucker in the
     # package python_fdasrsf <https://github.com/jdtuck/fdasrsf_python>.

    random_state = sklearn.utils.check_random_state(random_state)


    freq = shape_parameter + 1

    # Frequency
    omega = 2 * np.pi / freq
    sqrt2 = np.sqrt(2)
    sqrt_sigma = np.sqrt(sigma)

     # Originally it is compute in (0,1), then it is rescaled
    time = np.outer(np.linspace(0, 1, n_features), np.ones(n_samples))

    # Operates trasposed to broadcast dimensions
    v = np.outer(np.ones(n_features),
                 random_state.normal(scale=sqrt_sigma, size=n_samples))

    for j in range(2,2+n_random):
        alpha = random_state.normal(scale=sqrt_sigma, size=(2, n_samples))
        alpha *= sqrt2
        v += alpha[0] * np.cos(j*omega*time)
        v += alpha[1] * np.sin(j*omega*time)

    v -= v.mean(axis=0)

    # Equivalent to RN norm (equispaced monitorization points)
    v_norm = np.linalg.norm(v, axis=0)
    v_norm /= np.sqrt(n_features)

    # Exponential map of vectors
    v *= np.sin(v_norm) / v_norm
    v += np.cos(v_norm)
    np.square(v, out=v)

    # Creation of FDataGrid in the corresponding domain
    data_matrix = scipy.integrate.cumtrapz(v, dx=1./n_features, initial=0,
                                           axis=0)
    warping = FDataGrid(data_matrix.T, sample_points=time[:,0])
    warping = normalize_warping(warping, domain_range=(start, stop))
    warping.interpolator = SplineInterpolator(interpolation_order=3,
                                              monotone=True)


    return warping
