from __future__ import annotations

import itertools
from typing import Any, Callable, Sequence, Union

import numpy as np
import scipy.integrate
from scipy.stats import multivariate_normal
from typing_extensions import Protocol

from .._utils import _cartesian_product, _to_grid_points, normalize_warping
from ..misc.covariances import Brownian, CovarianceLike, _execute_covariance
from ..misc.validation import validate_random_state
from ..representation import FDataGrid
from ..representation.interpolation import SplineInterpolation
from ..typing._base import DomainRangeLike, GridPointsLike, RandomStateLike
from ..typing._numpy import ArrayLike, NDArrayFloat

MeanCallable = Callable[[np.ndarray], np.ndarray]
CovarianceCallable = Callable[[np.ndarray, np.ndarray], np.ndarray]

MeanLike = Union[float, NDArrayFloat, MeanCallable]
SDETerm = Callable[[float, NDArrayFloat], NDArrayFloat]


class InitialValueGenerator(Protocol):
    """Class to represent SDE initial value generators.

    This is intented to be an interface compatible with the rvs method of
    SciPy distributions.
    """

    def __call__(
        self,
        size: int,
        random_state: RandomStateLike,
    ) -> NDArrayFloat:
        """Interface of initial value generator."""


def euler_maruyama(  # noqa: WPS210
    initial_condition: ArrayLike | InitialValueGenerator,
    n_grid_points: int = 100,
    drift: SDETerm | ArrayLike | None = None,
    diffusion: SDETerm | ArrayLike | None = None,
    n_samples: int | None = None,
    start: float = 0.0,  # noqa: WPS358 -- Distinguish float from integer
    stop: float = 1.0,
    diffusion_matricial_term: bool = True,
    random_state: RandomStateLike = None,
) -> FDataGrid:
    r"""Numerical integration of an Itô SDE using the Euler-Maruyana scheme.

    An SDE can be expressed with the following formula

    .. math::

        d\mathbf{X}(t) = \mathbf{F}(t, \mathbf{X}(t))dt + \mathbf{G}(t,
        \mathbf{X}(t))d\mathbf{W}(t).

    In this equation, :math:`\mathbf{X} = (X^{(1)}, X^{(2)}, ... , X^{(n)})
    \in \mathbb{R}^q` is a vector that represents the state of the stochastic
    process. The function :math:`\mathbf{F}(t, \mathbf{X}) = (F^{(1)}(t,
    \mathbf{X}), ..., F^{(q)}(t, \mathbf{X}))` is called drift and refers
    to the deterministic component of the equation. The function
    :math:`\mathbf{G} (t, \mathbf{X}) = (G^{i, j}(t, \mathbf{X}))_{i=1, j=1}
    ^{q, m}` is denoted as the diffusion term and refers to the stochastic
    component of the evolution. :math:`\mathbf{W}(t)` refers to a Wiener
    process (Standard Brownian motion) of dimension :math:`m`. Finally,
    :math:`q` refers to the dimension of the variable :math:`\mathbf{X}`
    (dimension of the codomain) and :math:`m` to the dimension of the noise.

    Euler-Maruyama's method computes the approximated solution using the
    Markov chain

    .. math::

        X_{n + 1}^{(i)} = X_n^{(i)} + F^{(i)}(t_n, \mathbf{X}_n)\Delta t_n +
        \sum_{j=1}^m G^{i,j}(t_n, \mathbf{X}_n)\sqrt{\Delta t_n}\Delta Z_n^j,

    where :math:`X_n^{(i)}` is the approximated value of :math:`X^{(i)}(t_n)`
    and the :math:`\mathbf{Z}_m` are independent, identically distributed
    :math:`m`-dimensional standard normal random variables.

    Args:
        initial_condition: Initial condition of the SDE. It can have one of
            three formats: An starting initial value from which to
            calculate *n_samples* trajectories. An array of initial values.
            For each starting point a trajectory will be calculated. A
            function that generates random numbers or vectors. It should
            have two parameters called size and random_state and it should
            return an array.
        n_grid_points: The total number of points of evaluation.
        drift: Drift coefficient (:math:`F(t,\mathbf{X})` in the equation).
        diffusion: Diffusion coefficient (:math:`G(t,\mathbf{X})` in the
            equation).
        n_samples: Number of trajectories integrated.
        start: Starting time of the trajectories.
        stop: Ending time of the trajectories.
        diffusion_matricial_term: True if the diffusion coefficient is a
            matrix.
        random_state: Random state.


    Returns:
        :class:`FDataGrid` object comprising all the trajectories.

    See also:
        :func:`make_gaussian_process`: Simpler function for generating
        Gaussian processes.

    Examples:
        Example of the use of euler_maruyama for an Ornstein-Uhlenbeck process
        that has the equation:

        ..  math:

            dX(t) = -A(X(t) - \mu)dt + BdW(t)

        >>> from scipy.stats import norm
        >>> A = 1
        >>> mu = 3
        >>> B = 0.5
        >>> def ou_drift(t: float, x: np.ndarray) -> np.ndarray:
        ...     return -A * (x - mu)
        >>> initial_condition = norm().rvs
        >>>
        >>> trajectories = euler_maruyama(
        ...     initial_condition=initial_condition,
        ...     n_samples=10,
        ...     drift=ou_drift,
        ...     diffusion=B,
        ... )

    """
    random_state = validate_random_state(random_state)

    if n_samples is None:
        if callable(initial_condition):
            raise ValueError(
                "Invalid initial conditions. If a function is given, the "
                "n_samples argument must be included.",
            )

        initial_values = np.atleast_1d(initial_condition)
        n_samples = len(initial_values)
    else:
        if callable(initial_condition):
            initial_values = initial_condition(
                size=n_samples,
                random_state=random_state,
            )
        else:
            initial_condition = np.atleast_1d(initial_condition)
            dim_codomain = len(initial_condition)
            initial_values = (
                initial_condition
                * np.ones((n_samples, dim_codomain))
            )

    if initial_values.ndim == 1:
        initial_values = initial_values[:, np.newaxis]
    elif initial_values.ndim > 2:
        raise ValueError(
            "Invalid initial conditions. Each of the starting points "
            "must be a flat array.",
        )
    (n_samples, dim_codomain) = initial_values.shape

    if dim_codomain == 1:
        diffusion_matricial_term = False

    if drift is None:
        drift = 0.0  # noqa: WPS358 -- Distinguish float from integer

    if callable(drift):
        drift_function = drift
    else:
        def constant_drift(  # noqa: WPS430 -- We need internal functions
            t: float,
            x: NDArrayFloat,
        ) -> NDArrayFloat:
            return np.atleast_1d(drift)

        drift_function = constant_drift

    if diffusion is None:
        if diffusion_matricial_term:
            diffusion = np.eye(dim_codomain)
        else:
            diffusion = 1.0

    if callable(diffusion):
        diffusion_function = diffusion
    else:
        def constant_diffusion(  # noqa: WPS430 -- We need internal functions
            t: float,
            x: NDArrayFloat,
        ) -> NDArrayFloat:
            return np.atleast_1d(diffusion)

        diffusion_function = constant_diffusion

    def vector_diffusion_times_noise(  # noqa: WPS430 We need internal functons
        t_n: float,
        x_n: NDArrayFloat,
        noise: NDArrayFloat,
    ) -> NDArrayFloat:
        return diffusion_function(t_n, x_n) * noise

    def matrix_diffusion_times_noise(  # noqa: WPS430 We need internal functons
        t_n: float,
        x_n: NDArrayFloat,
        noise: NDArrayFloat,
    ) -> Any:
        return np.einsum(
            '...dj, ...j -> ...d',
            diffusion_function(t_n, x_n),
            noise,
        )

    dim_noise = dim_codomain

    if diffusion_matricial_term:
        diffusion_times_noise = matrix_diffusion_times_noise
        dim_noise = diffusion_function(start, initial_values).shape[-1]
    else:
        diffusion_times_noise = vector_diffusion_times_noise

    data_matrix = np.zeros((n_samples, n_grid_points, dim_codomain))
    times = np.linspace(start, stop, n_grid_points)
    delta_t = times[1:] - times[:-1]
    noise = random_state.standard_normal(
        size=(n_samples, n_grid_points - 1, dim_noise),
    )
    data_matrix[:, 0] = initial_values

    for n in range(n_grid_points - 1):
        t_n = times[n]
        x_n = data_matrix[:, n]

        data_matrix[:, n + 1] = (
            x_n
            + delta_t[n] * drift_function(t_n, x_n)
            + diffusion_times_noise(t_n, x_n, noise[:, n])
            * np.sqrt(delta_t[n])
        )

    return FDataGrid(
        grid_points=times,
        data_matrix=data_matrix,
    )


def make_gaussian(
    n_samples: int = 100,
    *,
    grid_points: GridPointsLike,
    domain_range: DomainRangeLike | None = None,
    mean: MeanLike = 0,
    cov: CovarianceLike | None = None,
    noise: float = 0,
    random_state: RandomStateLike = None,
) -> FDataGrid:
    """
    Generate Gaussian random fields.

    Args:
        n_samples: The total number of trajectories.
        grid_points: Sample points for the evaluation grid of the
            Gaussian field.
        domain_range: The domain range of the returned functional
            observations.
        mean: The mean function of the random field. Can be a callable
            accepting a vector with the locations, or a vector with
            appropriate size.
        cov: The covariance function of the process. Can be a
            callable accepting two vectors with the locations, or a
            matrix with appropriate size. By default,
            the Brownian covariance function is used.
        noise: Standard deviation of Gaussian noise added to the data.
        random_state: Random state.

    Returns:
        :class:`FDataGrid` object comprising all the trajectories.

    See also:
        :func:`make_gaussian_process`: Simpler function for generating
        Gaussian processes.

    """
    random_state = validate_random_state(random_state)

    if cov is None:
        cov = Brownian()

    grid_points = _to_grid_points(grid_points)

    input_points = _cartesian_product(grid_points)

    covariance = _execute_covariance(
        cov,
        input_points,
        input_points,
    )

    if noise:
        covariance += np.eye(len(covariance)) * noise ** 2

    mu = np.zeros(len(input_points))
    if callable(mean):
        mean = mean(input_points)

    mu += np.ravel(mean)

    data_matrix = random_state.multivariate_normal(
        mu.ravel(),
        covariance,
        n_samples,
    )

    data_matrix = data_matrix.reshape(
        [n_samples] + [len(t) for t in grid_points] + [-1],
    )

    return FDataGrid(
        grid_points=grid_points,
        data_matrix=data_matrix,
        domain_range=domain_range,
    )


def make_gaussian_process(
    n_samples: int = 100,
    n_features: int = 100,
    *,
    start: float = 0,
    stop: float = 1,
    mean: MeanLike = 0,
    cov: CovarianceLike | None = None,
    noise: float = 0,
    random_state: RandomStateLike = None,
) -> FDataGrid:
    """Generate Gaussian process trajectories.

    Args:
        n_samples: The total number of trajectories.
        n_features: The total number of features (points of evaluation).
        start: Starting point of the trajectories.
        stop: Ending point of the trajectories.
        mean: The mean function of the process. Can be a callable accepting
              a vector with the locations, or a vector with length
              ``n_features``.
        cov: The covariance function of the process. Can be a
              callable accepting two vectors with the locations, or a
              matrix with size ``n_features`` x ``n_features``. By default,
              the Brownian covariance function is used.
        noise: Standard deviation of Gaussian noise added to the data.
        random_state: Random state.

    Returns:
        :class:`FDataGrid` object comprising all the trajectories.

    See also:
        :func:`make_gaussian`: More general function that allows to
        select the points of evaluation and to
        generate data in higer dimensions.

    """
    t = np.linspace(start, stop, n_features)

    return make_gaussian(
        n_samples=n_samples,
        grid_points=[t],
        mean=mean,
        cov=cov,
        noise=noise,
        random_state=random_state,
    )


def make_sinusoidal_process(  # noqa: WPS211
    n_samples: int = 15,
    n_features: int = 100,
    *,
    start: float = 0,
    stop: float = 1,
    period: float = 1,
    phase_mean: float = 0,
    phase_std: float = 0.6,
    amplitude_mean: float = 1,
    amplitude_std: float = 0.05,
    error_std: float = 0.2,
    random_state: RandomStateLike = None,
) -> FDataGrid:
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
    random_state = validate_random_state(random_state)

    t = np.linspace(start, stop, n_features)

    alpha = np.diag(random_state.normal(
        amplitude_mean,
        amplitude_std,
        n_samples,
    ))

    phi = np.outer(
        random_state.normal(phase_mean, phase_std, n_samples),
        np.ones(n_features),
    )

    error = random_state.normal(0, error_std, (n_samples, n_features))

    y = alpha @ np.sin((2 * np.pi / period) * t + phi) + error

    return FDataGrid(grid_points=t, data_matrix=y)


def make_multimodal_landmarks(
    n_samples: int = 15,
    *,
    n_modes: int = 1,
    dim_domain: int = 1,
    dim_codomain: int = 1,
    start: float = -1,
    stop: float = 1,
    std: float = 0.05,
    random_state: RandomStateLike = None,
) -> NDArrayFloat:
    """Generate landmarks points.

    Used by :func:`make_multimodal_samples` to generate the location of the
    landmarks.

    Generates a matrix containing the landmarks or locations of the modes
    of the samples generates by :func:`make_multimodal_samples`.

    If the same random state is used when generating the landmarks and
    multimodal samples, these will correspond to the position of the modes of
    the multimodal samples.

    Args:
        n_samples: Total number of samples.
        n_modes: Number of modes of each sample.
        dim_domain: Number of dimensions of the domain.
        dim_codomain: Number of dimensions of the codomain.
        start: Starting point of the samples. In multidimensional objects the
            starting point of the axis.
        stop: Ending point of the samples. In multidimensional objects the
            ending point of the axis.
        std: Standard deviation of the variation of the modes location.
        random_state: Random state.

    Returns:
        :class:`np.ndarray` with the location of the modes, where the component
        (i,j,k) corresponds to the mode k of the image dimension j of the
        sample i.

    """
    random_state = validate_random_state(random_state)

    modes_location = np.linspace(start, stop, n_modes + 2)[1:-1]
    modes_location = np.repeat(
        modes_location[:, np.newaxis],
        dim_domain,
        axis=1,
    )

    variation = random_state.multivariate_normal(
        (0,) * dim_domain,
        std * np.eye(dim_domain),
        size=(n_samples, dim_codomain, n_modes),
    )

    return modes_location + variation


def make_multimodal_samples(  # noqa: WPS211
    n_samples: int = 15,
    *,
    n_modes: int = 1,
    points_per_dim: int = 100,
    dim_domain: int = 1,
    dim_codomain: int = 1,
    start: float = -1,
    stop: float = 1,
    std: float = 0.05,
    mode_std: float = 0.02,
    noise: float = 0,
    modes_location: Sequence[float] | NDArrayFloat | None = None,
    random_state: RandomStateLike = None,
) -> FDataGrid:
    r"""
    Generate multimodal samples.

    Each sample :math:`x_i(t)` is proportional to a gaussian mixture, generated
    as the sum of multiple pdf of multivariate normal distributions with
    different means.

    .. math::

        x_i(t) \propto \sum_{n=1}^{\text{n\_modes}} \exp \left (
        {-\frac{1}{2\sigma} (t-\mu_n)^T \mathbb{1} (t-\mu_n)} \right )

    Where :math:`\mu_n=\text{mode\_location}_n+\epsilon` and :math:`\epsilon`
    is normally distributed, with mean :math:`\mathbb{0}` and standard
    deviation given by the parameter `std`.

    Args:
        n_samples: Total number of samples.
        n_modes: Number of modes of each sample.
        points_per_dim: Points per sample. If the object is multidimensional
            indicates the number of points for each dimension in the domain.
            The sample will have :math:
            `\text{points_per_dim}^\text{dim_domain}` points of
            discretization.
        dim_domain: Number of dimensions of the domain.
        dim_codomain: Number of dimensions of the image
        start: Starting point of the samples. In multidimensional objects the
            starting point of each axis.
        stop: Ending point of the samples. In multidimensional objects the
            ending point of each axis.
        std: Standard deviation of the variation of the modes location.
        mode_std: Standard deviation :math:`\sigma` of each mode.
        noise: Standard deviation of Gaussian noise added to the data.
        modes_location:  List of coordinates of each mode.
        random_state: Random state.

    Returns:
        :class:`FDataGrid` object comprising all the samples.

    """
    random_state = validate_random_state(random_state)

    if modes_location is None:

        location = make_multimodal_landmarks(
            n_samples=n_samples,
            n_modes=n_modes,
            dim_domain=dim_domain,
            dim_codomain=dim_codomain,
            start=start,
            stop=stop,
            std=std,
            random_state=random_state,
        )

    else:

        location = np.asarray(modes_location).reshape(
            (n_samples, dim_codomain, n_modes, dim_domain),
        )

    axis = np.linspace(start, stop, points_per_dim)

    if dim_domain == 1:
        grid_points = axis
        evaluation_grid = axis
    else:
        grid_points = np.repeat(axis[:, np.newaxis], dim_domain, axis=1).T

        meshgrid = np.meshgrid(*grid_points)

        evaluation_grid = np.empty(meshgrid[0].shape + (dim_domain,))

        for i in range(dim_domain):
            evaluation_grid[..., i] = meshgrid[i]

    # Data matrix of the grid
    shape = (n_samples,) + dim_domain * (points_per_dim,) + (dim_codomain,)
    data_matrix = np.zeros(shape)

    # Covariance matrix of the samples
    cov = mode_std * np.eye(dim_domain)

    for i, j, k in itertools.product(  # noqa: WPS440
        range(n_samples),
        range(dim_codomain),
        range(n_modes),
    ):
        data_matrix[i, ..., j] += multivariate_normal.pdf(
            evaluation_grid,
            location[i, j, k],
            cov,
        )

    # Constant to make modes value aprox. 1
    data_matrix *= (2 * np.pi * mode_std) ** (dim_domain / 2)

    data_matrix += random_state.normal(0, noise, size=data_matrix.shape)

    return FDataGrid(grid_points=grid_points, data_matrix=data_matrix)


def make_random_warping(
    n_samples: int = 15,
    n_features: int = 100,
    *,
    start: float = 0,
    stop: float = 1,
    sigma: float = 1,
    shape_parameter: float = 50,
    n_random: int = 4,
    random_state: RandomStateLike = None,
) -> FDataGrid:
    r"""Generate random warping functions.

    Let :math:`v(t)` be a randomly generated function defined in :math:`[0,1]`

    .. math::
        sv(t) = \sum_{j=0}^{N} a_j \sin(\frac{2 \pi j}{K}t) + b_j
        \cos(\frac{2 \pi j}{K}t)

    where :math:`a_j, b_j \sim N(0, \sigma)`.

    The random warping it is constructed making an exponential map to
    :math:`\Gamma`.

     .. math::
        \gamma(t) = \int_0^t \left ( \frac{\sin(\|v\|)}{\|v\|} v(s) +
        \cos(\|v\|) \right )^2 ds

    An affine traslation it is used to define the warping in :math:`[a,b]`.

    The smoothing and shape of the warpings can be controlling changing
    :math:`N`, :math:`\sigma` and :math:`K= 1 + \text{shape\_parameter}`.


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
        Object comprising all the samples.

    """
    # Based on the original implementation of J. D. Tucker in the
    # package python_fdasrsf <https://github.com/jdtuck/fdasrsf_python>.

    random_state = validate_random_state(random_state)

    freq = shape_parameter + 1

    # Frequency
    omega = 2 * np.pi / freq
    sqrt2 = np.sqrt(2)
    sqrt_sigma = np.sqrt(sigma)

    # Originally it is compute in (0,1), then it is rescaled
    time = np.outer(np.linspace(0, 1, n_features), np.ones(n_samples))

    # Operates trasposed to broadcast dimensions
    v = np.outer(
        np.ones(n_features),
        random_state.normal(scale=sqrt_sigma, size=n_samples),
    )

    for j in range(2, 2 + n_random):
        alpha = random_state.normal(scale=sqrt_sigma, size=(2, n_samples))
        alpha *= sqrt2
        v += alpha[0] * np.cos(j * omega * time)
        v += alpha[1] * np.sin(j * omega * time)

    v -= v.mean(axis=0)

    # Equivalent to RN norm (equispaced monitorization points)
    v_norm = np.linalg.norm(v, axis=0)
    v_norm /= np.sqrt(n_features)

    # Exponential map of vectors
    v *= np.sin(v_norm) / v_norm
    v += np.cos(v_norm)
    np.square(v, out=v)

    # Creation of FDataGrid in the corresponding domain
    data_matrix = scipy.integrate.cumulative_trapezoid(
        v,
        dx=1 / n_features,
        initial=0,
        axis=0,
    )
    warping = FDataGrid(data_matrix.T, grid_points=time[:, 0])
    warping = normalize_warping(warping, domain_range=(start, stop))
    warping.interpolation = SplineInterpolation(
        interpolation_order=3,
        monotone=True,
    )

    return warping
