import numpy as np
from ...representation.grid import FDataGrid
from ...datasets import fetch_phoneme
from scipy.special import jv
from ...misc.kernels import normal
from ...preprocessing.smoothing import KernelSmoother
from ...misc.hat_matrix import NadarayaWatsonHatMatrix

def get_phoneme_data(n_samples, label=1):

    phoneme, y = fetch_phoneme(return_X_y=True)

    if label < 1 or label > 5:
        raise ValueError("Label must be between 1 and 5.")

    fd_train = phoneme[y == label]

    smoother = KernelSmoother(
        NadarayaWatsonHatMatrix(
            bandwidth=0.15,
            kernel=normal,
        ),
    )
    fd_train = smoother.fit_transform(fd_train)
    if len(fd_train) < n_samples:
        raise Exception("Not enough samples in the data base")
    return fd_train[:n_samples]


def generate_sin_function(
    n_samples=1000,
    n_points=100,
    domain_range=(0, 1),
    frequency_range=(3.5 * np.pi, 4.5 * np.pi),
    phase_range=(0, 0.3 * np.pi),
    amplitude_range=(0.5, 2.0),
    noise=False,
):
    """
    Generate synthetic functional data with varying parameters.
    The parameters are uniformly sampled within specified ranges.

    Args:
        n_samples: Number of functional samples to generate
        n_points: Number of discretization points
        domain_range: Tuple (start, end) for the domain
        frequency_range: Tuple (min, max) for frequency
        phase_range: Tuple (min, max) for phase shift
        amplitude_range: Tuple (min, max) for amplitude
        noise: Boolean indicating whether to add Gaussian noise

    Returns:
        FDataGrid object containing the synthetic functions
    """
    t = np.linspace(domain_range[0], domain_range[1], n_points)
    data_matrix = np.zeros((n_samples, n_points, 1))
    epsilon = 1e-6
    if isinstance(frequency_range, (int, float)):
        frequency_range = (frequency_range - epsilon, frequency_range + epsilon)
    if isinstance(phase_range, (int, float)):
        phase_range = (phase_range - epsilon, phase_range + epsilon)
    if isinstance(amplitude_range, (int, float)):
        amplitude_range = (amplitude_range - epsilon,
                           amplitude_range + epsilon)
    print("Generating sin functions with parameters:")
    print(frequency_range, phase_range, amplitude_range)
    for i in range(n_samples):
        # Random amplitude, frequency, and phase
        A = np.random.uniform(amplitude_range[0], amplitude_range[1])
        frequency = np.random.uniform(frequency_range[0], frequency_range[1])
        phi = np.random.uniform(phase_range[0], phase_range[1])

        # Generate function with small noise
        x_t = A * np.sin(frequency * t + phi)
        if noise:
            x_t += np.random.normal(0, 0.05, n_points)
        data_matrix[i, :, 0] = x_t

    return FDataGrid(data_matrix=data_matrix, grid_points=t)


def generate_step_function(
    n_samples=1000,
    n_points=128,
    steps_in=[20, 40, 60, 80],
    steps_values=[0.3, (0.3, 0.7), 0.2, (0.4, 0.8), 0.5],
):
    if len(steps_in) + 1 != len(steps_values):
        raise ValueError(
            "Length of steps_values must be one more than length of steps"
        )
    steps = steps_in.copy()
    steps.insert(0, 0)
    steps.append(n_points)
    data = np.zeros((n_samples, n_points))

    for i in range(1, len(steps)):
        if isinstance(steps_values[i - 1], tuple | list):
            data[:, steps[i - 1]: steps[i]] = np.random.uniform(
                steps_values[i - 1][0],
                steps_values[i - 1][1],
                size=(n_samples, 1),
            )
        else:
            data[:, steps[i - 1]: steps[i]] = steps_values[i - 1]

    return FDataGrid(data)


def generate_bessel_function(
    n_samples=1000,
    n_points=100,
    domain_range=(0, 20),
    order_range=(0, 5),
    omega_range=(0.8, 1.2),
    phase_range=(0, 2 * np.pi),
    amplitude_range=(0.5, 2.0),
    noise=False,
):
    """
    Generate synthetic functional data based on Bessel functions of the first kind.
    
    The function generates curves of the form: f(t) = A * J_alpha(omega * t) + noise
    where alpha is the order of the Bessel function.

    Args:
        n_samples: Number of functional samples to generate
        n_points: Number of discretization points
        domain_range: Tuple (start, end) for the domain. 
                      (0, 20) is recommended to visualize the decay.
        order_range: Tuple (min, max) for the order (alpha) of the Bessel function.
        omega_range: Tuple (min, max) for the frequency/scaling factor.
        phase_range: Tuple (min, max) for the horizontal phase shift.
        amplitude_range: Tuple (min, max) for amplitude.
        noise: Boolean indicating whether to add Gaussian noise.

    Returns:
        FDataGrid object containing the synthetic functions
    """
    # Create the grid points
    t = np.linspace(domain_range[0], domain_range[1], n_points)
    epsilon = 1e-8
    # Initialize the data matrix (samples, points, dimensions)
    data_matrix = np.zeros((n_samples, n_points, 1))
    if isinstance(order_range, (int, float)):
        order_range = (order_range - epsilon, order_range + epsilon)
    if isinstance(omega_range, (int, float)):
        omega_range = (omega_range - epsilon,
                       omega_range + epsilon)
    if isinstance(phase_range, (int, float)):
        phase_range = (phase_range - epsilon,
                       phase_range + epsilon)
    if isinstance(amplitude_range, (int, float)):
        amplitude_range = (amplitude_range - epsilon,
                           amplitude_range + epsilon)

    for i in range(n_samples):
        # Randomize parameters uniformly
        A = np.random.uniform(amplitude_range[0], amplitude_range[1])
        omega = np.random.uniform(omega_range[0], omega_range[1])
        alpha = np.random.uniform(order_range[0], order_range[1])
        phi = np.random.uniform(phase_range[0], phase_range[1])

        # Calculate Bessel function J_alpha(omega * t + phi)
        # We use standard absolute value on t to avoid complex numbers if domain < 0 
        # (though usually domain is > 0 for standard Bessel time series)
        val = jv(alpha, np.abs(omega * t + phi))
        x_t = A * val

        # Generate Gaussian noise
        if noise:
            x_t += np.random.normal(0, 0.05, n_points)
        data_matrix[i, :, 0] = x_t

    return FDataGrid(data_matrix=data_matrix, grid_points=t)


def generate_lines_function(n_samples=1000, n_points=100, domain_range=(0, 1), slope_range=(-2.0, 2.0), intercept_range=(-1.0, 1.0), noise=False):
    """
    Generate synthetic functional data consisting of random lines.

    Each function is a straight line defined by a random slope and intercept,
    with added Gaussian noise if noise is True.

    Args:
        n_samples: Number of functional samples to generate
        n_points: Number of discretization points
        domain_range: Tuple (start, end) for the domain
        slope_range: Tuple (min, max) for the slope of the lines
        intercept_range: Tuple (min, max) for the intercept of the lines
        noise: Boolean indicating whether to add Gaussian noise

    Returns:
        FDataGrid object containing the synthetic line functions
    """
    t = np.linspace(domain_range[0], domain_range[1], n_points)
    data_matrix = np.zeros((n_samples, n_points, 1))
    epsilon = 1e-8
    # If slope range is a number, convert to tuple
    if isinstance(slope_range, (int, float)):
        slope_range = (slope_range - epsilon, slope_range + epsilon)
    if isinstance(intercept_range, (int, float)):
        intercept_range = (intercept_range - epsilon,
                           intercept_range + epsilon)

    for i in range(n_samples):
        slope = np.random.uniform(slope_range[0], slope_range[1])
        intercept = np.random.uniform(intercept_range[0], intercept_range[1])
        x_t = slope * t + intercept
        if noise:
            x_t += np.random.normal(0, 0.5, n_points)
        data_matrix[i, :, 0] = x_t

    return FDataGrid(data_matrix=data_matrix, grid_points=t)


def generate_data(n_samples, n_points, dataset, **kwargs):

    if dataset == 'sin':
        fd_data = generate_sin_function(n_samples, n_points, **kwargs)
    elif dataset == 'bessel':
        fd_data = generate_bessel_function(n_samples, n_points, **kwargs)
    elif dataset == 'step':
        fd_data = generate_step_function(n_samples, n_points, **kwargs)
    elif dataset == 'lines':
        fd_data = generate_lines_function(n_samples, n_points, **kwargs)
    elif dataset == 'phoneme':
        fd_data = get_phoneme_data(n_samples, **kwargs)
    else:
        raise ValueError("Unknown dataset: {}".format(dataset))

    return fd_data[0].grid_points[0], fd_data