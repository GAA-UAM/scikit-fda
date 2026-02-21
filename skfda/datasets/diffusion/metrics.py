import numpy as np
import skfda
import scipy.stats as stats

'''
This methods provides metrics to evaluate the performance of generative models for functional data.
The main metrics implemented are:
- Wasserstein metric to a uniform distribution
- Noise metric
'''


def wasserstein_to_uniform(fdata: np.ndarray, a: np.ndarray, b: np.ndarray) -> float:
    """
    Compute the Wasserstein distance between the distribution of the functional data
    and a uniform distribution between a and b shape (c,). The fdata array is no the values of the functions
    is an array shape (n_samples, c) where c is the number of characteristics of the functions.
    For example, if the functions are sinus fucntions, c=3 (amplitude, frequency, phase).
    For constant functions, c=1 (value of the constant function), if lines are used, c=2 (slope and intercept).

    Args:
        fdata: ndarray object containing the data of the functions.
        a: Lower bound of the uniform distribution, shape (c,).
        b: Upper bound of the uniform distribution, shape (c,).
    Returns:
        Wasserstein distance between the distribution of the functional data and the uniform distribution.
        shape (c,).
    """
    N = fdata.shape[0]
    # Need to sort the data for each characteristic
    fdata = np.sort(fdata, axis=0)
    time_factor = ((np.arange(N) + 0.5) / N)[:, np.newaxis]
    # Matix of differences
    # X_n - a + (b - a) * (n + 0.5)/N For each characteristic
    diff_matrix = fdata - (a  + (b - a) * time_factor)
    w_dist = np.mean(np.abs(diff_matrix), axis=0)

    return w_dist

def get_wasserstein_distance(fdata: skfda.FDataGrid, dataset : str, a:np.ndarray, b:np.ndarray, kwargs) -> np.ndarray:
    """
    Get the Wasserstein distance between the distribution of the functional data
    and a uniform distribution between a and b shape (c,). The fdata array is no the values of the functions
    is an array shape (n_samples, c) where c is the number of characteristics of the functions.
    For example, if the functions are sinus fucntions, c=3 (amplitude, frequency, phase).
    For constant functions, c=1 (value of the constant function), if lines are used, c=2 (slope and intercept).

    Args:
        fdata: FDataGrid object containing the functional data.
        dataset: Type of dataset. Options are 'sin', 'constant', 'lines', 'bessel', 'step'.
        a: Lower bound of the uniform distribution, shape (c,).
        b: Upper bound of the uniform distribution, shape (c,).
        kwargs: Additional arguments for specific datasets.
    Returns:
        ndarray object containing the Wasserstein distance between the distribution of the functional data
        and the uniform distribution. shape (c,).
    """
    characteristics = get_characteristics(fdata, dataset, kwargs)
    w_dist = wasserstein_to_uniform(characteristics, a, b)
    return w_dist

def get_characteristics(fdata: skfda.FDataGrid, dataset: str, kwargs) -> np.ndarray:
    """
    Get the characteristics of the functional data.
    For sinus functions, the characteristics are amplitude, frequency and phase.
    For lines, the characteristics are slope and intercept.
    For bessel functions, the characteristics are amplitude, frequency, order and phase.
    For step the values are the levels of the steps.

    Args:
        fdata: FDataGrid object containing the functional data.
        dataset: Type of dataset. Options are 'sin', 'constant', 'lines', 'bessel', 'step'.
        kwargs: Additional arguments for specific datasets.
    Returns:
        ndarray object containing the characteristics of the functional data. Shape (n_samples, c) where c is the number of characteristics.
    """

    if dataset == 'sin':
        return extract_sin_characteristics(fdata, **kwargs)
    elif dataset == 'lines':
        return extract_lines_characteristics(fdata, **kwargs)
    elif dataset == 'bessel':
        return extract_bessel_characteristics(fdata, **kwargs)
    elif dataset == 'step':
        return extract_step_characteristics(fdata, **kwargs)
    else:
        raise ValueError(f"Dataset {dataset} not supported for characteristics extraction.")

def extract_sin_characteristics(fdata: skfda.FDataGrid) -> np.ndarray:
    """
    Extract amplitude, frequency and phase from sinus functional data.

    Args:
        fdata: FDataGrid object containing the sinus functional data.
    Returns:
        ndarray object containing amplitude, frequency and phase. Shape (n_samples, 3).
    """
    Y = fdata.data_matrix[..., 0]
    x = fdata.grid_points[0]
    n_samples, n_points = Y.shape
    # 2. Vectorized Amplitude
    # Calculate max and min across the grid_points axis (axis 1)
    amplitudes = (np.max(Y, axis=1) - np.min(Y, axis=1)) / 2
    # shape of amplitudes is (n_samples,)

    # 3. Vectorized FFT
    # Subtract the mean of each sample (broadcasting)
    Y_centered = Y - np.mean(Y, axis=1, keepdims=True)
    n_padded = 100 * n_points  # Padding to increase frequency resolution
    # Compute FFT on the whole matrix at once along axis 1
    fft_coeffs = np.fft.fft(Y_centered, n=n_padded, axis=1)

    # Compute frequencies once (they are the same for all samples)
    frequencies = np.fft.fftfreq(n_padded, d=(x[1] - x[0]))

    # Create a mask for positive frequencies (indices 1 to n/2)
    # We only care about the positive half for finding the dominant frequency
    pos_mask = frequencies > 0
    pos_freqs = frequencies[pos_mask]

    # Slice the FFT coefficients to keep only positive frequency columns
    # Shape becomes (n_samples, n_positive_freqs)
    pos_coeffs = fft_coeffs[:, pos_mask]

    # 4. Find Dominant Frequency & Phase
    # Find the index of the max magnitude for each sample
    # peak_indices shape is (n_samples,)
    peak_indices = np.argmax(np.abs(pos_coeffs), axis=1)

    # Map indices to actual frequency values
    dominant_freqs = pos_freqs[peak_indices]
    # Convert to radians from 2pi*f -> w 
    dominant_freqs_rads = dominant_freqs * 2 * np.pi
    # Extract the complex value at the peak index to get the phase
    # We use advanced indexing: [range(N), peak_indices]
    peak_complex_vals = pos_coeffs[np.arange(n_samples), peak_indices]
    # Phases of cosines so we need to add pi/2
    phases = np.angle(peak_complex_vals) + np.pi/2
    # We want phases in [0, 2pi]
    phases = np.mod(phases, 2 * np.pi)

    # 5. Stack results
    # Stack the three arrays into shape (n_samples, 3)
    characteristics = np.column_stack((amplitudes, dominant_freqs_rads, phases))
    return characteristics

def extract_lines_characteristics(fdata: skfda.FDataGrid, has_slope: bool) -> np.ndarray:
    """
    Extract slope and intercept from lines functional data.

    Args:
        fdata: FDataGrid object containing the lines functional data.
        has_slope: Boolean indicating whether to extract slope.
    Returns:
        ndarray object containing slope and intercept. Shape (n_samples, 2) if has_slope is True, else (n_samples, 1).
    """
    Y = fdata.data_matrix[..., 0]
    x = fdata.grid_points[0]
    n_samples, n_points = Y.shape

    if has_slope:
        # Vectorized calculation of slope and intercept
        x_mean = np.mean(x)
        Y_mean = np.mean(Y, axis=1)

        numerator = np.sum((x - x_mean) * (Y - Y_mean[:, np.newaxis]), axis=1)
        denominator = np.sum((x - x_mean) ** 2)

        slopes = numerator / denominator
        intercepts = Y_mean - slopes * x_mean

        characteristics = np.column_stack((intercepts, slopes))
    else:
        # Only extract intercept
        intercepts = np.mean(Y, axis=1)
        characteristics = intercepts[:, np.newaxis]

    return characteristics

def extract_bessel_characteristics(fdata: skfda.FDataGrid, first_guess : np.ndarray) -> np.ndarray:
    """
    Extract amplitude, frequency, order and phase from bessel functional data.

    Args:
        fdata: FDataGrid object containing the bessel functional data.
        first_guess: Initial guess for the parameters for curve fitting.
    Returns:
        ndarray object containing amplitude, frequency, order and phase. Shape (n_samples, 4).
    """ 
    pass


def extract_step_characteristics(fdata: skfda.FDataGrid, steps_t: np.ndarray) -> np.ndarray:
    """
    Extract step levels from step functional data.

    Args:
        fdata: FDataGrid object containing the step functional data.
        steps_t: Array containing the time points of the steps in the step functions.
    Returns:
        ndarray object containing the step levels. Shape (n_samples, n_steps).
    """
    Y = fdata.data_matrix[..., 0]
    x = fdata.grid_points[0]
    n_samples, n_points = Y.shape
    n_steps = len(steps_t)

    step_levels = np.zeros((n_samples, n_steps))

    for i in range(n_steps):
        if i == 0:
            mask = x < steps_t[i]
        elif i == n_steps - 1:
            mask = x >= steps_t[i - 1]
        else:
            mask = (x >= steps_t[i - 1]) & (x < steps_t[i])

        # Calculate mean level for each sample in the current step interval
        step_levels[:, i] = np.mean(Y[:, mask], axis=1)

    return step_levels

def get_noise_metric(fdata: skfda.FDataGrid, dataset : str, kwargs) -> np.ndarray:
    """
    Get the noise metric for the functional data.
    The noise metric is defined as the mean squared error between the original functions
    and the reconstructed functions using the characteristics.

    Args:
        fdata: FDataGrid object containing the functional data.
        dataset: Type of dataset. Options are 'sin', 'constant', 'lines', 'bessel', 'step'.
        characteristics: ndarray object containing the characteristics of the functional data. Shape (n_samples, c).
        kwargs: Additional arguments for specific datasets.
    Returns:
        ndarray object containing the noise metric. Shape (n_samples,).
    """
    characteristics = get_characteristics(fdata, dataset, kwargs)
    if dataset == 'sin':
        return get_sin_noise_metric(fdata, characteristics)
    elif dataset == 'lines':
        return get_lines_noise_metric(fdata, characteristics)
    elif dataset == 'bessel':
        return get_bessel_noise_metric(fdata, characteristics)
    elif dataset == 'step':
        steps_t = kwargs.get('steps_t', None)
        if steps_t is None:
            raise ValueError("steps_t must be provided in kwargs for step dataset.")
        return get_step_noise_metric(fdata, characteristics, steps_t)
    else:
        raise ValueError(f"Dataset {dataset} not supported for noise metric calculation.")
    

def get_metrics(fdata: skfda.FDataGrid, dataset_name: str, a: np.ndarray, b: np.ndarray, kwargs) -> dict:
    """
    Get the metrics for the functional data.
    The metrics are Wasserstein distance to uniform distribution and noise metric.

    Args:
        fdata: FDataGrid object containing the functional data.
        dataset: Type of dataset. Options are 'sin', 'constant', 'lines', 'bessel', 'step'.
        a: Lower bound of the uniform distribution, shape (c,).
        b: Upper bound of the uniform distribution, shape (c,).
        kwargs: Additional arguments for specific datasets.
    Returns:
        dict object containing the metrics.
    """
    characteristics = get_characteristics(fdata, dataset_name, kwargs)
    w_dist = wasserstein_to_uniform(characteristics, a, b)
    noise_metric = get_noise_metric(fdata, dataset_name, characteristics, kwargs)

    metrics = {
        'wasserstein_distance': w_dist,
        'noise_metric': noise_metric
    }

    return metrics



def get_sin_noise_metric(fdata: skfda.FDataGrid, characteristics: np.ndarray) -> float:
    """
    Compute the noise metric for sinus functional data.
    The mean squared error between the original functions and the reconstructed functions
    using the characteristics (amplitude, frequency, phase).
    Args:
        fdata: FDataGrid object containing the sinus functional data.
        characteristics: ndarray object containing amplitude, frequency and phase. Shape (n_samples, 3).
    Returns:
        np.ndarray object containing the noise metric. Shape (n_samples,).
    """
    Y = fdata.data_matrix[..., 0]
    x = fdata.grid_points[0]
    n_samples, n_points = Y.shape

    amplitudes = characteristics[:, 0]
    frequencies = characteristics[:, 1]
    phases = characteristics[:, 2]

    # Reconstruct functions
    Y_reconstructed = np.zeros_like(Y)
    Y_reconstructed = amplitudes[:, np.newaxis] * np.sin(2 * np.pi * frequencies[:, np.newaxis] * x + phases[:, np.newaxis])
    # Compute mean squared error
    mse = np.mean((Y - Y_reconstructed) ** 2, axis=1)

    return mse

def get_lines_noise_metric(fdata: skfda.FDataGrid, characteristics: np.ndarray) -> float:
    """
    Compute the noise metric for lines functional data.
    The mean squared error between the original functions and the reconstructed functions
    using the characteristics (slope and intercept).
    Args:
        fdata: FDataGrid object containing the lines functional data.
        characteristics: ndarray object containing slope and intercept. Shape (n_samples, 2) or (n_samples, 1).
    Returns:
        np.ndarray object containing the noise metric. Shape (n_samples,).
    """
    Y = fdata.data_matrix[..., 0]
    x = fdata.grid_points[0]

    if characteristics.shape[1] == 2:
        slopes = characteristics[:, 1]
        intercepts = characteristics[:, 0]
        Y_reconstructed = slopes[:, np.newaxis] * x + intercepts[:, np.newaxis]
    else:
        intercepts = characteristics[:, 0]
        Y_reconstructed = intercepts[:, np.newaxis] * np.ones_like(x)

    # Compute mean squared error
    mse = np.mean((Y - Y_reconstructed) ** 2, axis=1)

    return mse

def get_bessel_noise_metric(fdata: skfda.FDataGrid, characteristics: np.ndarray) -> float:
    """
    Compute the noise metric for bessel functional data.[ 1.49981119  1.98       -1.50796447]
    The mean squared error between the original functions and the reconstructed functions
    using the characteristics (amplitude, frequency, order and phase).
    Args:
        fdata: FDataGrid object containing the bessel functional data.
        characteristics: ndarray object containing amplitude, frequency, order and phase. Shape (n_samples, 4).
    Returns:
        np.ndarray object containing the noise metric. Shape (n_samples,).
    """
    pass

def get_step_noise_metric(fdata: skfda.FDataGrid, characteristics: np.ndarray, steps_t: np.ndarray) -> float:
    """
    Compute the noise metric for step functional data.
    The mean squared error between the original functions and the reconstructed functions
    using the characteristics (step levels).
    Args:
        fdata: FDataGrid object containing the step functional data.
        characteristics: ndarray object containing the step levels. Shape (n_samples, n_steps).
        steps_t: Array containing the time points of the steps in the step functions.
    Returns:
        np.ndarray object containing the noise metric. Shape (n_samples,).
    """
    Y = fdata.data_matrix[..., 0]
    x = fdata.grid_points[0]
    n_samples, n_points = Y.shape
    n_steps = characteristics.shape[1]

    # Reconstruct functions
    Y_reconstructed = np.zeros_like(Y)

    for i in range(n_steps):
        if i == 0:
            mask = x < steps_t[i]
        elif i == n_steps - 1:
            mask = x >= steps_t[i - 1]
        else:
            mask = (x >= steps_t[i - 1]) & (x < steps_t[i])

        Y_reconstructed[:, mask] = characteristics[:, i][:, np.newaxis]

    # Compute mean squared error
    mse = np.mean((Y - Y_reconstructed) ** 2, axis=1)

    return mse
if __name__ == "__main__":
    from synthetic_data import generate_data
    import matplotlib.pyplot as plt

    N = 1000
    n_points = 100
    dataset_name = 'lines'
    num_batches_to_gen = 100
    _, fd_1 = generate_data(N* num_batches_to_gen, n_points, dataset_name, intercept_range = (1-1e-6,1+ 1e-6), slope_range = (-1e-6, 1e-6), noise = True)
    _, fd_2 = generate_data(N* num_batches_to_gen, n_points, dataset_name, intercept_range = (1-1e-6, 1+ 1e-6), slope_range = (-1e-6, 1e-6))
   
    a_gen = np.array([1-1e-6, -1e-6])  # Lower bound for intercept
    b_gen = np.array([1+1e-6, 1e-6])  # Upper bound for intercept
    wessier_results_1 = np.zeros((num_batches_to_gen,len(a_gen)))
    wessier_results_2 = np.zeros((num_batches_to_gen,len(a_gen)))
    for n in range(num_batches_to_gen):
        if n == 0:
            fd_1_batch = fd_1[0:N]
            fd_2_batch = fd_2[0:N]

        else:
            fd_1_batch = fd_1[n*N:(n+1)*N]
            fd_2_batch = fd_2[n*N:(n+1)*N]

        batch_wess_1 = get_wasserstein_distance(fd_1_batch, dataset_name, a_gen, b_gen, {'has_slope': True})
        batch_wess_2 = get_wasserstein_distance(fd_2_batch, dataset_name, a_gen, b_gen, {'has_slope': True})
        wessier_results_1[n] = batch_wess_1
        wessier_results_2[n] = batch_wess_2
    
    # For each character, plot an histogram with different color for each model
    bins = np.linspace(min(wessier_results_1.min(), wessier_results_2.min()), max(wessier_results_1.max(), wessier_results_2.max()), min(50, 5*num_batches_to_gen)).reshape(-1) 
    fig, axis = plt.subplots(2, len(a_gen), figsize=(8, 6))
    fd_1[:100].plot(axes=axis[0,0], color='blue', alpha=0.3)
    axis[0,0].set_title('Generated Data with Noise')
    fd_2[:100].plot(axes=axis[0,1], color='orange', alpha=0.3)
    axis[0,1].set_title('Generated Data without Noise')
    for i, axs in enumerate(axis[1, :]):
        axs.hist(wessier_results_1[:, i],
                bins=bins,
                alpha=0.5,
                label='1',
                color='blue',
                density=False)
        axs.hist(wessier_results_2[:, i],
                bins=bins,
                alpha=0.5,
                label='2',
                color='orange',
                density=False)
        axs.set_title('Wasserstein Distance to Uniform Distribution')
        axs.set_xlabel('Wasserstein Distance')
        axs.set_ylabel('Frequency')
        axs.legend()
    plt.tight_layout()
    plt.show()