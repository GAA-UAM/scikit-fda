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

    Uses a two-stage approach:
      1. Hann-windowed FFT (zero-padded) for a coarse frequency estimate.
      2. Fine grid search around the coarse estimate, fitting
         y = a*sin(ω*t) + b*cos(ω*t) via linear regression at each
         candidate ω and picking the one with lowest MSE.
    Amplitude and phase are then derived from the optimal regression
    coefficients: A = sqrt(a² + b²), φ = atan2(b, a).

    Args:
        fdata: FDataGrid object containing the sinus functional data.
    Returns:
        ndarray of shape (n_samples, 3) with columns
        [amplitude, angular_frequency (rad/s), phase (in [0, 2π])].
    """
    Y = fdata.data_matrix[..., 0]          # (n_samples, n_points)
    x = fdata.grid_points[0]               # (n_points,)
    n_samples, n_points = Y.shape
    dt = x[1] - x[0]

    # ---- Stage 1: coarse frequency via Hann-windowed FFT ----
    Y_centered = Y - np.mean(Y, axis=1, keepdims=True)
    window = np.hanning(n_points)           # reduces spectral leakage
    Y_windowed = Y_centered * window[np.newaxis, :]

    n_padded = 100 * n_points               # heavy zero-padding for smooth spectrum
    fft_coeffs = np.fft.fft(Y_windowed, n=n_padded, axis=1)

    frequencies = np.fft.fftfreq(n_padded, d=dt)
    pos_mask = frequencies > 0
    pos_freqs = frequencies[pos_mask]
    pos_mags = np.abs(fft_coeffs[:, pos_mask])

    peak_indices = np.argmax(pos_mags, axis=1)             # (n_samples,)
    w_coarse = pos_freqs[peak_indices] * 2 * np.pi         # rad/s per sample

    # ---- Stage 2: fine grid search + linear regression ----
    df_rad = (pos_freqs[1] - pos_freqs[0]) * 2 * np.pi    # one FFT bin in rad/s
    n_fine = 201                                            # search points
    offsets = np.linspace(-df_rad, df_rad, n_fine)          # symmetric around coarse

    best_mse = np.full(n_samples, np.inf)
    best_w = w_coarse.copy()
    best_a = np.zeros(n_samples)
    best_b = np.zeros(n_samples)

    for offset in offsets:
        w_test = w_coarse + offset                          # (n_samples,)
        S = np.sin(w_test[:, np.newaxis] * x[np.newaxis, :])  # (n_samples, n_points)
        C = np.cos(w_test[:, np.newaxis] * x[np.newaxis, :])

        # Vectorized normal equations for y = a*S + b*C
        SS = np.sum(S * S, axis=1)
        SC = np.sum(S * C, axis=1)
        CC = np.sum(C * C, axis=1)
        Sy = np.sum(S * Y, axis=1)
        Cy = np.sum(C * Y, axis=1)

        det = SS * CC - SC * SC
        safe_det = np.where(np.abs(det) < 1e-12, 1.0, det)
        a = (CC * Sy - SC * Cy) / safe_det
        b = (SS * Cy - SC * Sy) / safe_det

        residuals = Y - a[:, np.newaxis] * S - b[:, np.newaxis] * C
        mse = np.mean(residuals ** 2, axis=1)

        improved = mse < best_mse
        best_mse = np.where(improved, mse, best_mse)
        best_w = np.where(improved, w_test, best_w)
        best_a = np.where(improved, a, best_a)
        best_b = np.where(improved, b, best_b)

    # ---- Derive amplitude and phase from regression coefficients ----
    amplitudes = np.sqrt(best_a ** 2 + best_b ** 2)
    phases = np.arctan2(best_b, best_a) % (2 * np.pi)
    # Snap phases very close to 2π back to 0 (numerical noise in atan2)
    phases = np.where(2 * np.pi - phases < 1e-3, 0.0, phases)

    characteristics = np.column_stack((amplitudes, best_w, phases))
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

def get_noise_metric(fdata: skfda.FDataGrid, dataset : str, kwargs, characteristics: np.ndarray = None) -> np.ndarray:
    """
    Get the noise metric for the functional data.
    The noise metric is defined as the mean squared error between the original functions
    and the reconstructed functions using the characteristics.

    Args:
        fdata: FDataGrid object containing the functional data.
        dataset: Type of dataset. Options are 'sin', 'constant', 'lines', 'bessel', 'step'.
        kwargs: Additional arguments for specific datasets.
        characteristics: Optional pre-computed characteristics. If None, they will be extracted.
            Shape (n_samples, c).
    Returns:
        ndarray object containing the noise metric. Shape (n_samples,).
    """
    if characteristics is None:
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
    noise_metric = get_noise_metric(fdata, dataset_name, kwargs, characteristics=characteristics)

    metrics = {
        'wasserstein_distance': w_dist,
        'noise_metric': noise_metric,
        'characteristics': characteristics,
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

    # Reconstruct functions: frequencies are already in rad/s (ω), no extra 2π
    Y_reconstructed = amplitudes[:, np.newaxis] * np.sin(frequencies[:, np.newaxis] * x + phases[:, np.newaxis])
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
    from .synthetic_data import generate_data
    import matplotlib.pyplot as plt

    # ==================== TEST: Sin characteristic extraction ====================
    print("=" * 60)
    print("TEST: extract_sin_characteristics + get_sin_noise_metric")
    print("=" * 60)

    # 1. Generate sine data with known parameters (no noise)
    N = 500
    n_points = 100
    A_range = (0.5, 2.0)
    w_range = (3.5 * np.pi, 4.5 * np.pi)  # angular frequency (rad/s)
    phi_range = (0, 0.3 * np.pi)

    params, fd_sin = generate_data(
        N, n_points, 'sin',
        amplitude_range=A_range,
        frequency_range=w_range,
        phase_range=phi_range,
        noise=False,
    )

    # 2. Extract characteristics
    chars = extract_sin_characteristics(fd_sin)
    amplitudes_ext = chars[:, 0]
    freqs_ext = chars[:, 1]
    phases_ext = chars[:, 2]
    t = params  # grid points

    print(f"\nExtracted ranges:")
    print(f"  Amplitude: [{amplitudes_ext.min():.4f}, {amplitudes_ext.max():.4f}]  "
          f"(expected [{A_range[0]}, {A_range[1]}])")
    print(f"  Frequency: [{freqs_ext.min():.4f}, {freqs_ext.max():.4f}]  "
          f"(expected [{w_range[0]:.4f}, {w_range[1]:.4f}])")
    print(f"  Phase:     [{phases_ext.min():.4f}, {phases_ext.max():.4f}]  "
          f"(expected [{phi_range[0]:.4f}, {phi_range[1]:.4f}])")

    # 3. Noise metric should be near zero for clean data
    mse = get_sin_noise_metric(fd_sin, chars)
    print(f"\nNoise MSE (no noise, should be ~0):")
    print(f"  Mean: {mse.mean():.8f},  Max: {mse.max():.8f}")

    # 4. Test with a single known function: A=1.5, w=4*pi, phi=0.2
    A_true, w_true, phi_true = 1.5, 4 * np.pi, 0.2
    y_single = A_true * np.sin(w_true * t + phi_true)
    fd_single = skfda.FDataGrid(
        data_matrix=y_single.reshape(1, -1, 1),
        grid_points=t,
    )
    chars_single = extract_sin_characteristics(fd_single)
    A_ext, w_ext, phi_ext = chars_single[0]
    mse_single = get_sin_noise_metric(fd_single, chars_single)[0]

    print(f"\nSingle known function test:")
    print(f"  Amplitude: {A_ext:.6f}  (true: {A_true})")
    print(f"  Frequency: {w_ext:.6f}  (true: {w_true:.6f})")
    print(f"  Phase:     {phi_ext:.6f}  (true: {phi_true:.6f})")
    print(f"  MSE:       {mse_single:.10f}")

    # 5. Assertions
    tol_A = 0.01
    tol_w = 0.01
    tol_phi = 0.01
    tol_mse_single = 1e-6
    tol_mse_batch = 1e-3  # batch includes harder edge-case frequencies

    assert abs(A_ext - A_true) < tol_A, \
        f"Amplitude error: {abs(A_ext - A_true):.6f} > {tol_A}"
    assert abs(w_ext - w_true) < tol_w, \
        f"Frequency error: {abs(w_ext - w_true):.6f} > {tol_w}"
    # Phase comparison must handle circular wraparound (0 ≈ 2π)
    phi_err = min(abs(phi_ext - phi_true), 2 * np.pi - abs(phi_ext - phi_true))
    assert phi_err < tol_phi, \
        f"Phase error: {phi_err:.6f} > {tol_phi}"
    assert mse_single < tol_mse_single, \
        f"Single MSE too large: {mse_single:.10f} > {tol_mse_single}"
    assert mse.mean() < tol_mse_batch, \
        f"Batch mean MSE too large: {mse.mean():.10f} > {tol_mse_batch}"

    print("\n✓ All assertions passed!")

    # 6. Visual check: overlay original vs reconstructed for a few samples
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    for idx, ax in enumerate(axes):
        y_orig = fd_sin.data_matrix[idx, :, 0]
        A_i, w_i, phi_i = chars[idx]
        y_recon = A_i * np.sin(w_i * t + phi_i)
        ax.plot(t, y_orig, label='Original', linewidth=2)
        ax.plot(t, y_recon, '--', label='Reconstructed', linewidth=2)
        ax.set_title(f"Sample {idx}: A={A_i:.2f}, ω={w_i:.2f}, φ={phi_i:.2f}")
        ax.legend()
        ax.grid(alpha=0.3)
    plt.suptitle("Sin Characteristic Extraction Test", fontsize=14)
    plt.tight_layout()
    plt.show()