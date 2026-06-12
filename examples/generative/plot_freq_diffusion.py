"""
Frequency control diffusion processes.
=======================================================================

This example shows how to modify the decay rate of the different
frequencies of a signal (1D function).
"""

# Author: Diego Rodríguez Ortiz
# License: MIT
# sphinx_gallery_thumbnail_number = 1

# %%
# Frequency-Domain Diffusion with Circulant Matrices
# --------------------------------------------------
#
#
# This example explores the `CirculantSymmetricMatrixDiffusionProcess` class
# in depth and demonstrates how to design diffusion processes directly in the
# frequency domain. We recommend reading
# :ref:`sphx_glr_auto_examples_generative_plot_vp_vs_ve.py` first to
# understand the core concepts of diffusion models. We also recommend reading
# :ref:`sphx_glr_auto_examples_generative_plot_vp_vs_freq.py` to see how the
# same class can impose spatial structure on the forward process — in that
# example the process was defined via the first row of a circulant symmetric
# matrix, whereas here we work directly with the eigenvalues of that matrix.
#
# The Cosine Eigenbasis
# ^^^^^^^^^^^^^^^^^^^^^
#
# A circulant symmetric matrix can be diagonalized by the Fourier matrix, and
# its eigenvalues equal the Fourier transform of its first row. Since we prefer
# real-valued representations, we diagonalize using the discrete cosine basis,
# which shares the same eigenvalues as the Fourier basis for symmetric
# matrices.
#
# The eigenvector matrix **Q** has the following form:
#
# .. math::
#
#    \mathbf{Q}_{m,k} =
#    \begin{cases}
#    \frac{1}{\sqrt{M}} & \text{if } k = 0 \\
#    \sqrt{\frac{2}{M}} \cos\left(\frac{\pi \cdot m \cdot k}{M}\right)
#    & \text{if } k \in \{1, 2, \dots, M-1\}
#    \end{cases}
#
#    \quad \text{for } m \in \{0, 1, \dots, M-1\}
#
# The `CirculantSymmetricMatrixDiffusionProcess` class lets us define a
# diffusion process by specifying the eigenvalues of the circulant symmetric
# matrices that control the drift and diffusion terms. We pass a function that
# receives the diffusion time step and returns half the eigenvalues (the other
# half is fixed by symmetry) for both the drift and diffusion matrices.
#
# Experiment Design
# ^^^^^^^^^^^^^^^^^
#
# In this example, we compare two frequency-selective schedules:
#
# - **High-frequency-first**: higher frequencies decay during
#   :math:`t \in [0, 0.5]`; lower frequencies decay during
#   :math:`t \in [0.5, 1]`.
#
# - **Low-frequency-first**: lower frequencies decay during
#   :math:`t \in [0, 0.5]`; higher frequencies decay during
#   :math:`t \in [0.5, 1]`.
#
# In both processes, each frequency follows a variance-preserving (VP) schedule
# with a time-warped :math:`\beta(t)` — effectively running a full VP decay in
# half the allotted time window. A standard VP process with the same
# :math:`\beta(t)` schedule serves as the baseline.
#
# We evaluate both processes on a synthetic signal that is a sum of a
# low-frequency and a high-frequency cosine wave, then compare the generated
# samples from all three models.

# %%
# Synthetic Dataset
# ^^^^^^^^^^^^^^^^^
#
# The dataset consists of two superimposed cosine waves sharing the same phase.
# The low-frequency component has frequency :math:`1` and constant amplitude
# :math:`1.0`. The high-frequency component has frequency :math:`7` and
# constant amplitude :math:`0.3`. The shared phase :math:`\phi` is drawn
# uniformly from :math:`[0, 2\pi]` for each sample. Each signal is defined
# on the interval :math:`[0, 1]` and sampled at :math:`128` equidistant points.
#
# The ``train``, ``load_trained`` and ``save_model`` flags below control
# whether the generators used later in the example are trained from scratch
# or restored from the pre-trained weights shipped with the example. Set
# ``train = True`` to retrain; by default the pre-trained models are loaded
# from ``./models/frequency/``.

# %%

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import torch

from skfda import FDataGrid

seed = 13
random_state = np.random.RandomState(seed=seed)

device = "cpu"

n_samples = 2000
n_points = 128
data_matrix = np.zeros((n_samples, n_points))
grid_points = np.arange(n_points) / n_points

load_trained = True
train = False
save_model = False


frequency_small = 1
frequency_big = 7

phase_range = (0., 2 * np.pi)
phase = random_state.uniform(
    phase_range[0], phase_range[1], size=(n_samples, 1),
)

# Implicitly broadcast to (n_samples, n_points)
# Base cosine
data_matrix = np.cos(
    2 * np.pi * frequency_small * grid_points[np.newaxis, :] + phase,
)
# Second cosine higher frequency
data_matrix += 0.3 * np.cos(
    2 * np.pi * frequency_big * grid_points[np.newaxis, :] + phase,
)

fd = FDataGrid(data_matrix=data_matrix, grid_points=grid_points)


fig, ax = plt.subplots()
fd[:3].plot(axes=ax)
ax.set_title("Data functions")

plt.show()


# %%
# Frequency-Selective Beta Schedules
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#
# We classify frequencies as *high* if they lie above the midpoint between the
# two signal frequencies, and *low* otherwise. The key expression controlling
# the time-warping is:
#
# .. code-block:: python
#
#    masked_t = 2 * torch.max(0.5 * mask, t) - mask
#
# - **`mask = 0` (first frequency group)**: the schedule is active from the
#   start, compressing the VP decay into the :math:`t \in [0, 0.5]` window via
#   the transformation :math:`2t`.
#
# - **`mask = 1` (second frequency group)**: the schedule is held at zero noise
#   during the first half of the process. The active window
#   :math:`t \in [0.5, 1.0]` is then mapped to :math:`[0, 1]` via the
#   time-shifted transformation :math:`2\max(0.5, t) - 1`.
#
# Finally, the entire schedule is multiplied by :math:`2` so that each
# frequency group integrates to the same cumulative noise as the standard
# VP schedule, preserving the signal-to-noise ratio at the end of the
# forward process:
#
# .. math::
#
#    \begin{aligned}
#    &\int_{0}^{1} \beta(t) dt
#    = \int_0^{1} \beta_{min} + (\beta_{max}-\beta_{min}) t\, dt
#    = \frac{\beta_{min} + \beta_{max}}{2} \\[2ex]
#    \\
#    &\int_{0}^{1} \beta^*(t) dt = \begin{cases}
#    \int_0^{0.5}2 (\beta_{min} + (\beta_{max} - \beta_{min}) 2 t ) dt
#    = \frac{\beta_{min} + \beta_{max}}{2}
#    \quad \quad \quad \text{if mask = 0} \\[2ex]
#    \\
#    \int_{0.5}^{1} 2 (\beta_{min} + (\beta_{max} - \beta_{min}) (2 t -1)) dt
#    = \frac{\beta_{min} + \beta_{max}}{2}
#    \quad \text{if mask = 1}
#    \end{cases}
#    \end{aligned}

# %%
from torch import Tensor

from skfda.ml.generative import (
    CirculantSymmetricMatrixDiffusionProcess,
)

num_eigen_values = n_points // 2 + 1

beta_min = .01
beta_max = 15.0

# Frequencies above this midpoint are treated as "high", the rest as "low"
freq_mean = (frequency_small + frequency_big) // 2


def beta_high(t: Tensor) -> Tensor:
    """Beta schedule that accelerates high freq and delays low freq."""
    # (N, 1) -> (N, num_eigen_values)
    t = t.unsqueeze(-1).expand(-1, num_eigen_values)
    mask = torch.ones(num_eigen_values, device=t.device)
    mask[freq_mean:] = 0.
    masked_t = 2 * torch.max(0.5 * mask, t) - mask
    return 2 * (beta_min + (beta_max - beta_min) * masked_t)

def beta_low(t: Tensor) -> Tensor:
    """Beta schedule that accelerates low freq and delays high freq."""
    # (N, 1) -> (N, num_eigen_values)
    t = t.unsqueeze(-1).expand(-1, num_eigen_values)
    mask = torch.ones(num_eigen_values, device=t.device)
    mask[:freq_mean] = 0.
    masked_t = 2 * torch.max(0.5 * mask, t) - mask
    return 2 * (beta_min + (beta_max - beta_min) * masked_t)

def drift_low_frequency(t: Tensor):
    """Drift term that accelerates low freq and delays high freq."""
    return -0.5 * beta_low(t)

def diffusion_low_frequency(t: Tensor):
    """Diffusion term that accelerates low freq and delays high freq."""
    return torch.sqrt(beta_low(t))

def drift_high_frequency(t: Tensor):
    """Drift term that accelerates high freq and delays low freq."""
    return -0.5 * beta_high(t)

def diffusion_high_frequency(t: Tensor):
    """Diffusion term that accelerates high freq and delays low freq."""
    return torch.sqrt(beta_high(t))


# %%
# Instantiating the Processes
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^
#
# We pass the drift and diffusion functions to
# `CirculantSymmetricMatrixDiffusionProcess` with ``fourier_drift=True`` and
# ``fourier_diffusion=True``, indicating that the supplied functions return
# the eigenvalues of the matrices rather than their first rows. The VP
# baseline is built with the same class: its schedule simply applies the same
# :math:`\beta(t)` to every frequency.

# %%
# Each schedule maps (1, M) eigenvalues + (N, 1) times -> (N, M)
process_low_fq = CirculantSymmetricMatrixDiffusionProcess(
    drift_term=drift_low_frequency,
    diffusion_term=diffusion_low_frequency,
    fourier_drift=True,
    fourier_diffusion=True,
    seed=seed,
)

process_high_fq = CirculantSymmetricMatrixDiffusionProcess(
    drift_term=drift_high_frequency,
    diffusion_term=diffusion_high_frequency,
    fourier_drift=True,
    fourier_diffusion=True,
    seed=seed,
)

def normal_beta(t: Tensor) -> Tensor:
    """Standard linear VP beta schedule, for reference."""
    t = t.unsqueeze(-1)

    return beta_min + (beta_max - beta_min) * t

def drift_vp(t: Tensor) -> Tensor:
    """Standard linear VP drift term, for reference."""
    return -0.5 * normal_beta(t)

def diffusion_vp(t: Tensor) -> Tensor:
    """Standard linear VP diffusion term, for reference."""
    return torch.sqrt(normal_beta(t))

# Ground truth processes
vp_process = CirculantSymmetricMatrixDiffusionProcess(
    drift_term=drift_vp,
    diffusion_term=diffusion_vp,
    fourier_drift=True,
    fourier_diffusion=True,
    seed=seed,
)

# %%
# Fitting the Diffusion Processes
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#
# We call `.fit()` solely to communicate the data dimensionality to each
# process. Ordinarily this step is handled automatically inside
# `FunctionalDiffusionGenerator.fit()`, but since we are using the processes
# standalone here we must pass the dimensionality explicitly.

# %%
_ = process_low_fq.fit(torch.zeros((1,  n_points), device=device))
_ = process_high_fq.fit(torch.zeros((1, n_points), device=device))
_ = vp_process.fit(torch.zeros((1, n_points), device=device))

# %%
# Mean and Variance Evolution in the Frequency Domain
# """""""""""""""""""""""""""""""""""""""""""""""""""
#
# The plots below show the evolution of the conditional mean and variance in
# the frequency domain. The standard VP process (left column) decays all
# frequencies at the same rate. For the other two processes, the low-frequency
# component (blue) and the high-frequency component (orange) decay at different
# rates. In the *Low Frequency Decay First* process, for instance, the blue
# curve reaches zero by :math:`\tau = 0.5` while the orange curve remains
# stationary until the second half of the process.
#
# Note that, to plot the variance evolution, we peek at the private
# ``diagonal_process_._cov`` attribute of each process; this is done for
# visualization purposes only.

# %%
y0 = torch.ones((1, n_points), device=device)  # (1, n_points)
process_names = ["VP", "Low Freq Decays First", "High Freq Decays First"]
fig, axes = plt.subplots(2, 3, figsize=(24, 15), constrained_layout=True)

eigen_values_to_plot = [frequency_small, frequency_big]

for i, process in enumerate([vp_process, process_low_fq, process_high_fq]):
    ax = axes[0, i]
    mean_over_time = []
    t_vals = torch.linspace(
        0., 1., steps=100, requires_grad=False, device=device,
    )

    for t in t_vals:
        mean_at_t = process.diagonal_process_.mean_cond(y0, t)
        mean_over_time.append(mean_at_t.cpu().squeeze(0).numpy())
    mean_over_time = np.array(mean_over_time)  # (n_time_points, n_points)

    for j in eigen_values_to_plot:
        ax.plot(t_vals.cpu().numpy(), mean_over_time[:, j], linewidth=4,
                label="High Freq" if j > freq_mean else "Low Freq")
    ax.axhline(y=0.0, color="black", linestyle="--", alpha=0.5)
    ax.set_xlabel(r"$\tau$", fontsize=22)
    ax.set_ylabel(r"$\mu(\tau)$", fontsize=22)
    ax.set_title(f"{process_names[i]}\n", fontsize=25, fontweight="bold")
    ax.tick_params(axis="both", labelsize=20)

    ax.grid(alpha=0.3)

    ax = axes[1, i]
    sigma_over_time = []

    for t in t_vals:
        cov_at_t = process.diagonal_process_._cov(t.reshape(1))  # noqa: SLF001
        sigma_over_time.append(cov_at_t.cpu().squeeze(0).numpy())
    sigma_over_time = np.array(sigma_over_time)  # (n_time_points, n_points)

    for j in eigen_values_to_plot:
        ax.plot(t_vals.cpu().numpy(), sigma_over_time[:, j], linewidth=4,
                label="High Freq" if j > freq_mean else "Low Freq")
    ax.axhline(y=1.0, color="black", linestyle="--", alpha=0.5)
    ax.set_xlabel(r"$\tau$", fontsize=22)
    ax.set_ylabel(r"$\sigma^2(\tau)$", fontsize=22)
    # Make ticks larger and more visible
    ax.tick_params(axis="both", labelsize=20)
    ax.set_title("")
    ax.grid(alpha=0.3)

# ── Shared titles ──────────────────────────────────────────────
row0_y = (axes[0, 0].get_position().y0 + axes[0, 0].get_position().y1) / 2
row1_y = (axes[1, 0].get_position().y0 + axes[1, 0].get_position().y1) / 2

fig.text(0.01, row0_y, "Mean evolution",
         va="center", ha="center", fontsize=25, fontweight="bold", rotation=90)
fig.text(0.01, row1_y, "Variance evolution",
         va="center", ha="center", fontsize=25, fontweight="bold", rotation=90)
handles, labels = axes[0, 0].get_legend_handles_labels()
fig.legend(handles, labels,
           loc="lower center",
           bbox_to_anchor=(0.48, -0.1),
           ncol=2,
           fontsize=25,
           frameon=True)

plt.show()

# %%
# Forward Diffusion: Mean Evolution
# """""""""""""""""""""""""""""""""
#
# The animation below shows the evolution of the conditional mean for each
# process. We plot only the mean rather than a full sample from the
# distribution: adding noise on top would make it harder to appreciate the
# effect of the frequency-selective decay rates.

# %%
from matplotlib import rc
from matplotlib.animation import FuncAnimation

torch_generator = torch.Generator(device).manual_seed(seed)

n_frames = 25
# Evenly spaced diffusion times, one per animation frame
torch_time_steps = torch.linspace(0.0, 1.0, n_frames, device=device)
fd_min = fd.data_matrix.min()
fd_max = fd.data_matrix.max()
sample_function = fd[0]

torch_sample = torch.from_numpy(
    sample_function.data_matrix[..., 0],
).float().to(device)

fig, axes = plt.subplots(1, 3, figsize=(18, 5))

rc("animation", html="jshtml")


process_names = ["VP", "Low Freq", "High Freq"]

processes = [vp_process, process_low_fq, process_high_fq]

lines = []

x_data = np.asarray(fd.grid_points[0])
y_original = torch_sample.cpu().squeeze(0).numpy()

# Initialize the plots

for i, ax in enumerate(axes):

    ax.plot(
        x_data,
        y_original,
        color="grey",
        alpha=1.,
        linestyle="--",
        lw=2.,
        label="Original Data",
    )

    line, = ax.plot([], [], lw=2, color=f"C{i}")
    lines.append(line)
    ax.set_xlim(0,1)
    ax.set_ylim(fd_min - 0.5, fd_max*1.2)
    ax.set_xlabel("x")
    ax.set_title(f"{process_names[i]} at t=0.00")


def update(frame: int) -> list[mpl.artist.Artist]:
    """Creation of each frame of the animation."""
    t = torch_time_steps[frame]
    for i, process in enumerate(processes):

        mean_at_t = process.mean_cond(torch_sample, t)

        noised_function = mean_at_t

        y_data = noised_function.cpu().squeeze(0).numpy()

        lines[i].set_data(x_data, y_data)

        axes[i].set_title(f"{process_names[i]} at t={t.item():.3f}")
    return list(lines)


anim = FuncAnimation(fig, update, frames=n_frames, interval=100)

plt.close()

anim


# %%
# Training the Generators
# """""""""""""""""""""""
#
# We now train three `FunctionalDiffusionGenerator` instances — one per
# diffusion process — on the synthetic dataset. Each generator wraps its
# respective process and learns a score network via denoising score matching.
# Training runs for 200 epochs per model (the generator default). To skip
# training and load pre-trained weights instead, keep ``train = False`` and
# ``load_trained = True`` as set at the top of the example.

# %%
from skfda.ml.generative import FunctionalDiffusionGenerator

generator_low_fq = FunctionalDiffusionGenerator(
    diff_process=process_low_fq, seed=seed, device=device,
)
generator_high_fq = FunctionalDiffusionGenerator(
    diff_process=process_high_fq, seed=seed, device=device,
)
vp_generator = FunctionalDiffusionGenerator(
    diff_process=vp_process, seed=seed, device=device,
)


# %%
if train:
    generator_low_fq.fit(fd)
    generator_high_fq.fit(fd)
    vp_generator.fit(fd)

    if save_model:
        generator_low_fq.save("./models/frequency/frequency_generator_low_fq.pth")
        generator_high_fq.save("./models/frequency/frequency_generator_high_fq.pth")
        vp_generator.save("./models/frequency/frequency_vp_generator.pth")
elif load_trained:
    generator_low_fq = FunctionalDiffusionGenerator.load(
        "./models/frequency/frequency_generator_low_fq.pth",
        device=device,
        diff_process=process_low_fq,
    )
    generator_high_fq = FunctionalDiffusionGenerator.load(
        "./models/frequency/frequency_generator_high_fq.pth",
        device=device,
        diff_process=process_high_fq,
    )
    vp_generator = FunctionalDiffusionGenerator.load(
        "./models/frequency/frequency_vp_generator.pth",
        device=device,
        diff_process=vp_process,
    )
else:
    msg = ("You must choose either to train the generators or "
           "to load pre-trained models.")
    raise ValueError(msg)


# %%
# Reverse Evolution
# """""""""""""""""
#
# The animation below shows one sample per model evolving from pure noise
# into a generated function. We discretize the reverse process into 25
# frames and let the integrator take 10 steps between consecutive frames,
# so generation uses 240 Euler-Maruyama steps in total (24 intervals x 10
# steps) from :math:`t=1` down to :math:`t=0`.

# %%
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import rc
from matplotlib.animation import FuncAnimation

from skfda.ml.generative import (
    EulerMaruyamaIntegrator,
    SDEReverseDiffusionProcess,
)

n_frames = 25
timesteps = np.linspace(1.0, 0.0, n_frames)

n_samples_to_plot = 1  # Choose how many generated samples to visualize

f_data_evo_low_fq = generator_low_fq.generate_evolution(
    timesteps=timesteps,
    n_samples=n_samples_to_plot,
        reverse_process=SDEReverseDiffusionProcess(
            EulerMaruyamaIntegrator(
                n_steps=10,
                seed=seed,
                device=device,
            ),
        ),
)
f_data_evo_high_fq = generator_high_fq.generate_evolution(
    timesteps=timesteps,
    n_samples=n_samples_to_plot,
    reverse_process=SDEReverseDiffusionProcess(
        EulerMaruyamaIntegrator(
            n_steps=10,
            seed=seed,
            device=device,
        ),
    ),
)
vp_f_data_evo = vp_generator.generate_evolution(
    timesteps=timesteps,
    n_samples=n_samples_to_plot,
    reverse_process=SDEReverseDiffusionProcess(
        EulerMaruyamaIntegrator(
            n_steps=10,
            seed=seed,
            device=device,
        ),
    ),
)
f_data_evolutions = [vp_f_data_evo, f_data_evo_low_fq, f_data_evo_high_fq]
model_names = ["VP", "Low Freq First", "High Freq First"]

fig_anim, axes_anim = plt.subplots(1, 3, figsize=(18, 5))
rc("animation", html="jshtml")
# Extract grid points for plotting
x_data = f_data_evolutions[0][0].grid_points[0]

lines_per_ax = [[], [], []]
for i, ax in enumerate(axes_anim):
    for j in range(n_samples_to_plot):
        color = f"C{(j+i*n_samples_to_plot)%10}" # Cycle through colors
        line, = ax.plot([], [], lw=2, color=color, alpha=0.7)
        lines_per_ax[i].append(line)

    ax.set_xlim(x_data.min(), x_data.max())
    # Expand the y-limits slightly beyond the data range; adjust these
    # factors if you change the data or the process.
    ax.set_ylim(fd_min - 0.5, fd_max*1.2)

    # Hide the ticks of both axes
    ax.set_xticks([])
    ax.set_yticks([])

    ax.set_title(f"{model_names[i]} at t=1.000")

def update(frame: int) -> list[mpl.artist.Artist]:
    """Creation of each frame of the animation."""
    t = timesteps[frame]
    artists: list[mpl.artist.Artist] = []

    # Iterate over both models/axes
    for i, ax in enumerate(axes_anim):
        # Get the FDataGrid for the current model at the current frame
        current_fdata = f_data_evolutions[i][frame]
        y_data = current_fdata.data_matrix[..., 0]

        # Update the lines for this specific subplot
        for j, line in enumerate(lines_per_ax[i]):
            line.set_data(x_data, y_data[j])
            artists.append(line)

        ax.set_title(f"{model_names[i]} t={t:.3f}", fontsize=18)
    return artists

# Create the animation
anim = FuncAnimation(fig_anim, update, frames=n_frames, interval=100)

plt.close()

anim

# %%
# Comparing Generated Samples
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^
#
# We now generate 100 samples per model and plot a few of them next to the
# original data for a first visual comparison.

# %%
n_gen_samples = 100
low_fq_generated = generator_low_fq.generate(n_gen_samples)
high_fq_generated = generator_high_fq.generate(n_gen_samples)
vp_generator_generated = vp_generator.generate(n_gen_samples)

# %%
fig, axes = plt.subplots(
    1, 4, figsize=(24, 7), sharey=True, layout="constrained",
)
n_gen_to_plot = 3
fd[:n_gen_to_plot].plot(axes=axes[0], lw=3.5, label="Original data")

vp_generator_generated[:n_gen_to_plot].plot(
    axes=axes[1], lw=3.5,label="Equal Decay ~ VP",
)

low_fq_generated[:n_gen_to_plot].plot(
    axes=axes[2], lw=3.5, label="Low Frequency Decay First",
)
high_fq_generated[:n_gen_to_plot].plot(
    axes=axes[3], lw=3.5, label="High Frequency Decay First",
)
axes[0].set_title("Real Data\n", fontsize=25, fontweight="bold")
axes[1].set_title("VP\n", fontsize=25, fontweight="bold")
axes[2].set_title("Low Freq\n First", fontsize=25, fontweight="bold")
axes[3].set_title("High Freq\n First", fontsize=25, fontweight="bold")

for ax in axes:
    ax.tick_params(axis="both", labelbottom=False, labelleft=False)

plt.show()


# %%
# Evaluating the Results
# ^^^^^^^^^^^^^^^^^^^^^^
#
# To evaluate quantitatively, we exploit the parametric form of the dataset:
# every sample is fully described by the parameters
# :math:`(A_1, A_7, \phi)` of the shared-phase model
# :math:`A_1\cos(2\pi t + \phi) + A_7\cos(14\pi t + \phi)`.
# For each generated function we recover these parameters by minimizing the
# MSE via L-BFGS-B; since the objective is periodic and non-convex in
# :math:`\phi`, we use stratified restarts on the phase.
#
# We then inspect:
#
# - **MSE**: how closely each generated function matches its best-fitting
#   parametrized form — lower values mean the generator produces functions of
#   the correct shape.
# - **Amplitudes** (:math:`A_1`, :math:`A_7`): the generator should reproduce
#   the constant amplitudes of the dataset (true values: 1.0 and 0.3).
# - **Phase** (:math:`\phi`): the generator should reproduce the
#   :math:`\mathcal{U}(0, 2\pi)` distribution; we quantify the deviation from
#   uniform with the Wasserstein-1 distance.

# %%
from scipy.optimize import minimize
from scipy.stats import iqr, wasserstein_distance


def recover_params_frequency(fd_gen, freq_small=1, freq_big=7,
                              n_restarts=10, param_bounds=None, seed=0):
    """Recovers the parameters (A1, A7, phi) of each function.

    For each function, fits A1, A7 and phi minimizing the MSE of
    A1*cos(2*pi*freq_small*t + phi) + A7*cos(2*pi*freq_big*t + phi).
    Both cosines share phi. Uses stratified restarts on phi because the
    objective is periodic and non-convex in the phase.

    Args:
        fd_gen: FDataGrid with the functions to fit.
        freq_small: Frequency of the first cosine.
        freq_big: Frequency of the second cosine.
        n_restarts: Number of stratified restarts on phi.
        param_bounds: Bounds for (A1, A7, phi) as a list of tuples.
        seed: Seed for the random initial guesses.

    Returns:
        params: Array of shape (N, 3) with the fitted [A1, A7, phi].
        mses: Array of shape (N,) with the MSE of each fit.
    """
    if param_bounds is None:
        param_bounds = [(0.0, 1.5), (0.0, 0.6), (0.0, 2 * np.pi)]
    rng = np.random.RandomState(seed)
    y = fd_gen.data_matrix.squeeze(-1)
    t = fd_gen.grid_points[0]
    n = y.shape[0]
    params, mses = np.zeros((n, 3)), np.zeros(n)

    def obj(theta, y_i):
        a1, a7, phi = theta
        pred = a1 * np.cos(2 * np.pi * freq_small * t + phi) \
             + a7 * np.cos(2 * np.pi * freq_big * t + phi)
        return np.mean((y_i - pred) ** 2)

    phi_guesses = np.linspace(
        param_bounds[2][0], param_bounds[2][1], n_restarts, endpoint=False,
    )

    for i in range(n):
        best_mse, best_theta = np.inf, np.zeros(3)
        for j in range(n_restarts):
            theta0 = np.array([rng.uniform(lo, hi) for lo, hi in param_bounds])
            theta0[2] = phi_guesses[j]
            res = minimize(obj, theta0,
                           args=(y[i],),
                           method="L-BFGS-B",
                           bounds=param_bounds,
                           options={"maxiter": 200,"ftol": 1e-10},
                    )
            if res.fun < best_mse:
                best_mse, best_theta = res.fun, res.x.copy()
        if best_theta[0] < 0:
            best_theta[0] = -best_theta[0]
            best_theta[2] = (best_theta[2] + np.pi) % (2 * np.pi)
        params[i], mses[i] = best_theta, best_mse
    return params, mses


# %%
# We recover :math:`(A_1, A_7, \phi)` per sample for the real data and for
# each model.

# %%
print("Evaluating frequency model...")
model_names = ("Real Data", "VP", "Low Freq\n First", "High Freq\n First")
colors = ["tab:gray", "tab:blue", "tab:orange", "tab:green"]
fds = [
    fd[:n_gen_samples],
    vp_generator_generated,
    low_fq_generated,
    high_fq_generated,
]
a1_true, a7_true = 1.0, 0.3
param_bounds = [(0.0, 1.5), (0.0, 0.6), phase_range]

recovered, mses_dict = {}, {}
for name, fd_item in zip(model_names, fds, strict=True):
    p, m = recover_params_frequency(
        fd_item, freq_small=frequency_small, freq_big=frequency_big,
        n_restarts=10, param_bounds=param_bounds, seed=0,
    )
    recovered[name], mses_dict[name] = p, m

results_freq = {}

# %%
# Goodness of Fit (MSE)
# """""""""""""""""""""
#
# The first figure shows the histogram of the per-sample MSE for each model.
# It measures how closely each generated function matches its best-fitting
# parametrized form, regardless of which parameters were recovered.

# %%
fig, axes = plt.subplots(1, 4, figsize=(20, 4))

all_mse = np.concatenate(list(mses_dict.values()))
bins_mse = np.linspace(0, np.percentile(all_mse, 98) * 1.1, 35)
for ax, name, color in zip(axes, model_names, colors, strict=True):
    med, iq = np.median(mses_dict[name]), iqr(mses_dict[name])
    ax.hist(mses_dict[name], bins=bins_mse, density=True, alpha=0.75,
            color=color, edgecolor="black")
    ax.axvline(med, color="red", linestyle="--", linewidth=2)
    ax.set_title(
        f"{name}\nMedian={med:.2e}", fontsize=20, fontweight="bold",
    )
    ax.set_ylabel("Density", fontsize=16)
    ax.set_xlabel("MSE", fontsize=16)
    ax.tick_params(axis="both", labelsize=14)
    results_freq[f"{name}_median_mse"] = float(med)
    results_freq[f"{name}_iqr_mse"] = float(iq)
fig.tight_layout()

plt.show()

# %%
# Parameter Distributions
# """""""""""""""""""""""
#
# The second figure shows the recovered parameter distributions, one model
# per row and one parameter per column (:math:`A_1`, :math:`A_7`,
# :math:`\phi`). Fixed bins are used for the amplitudes so all rows share the
# same x-range and bin width, and ``sharey="col"`` lets the tall real-data
# spike set the scale — making it immediately visible how much more spread
# the generated distributions are. The phase column includes the uniform
# reference density and the :math:`W_1` distance to it.

# %%
a1_bins = np.linspace(0.5, 1.5, 51)
a7_bins = np.linspace(0.0, 0.6, 31)
phi_bins = np.linspace(phase_range[0], phase_range[1], 31)
ideal_n = 5000
fig, axes = plt.subplots(
    4, 3, figsize=(15, 16), sharey="col", sharex="col",
)

col_titles = [
    r"Distribution $A_1$",
    r"Distribution $A_7$",
    r"Distribution $\phi$",
]

for col, col_title in enumerate(col_titles):
    axes[0, col].set_title(
        col_title, fontsize=20, fontweight="bold", pad=10,
    )

for row, (name, color) in enumerate(zip(model_names, colors, strict=True)):
    p = recovered[name]

    # Col 0 — A1
    ax = axes[row, 0]
    ax.hist(
        p[:, 0],
        bins=a1_bins,
        density=True,
        alpha=0.75,
        color=color,
        edgecolor="black",
    )

    ax.axvline(
        a1_true,
        color="red",
        linestyle="--",
        linewidth=2,
        label=f"True={a1_true}",
    )

    mu, sig = p[:, 0].mean(), p[:, 0].std()
    if row == 0:
        title_a1 = (r"$\bf{Distribution}\,A_1$"
                    f"\nMean={mu:.3f}, Std={sig:.3f}")
    else:
        title_a1 = f"Mean={mu:.3f}, Std={sig:.3f}"

    ax.set_title(title_a1, fontsize=20)

    ax.set_xlabel(r"$A_1$", fontsize=16)
    ax.set_ylabel("Density", fontsize=16)
    ax.legend(fontsize=14)
    ax.tick_params(axis="both", labelsize=14)
    results_freq[f"{name}_A1_mean"] = float(mu)
    results_freq[f"{name}_A1_std"] = float(sig)

    # Col 1 — A7
    ax = axes[row, 1]
    ax.hist(
        p[:, 1],
        bins=a7_bins,
        density=True,
        alpha=0.75,
        color=color,
        edgecolor="black",
    )

    ax.axvline(
        a7_true,
        color="red",
        linestyle="--",
        linewidth=2,
        label=f"True={a7_true}",
    )

    mu7, sig7 = p[:, 1].mean(), p[:, 1].std()
    if row == 0:
        title_a7 = (r"$\bf{Distribution}\, A_7$"
                    f"\nMean={mu7:.3f}, Std={sig7:.3f}")
    else:
        title_a7 = f"Mean={mu7:.3f}, Std={sig7:.3f}"

    ax.set_title(title_a7, fontsize=20)
    ax.set_xlabel(r"$A_7$", fontsize=16)
    ax.set_ylabel("Density", fontsize=16)
    ax.legend(fontsize=14)
    ax.tick_params(axis="both", labelsize=14)
    results_freq[f"{name}_A7_mean"] = float(mu7)
    results_freq[f"{name}_A7_std"] = float(sig7)

    # Col 2 — phi
    ax = axes[row, 2]
    lo, hi = phase_range
    ax.hist(
        p[:, 2],
        bins=phi_bins,
        density=True,
        alpha=0.75,
        color=color,
        edgecolor="black",
    )

    xr = np.linspace(lo - 0.1 * (hi - lo), hi + 0.1 * (hi - lo), 300)

    ax.plot(xr, np.where((xr >= lo) & (xr <= hi), 1 / (hi - lo), 0),
            "r--", linewidth=1.5, label="Uniform")
    ideal = np.linspace(lo, hi, ideal_n)
    w1 = wasserstein_distance(p[:, 2], ideal)

    if row == 0:
        title_phi = (r"$\bf{Distribution} \, \phi$" + f"\n$W_1$={w1:.4f}")
    else:
        title_phi = f"$W_1$={w1:.4f}"

    ax.set_title(title_phi, fontsize=20)
    ax.set_xlabel(r"$\phi$ (rad)", fontsize=16)
    ax.set_ylabel("Density", fontsize=16)
    ax.legend(fontsize=14)
    ax.tick_params(axis="both", labelsize=14)
    results_freq[f"{name}_W1_phi"] = float(w1)

fig.subplots_adjust(left=0.08, top=0.95, hspace=0.35)
fig.canvas.draw()  # force the layout before computing positions

for row, name in enumerate(model_names):
    row_y = (
        axes[row, 0].get_position().y0 + axes[row, 0].get_position().y1
    ) / 2

    fig.text(0.01, row_y, name, va="center", ha="center",
            fontsize=20, fontweight="bold", rotation=90)
plt.show()

# %%
# Finally, we print a summary of all the metrics.

# %%
print("Frequency dataset results:")
for name in model_names:
    clean_name = name.replace("\n", "")
    print(
        f"  {clean_name}: "
        f"median MSE = {results_freq[f'{name}_median_mse']:.2e} | "
        f"A1 = {results_freq[f'{name}_A1_mean']:.3f} "
        f"± {results_freq[f'{name}_A1_std']:.3f} | "
        f"A7 = {results_freq[f'{name}_A7_mean']:.3f} "
        f"± {results_freq[f'{name}_A7_std']:.3f} | "
        f"W1(phi) = {results_freq[f'{name}_W1_phi']:.4f}",
    )

# %%
# Discussion
# ^^^^^^^^^^
#
# All three generators recover the parametric structure of the dataset well:
# the median MSE of the best parametric fit is below :math:`10^{-3}` for
# every model. Comparing the schedules:
#
# - The **low-frequency-first** process obtains the best overall fit: the
#   lowest median MSE among the generated models (about :math:`3.5 \times
#   10^{-4}` versus :math:`4.9 \times 10^{-4}` for VP), the tightest
#   high-frequency amplitude distribution (:math:`A_7 \approx 0.298 \pm
#   0.005`), and a phase distribution slightly closer to uniform than VP.
#
# - The **high-frequency-first** process matches the low-frequency amplitude
#   mean exactly (:math:`A_1 \approx 1.000`) but is noisier overall: it has
#   the highest median MSE, overestimates :math:`A_7` (about :math:`0.316`
#   versus the true :math:`0.3`), and shows the largest deviation of the
#   phase from uniformity.
#
# - Note that even the real data has :math:`W_1(\phi) \approx 0.19`: with
#   only :math:`100` samples this is the finite-sample floor of the metric,
#   so the values around :math:`0.21` to :math:`0.22` obtained by VP and the
#   low-frequency-first model are close to the best achievable.
#
# In this experiment, decaying the low frequencies first therefore preserves
# the high-frequency content of the signal slightly better than the standard
# VP schedule, while the reverse ordering degrades it. The exact numbers vary
# with the seed and training run, but the ordering of the models is stable.
