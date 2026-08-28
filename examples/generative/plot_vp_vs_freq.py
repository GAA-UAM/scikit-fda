"""
Matrix diffusion processes.
=======================================================================

This example shows how to use diffusion process with matrix terms to
introduce structure on the noise injection.
"""

# Author: Diego Rodríguez Ortiz
# License: MIT
# sphinx_gallery_thumbnail_number = 2

# %%
# This example introduces a novel non-diagonal diffusion process specifically
# designed for functional data generation. Before diving in, reviewing
# :ref:`sphx_glr_auto_examples_generative_plot_vp_vs_ve.py` is recommended to
# grasp the core concepts of diffusion models—specifically, how they inject
# noise into data and learn to reverse that process to generate new samples.
#
# In standard literature and previous examples, noise is injected
# independently at each pixel or grid point. This independent
# injection means that the terms of the Stochastic Differential
# Equation (SDE) defining the diffusion process are diagonal.
# The standard diagonal SDE is written as follows:
#
# .. math::
#
#    d\mathbf{X}(\tau) =
#    f(\mathbf{X}(\tau), \tau)\, d\tau + g(\tau)\, d\mathbf{W}(\tau)
#
# .. note::
#
#    **Note on Notation**: To avoid confusion with :math:`t` (the spatial/time
#    variable over which the functional data is defined), we use :math:`\tau`
#    to represent the diffusion time variable. Because the data is represented
#    as an ``FDataGrid``, it can be treated as a vector. Therefore, we omit
#    the spatial/time variable t and denote the function simply as
#    :math:`\mathbf{X}`, where :math:`\mathbf{X}(\tau)` represents the
#    function at diffusion time :math:`\tau`.
#
# ----
#
# Comparing scalar and matrix drift terms
# ---------------------------------------
#
# To induce spatial correlation, we introduce a non-diagonal drift term,
# :math:`f(\mathbf{X}(\tau), \tau)`, and compare it to the standard Variance
# Preserving (VP) method. The SDE governing the non-diagonal process is
# defined as follows:
#
# .. math::
#
#    d\mathbf{X}(\tau) =
#    \mathbf{B}(\tau) \mathbf{X}(\tau)\, d\tau
#    + \sqrt{\beta(\tau)}\, d\mathbf{W}(\tau)
#
# The drift matrix :math:`\mathbf{B}(\tau)` is given by this equation:
#
# .. math::
#
#    B(\tau) = -\frac{1}{2}\beta(\tau)\,I + \Delta\mathbf{L}
#
# In this formula, :math:`\beta(\tau)` follows a linear schedule defined as
# :math:`\beta(\tau) = \beta_{\min} + (\beta_{\max} - \beta_{\min})\tau`.
# The diffusion term, :math:`\sqrt{\beta(\tau)}`, remains identical for both
# processes. The key difference lies in :math:`\Delta\mathbf{L}`, a circulant
# symmetric matrix that creates spatial correlation between neighboring grid
# points.
#
# By contrast, the standard diagonal VP process simply sets
# :math:`\Delta\mathbf{L} = 0`, resulting in this equation:
#
# .. math::
#
#    d\mathbf{X}(\tau) =
#    -\frac{1}{2}\beta(\tau)\mathbf{X}(\tau)\,d\tau
#    + \sqrt{\beta(\tau)}\, d\mathbf{W}(\tau)
#
# ----
#
# The Spatial Coupling Matrix
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^
#
# The structure of the circulant symmetric matrix :math:`\Delta\mathbf{L}` is
# highly advantageous because it allows for efficient computation through
# diagonalization using the Fast Fourier Transform (FFT). The matrix
# :math:`\Delta\mathbf{L}` is explicitly structured as follows:
#
# .. math::
#
#    \Delta\mathbf{L} = \begin{bmatrix}
#    D & l_{1} & l_{2} & \cdots & l_{1} \\
#    l_{1} & D & l_{1} & \cdots & l_{2} \\
#    l_{2} & l_{1} & D & \cdots & l_{3} \\
#    \vdots & \vdots & \vdots & \ddots & \vdots \\
#    l_{1} & l_{2} & l_{3} & \cdots & D
#    \end{bmatrix}
#
# The term :math:`l_{j}` defines the coupling strength between grid points
# separated by a distance :math:`j`. The entries are proportional to a chosen
# decay law :math:`w_j` and normalized so that the off-diagonal entries of
# each row sum to :math:`D`. For instance, using the inverse square law:
#
# .. math::
#
#    l_{j} = D\,\frac{w_j}{\sum_{k} w_k},
#    \qquad w_j = \frac{1}{j^2}
#
# Here, :math:`D` is a hyperparameter that controls the overall strength of
# the spatial coupling. Note that, unlike :math:`\beta(\tau)`, the coupling
# matrix is constant in diffusion time. Since :math:`D` is small compared to
# :math:`\beta(\tau)`, the process behaves as a small structured perturbation
# of the standard VP process: the coupling slows the decay of the slowly
# varying components of the signal relative to the rapidly oscillating ones,
# injecting spatially correlated structure into the forward process.

# %%

import numpy as np
import torch

from skfda import FDataGrid

seed = 15
random_state = np.random.RandomState(seed)

device = "cpu"

# %%
# Customizing the Spatial Coupling Decay
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#
# This notebook allows you to experiment with different decay laws for
# :math:`w_j` by adjusting the ``method`` parameter in the
# ``get_decay_half_row_weights`` function. The available decay methods are:
#
# - **Inverse Distance**: :math:`w_j = \frac{1}{j}`
# - **Inverse Square**: :math:`w_j = \frac{1}{j^2}`
# - **Exponential**: :math:`w_j = \alpha^j`
#
# In every case, the off-diagonal entries are
# :math:`l_j = D\,w_j / \sum_k w_k`. For the exponential method,
# :math:`\alpha` serves as an additional hyperparameter dictating the
# spatial decay rate.
#
# Defining the Non-Diagonal Drift Term
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#
# To implement the custom non-diagonal drift term, we must define the expected
# circulant symmetric matrix. This is achieved by creating a function that
# returns half of the matrix's first row—specifically, the diagonal term
# :math:`D` followed by the values :math:`l_{j}` for :math:`j=1,...,n/2`
# (the other half of the row is fixed by symmetry).
#
# The ``get_decay_half_row_weights`` function handles this by accepting the
# time step ``t`` (:math:`\tau`), the number of grid points ``m``, the
# hyperparameter ``d`` (:math:`D`), and the calculation ``method`` for
# :math:`w_j`, and returns a tensor containing the half-row values. For
# optimal performance, this function calculates values in-place without
# generating intermediate tensors, which saves both memory and computational
# resources.

# %%

import matplotlib.pyplot as plt
from torch import Tensor

from skfda.ml.generative import (
    CirculantSymmetricMatrixDiffusionProcess,
    VariancePreservingDiffusionProcess,
)


def get_decay_half_row_weights(
    d: float,
    t: Tensor | float,
    m: int,
    method: str = "inverse_square",
    alpha: float = 0.5,
    device: torch.device | str = "cpu",
) -> Tensor:
    """
    Returns the half-row representation of the spatial coupling matrix ΔL.
    The first element is the diagonal, the rest are the off-diagonal elements.
    """
    limit = m // 2
    n = t.shape[0] if isinstance(t, Tensor) else 1
    template = torch.empty((n, limit + 1), device=device, dtype=torch.float32)

    if isinstance(t, Tensor):
        template[:, 0] =  d
    else:
        template[0] =  d

    if limit > 0:
        j = torch.arange(1, limit + 1, device=device, dtype=torch.float32)

        if method == "exponential":
            w = torch.pow(alpha, j)
        elif method == "inverse_distance":
            w = 1.0 / j
        elif method == "inverse_square":
            w = 1.0 / (j ** 2)
        else:
            msg = f"Unknown method: {method}"
            raise ValueError(msg)

        if m % 2 == 0:
            w_sum = 2.0 * torch.sum(w[:-1]) + w[-1]
        else:
            w_sum = 2.0 * torch.sum(w)

        multiplier =  d / w_sum

        if isinstance(t, Tensor):
            template[:, 1:] = w * multiplier.unsqueeze(-1)
        else:
            template[:, 1:] = w * multiplier
    return template


# %%
# Instantiating the Processes
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^
#
# We now set the hyperparameters and define the drift and diffusion terms of
# both processes. The coupling strength is set to :math:`D = 0.02`, small
# relative to :math:`\beta(\tau) \in [0.01, 15]`, so the circulant process
# stays close to the VP baseline while still injecting spatial structure.

# %%
# ── Hyperparameters ─────────────────────────────────────────────────────────
# Linear VP beta schedule: β(τ) = beta_min + (beta_max - beta_min) * τ
beta_min = 0.01
beta_max = 15.0
D = 0.02

n_points = 128
num_rows = n_points // 2 + 1


def _beta_t(t: Tensor) -> Tensor:
    """Linear VP beta schedule."""
    return beta_min + (beta_max - beta_min) * t


# ── Non-diagonal VP drift: B(τ) = -0.5·β(τ)·I + ΔL ─────────────────────────
def drift_half_row(t: Tensor) -> Tensor:
    """Half-row of B(τ) = -0.5*β(τ)*I + ΔL."""
    half_row = get_decay_half_row_weights(
        m=n_points, d=D, t=t, method="inverse_square", device=device,
    )
    beta_t = _beta_t(t)
    if isinstance(t, Tensor):
        half_row[:, 0] = half_row[:, 0] - 0.5 * beta_t
    else:
        half_row[0] = half_row[0] - 0.5 * beta_t
    return half_row


# ── Diffusion term: g(τ) = sqrt(β(τ)) · I ────────────────────────────────────
half_row_diag = torch.zeros(num_rows, device=device)
half_row_diag[0] = 1.0


def diffusion_half_row(t: Tensor) -> Tensor:
    """Half-row of g(τ)·I = sqrt(β(τ))·I."""
    beta_t = _beta_t(t)
    sqrt_beta = torch.sqrt(beta_t)
    return sqrt_beta.unsqueeze(-1) * half_row_diag.unsqueeze(0)


# ── Instantiate processes ────────────────────────────────────────────────────
non_diag_proc = CirculantSymmetricMatrixDiffusionProcess(
    drift_term=drift_half_row,
    diffusion_term=diffusion_half_row,
    fourier_drift=False,
    fourier_diffusion=False,
    seed=seed,
)

diag_proc = VariancePreservingDiffusionProcess(
    beta_schedule="linear",
    beta_min=beta_min,
    beta_max=beta_max,
    seed=seed,
)


# %%
#
# Example: Constant Functions Generation
# --------------------------------------
#
# Now we will test how this new process behaves when we use it to generate
# data. We will use the ``FunctionalDiffusionGenerator`` class to generate
# data using both diffusion processes and compare the results. For this
# first experiment we will use constant functions between :math:`-1` and
# :math:`1` as the original data.
#
# The ``train``, ``load_trained`` and ``save_model`` flags below control
# whether the generators are trained from scratch or restored from the
# pre-trained weights shipped with the example. Set ``train = True`` to
# retrain; by default the pre-trained models are loaded from
# ``./models/vp_vs_freq/``.

# %%
n_samples = 2000
data_matrix = np.zeros((n_samples, n_points))
grid_points = np.linspace(0, 1, n_points)

load_trained = True
train = False
save_model = False

data_matrix = random_state.uniform(
    -1, 1, size =(n_samples, 1),
).repeat(n_points, axis=1)

fd = FDataGrid(data_matrix=data_matrix, grid_points=grid_points)
fig, ax = plt.subplots()
fd[:10].plot(axes=ax)
ax.set_title("Original Data for Training")
plt.show()


# %%
# Fitting and Training the Diffusion Models
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#
# We wrap each diffusion process in a ``FunctionalDiffusionGenerator``, which
# trains a score network via denoising score matching to learn how to reverse
# the corresponding forward process. Both generators share the same seed and
# default training configuration, so the only difference between them is the
# diffusion process they reverse.

# %%

from skfda.ml.generative import FunctionalDiffusionGenerator

non_diag_gen = FunctionalDiffusionGenerator(
    diff_process=non_diag_proc,
    seed=seed,
    device=device,
)
diag_gen = FunctionalDiffusionGenerator(
    diff_process=diag_proc,
    seed=seed,
    device=device,
)


# %%
if train:
    _ = non_diag_gen.fit(fd)
    _ = diag_gen.fit(fd)
    if save_model:
        non_diag_gen.save("./models/vp_vs_freq/non_diag_gen_cte.pth")
        diag_gen.save("./models/vp_vs_freq/diag_gen_cte.pth")
elif load_trained:
    non_diag_gen = FunctionalDiffusionGenerator.load(
        "./models/vp_vs_freq/non_diag_gen_cte.pth",
        device=device,
        diff_process=non_diag_proc,
    )
    diag_gen = FunctionalDiffusionGenerator.load(
        "./models/vp_vs_freq/diag_gen_cte.pth",
        device=device,
        diff_process=diag_proc,
    )
else:
    msg = ("You must choose either to train "
           "the generators or to load pre-trained models.")
    raise ValueError(msg)


# %%
# Generating the Reverse Evolution
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#
# Once the generators are fitted, we can use the ``generate_evolution`` method
# to visualize how ``n_samples`` evolve from pure noise into fully generated
# samples. This process works by integrating the reverse SDE backwards in
# diffusion time, from :math:`\tau=1` down to :math:`\tau=0`, using the
# Euler-Maruyama method to systematically undo the noise injection.
#
# It is important to understand how ``generate_evolution`` handles time
# stepping. The method applies your defined ``n_steps`` between each specified
# frame in your time array. For example, if you want to observe the process at
# :math:`\tau=[1, 0.5, 0]` and set ``n_steps=1000``, the integrator will take
# 1,000 steps from :math:`\tau=1` to :math:`\tau=0.5`, and another 1,000 steps
# from :math:`\tau=0.5` to :math:`\tau=0`.
#
# To keep computation times reasonable in this notebook, we use a more
# efficient step schedule:
#
# - We discretize the total time into **25 frames**, with quadratic spacing
#   so that more frames fall near :math:`\tau=0`, where the most interesting
#   part of the denoising happens.
# - We set the integrator to take only **20 steps** between each frame.
#
# This approach results in a total of **480 integration steps**
# (24 intervals x 20 steps) from :math:`\tau=1` down to :math:`\tau=0`.
# It provides a fast generation process while still maintaining enough
# resolution to visualize the reverse diffusion smoothly.

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
# Quadratic spacing: more frames near τ=0 for better visualization
timesteps = np.linspace(1.0, 0.0, n_frames)**2
n_frames = len(timesteps)
n_samples_to_plot = 1  # Choose how many generated samples to visualize

f_data_evo_circulant = non_diag_gen.generate_evolution(
    timesteps=timesteps,
    n_samples=n_samples_to_plot,
        reverse_process=SDEReverseDiffusionProcess(
            EulerMaruyamaIntegrator(
                n_steps=20,
                seed=seed,
                device=device,
            ),
        ),
)
f_data_evo_vp = diag_gen.generate_evolution(
    timesteps=timesteps,
    n_samples=n_samples_to_plot,
    reverse_process=SDEReverseDiffusionProcess(
        EulerMaruyamaIntegrator(
            n_steps=20,
            seed=seed,
            device=device,
        ),
    ),
)

fd_min = min(
    f_data_evo_circulant[0].data_matrix.min(),
    f_data_evo_vp[0].data_matrix.min(),
)
fd_max = max(f_data_evo_circulant[0].data_matrix.max(),
             f_data_evo_vp[0].data_matrix.max(),
        )

f_data_evolutions = [f_data_evo_circulant, f_data_evo_vp]
model_names = ["Circulant", "VP"]

fig_gen, axes_gen = plt.subplots(1, 2, figsize=(12, 5))
rc("animation", html="jshtml")
# Extract grid points for plotting
x_data = f_data_evolutions[0][0].grid_points[0]

lines_per_ax = [[], []]
for i, ax in enumerate(axes_gen):
    for j in range(n_samples_to_plot):
        color = f"C{(j+i*n_samples_to_plot)%10}" # Cycle through colors
        line, = ax.plot([], [], lw=2, color=color, alpha=0.7)
        lines_per_ax[i].append(line)

    ax.set_xlim(x_data.min(), x_data.max())
    # Expand the y-limits slightly beyond the data range; adjust these
    # factors if you change the data or the process.
    ax.set_ylim(fd_min*0.8 if fd_min > 0 else fd_min * 1.2, fd_max*1.2)


    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title(f"{model_names[i]} t=1.000", fontsize=16)

def update(frame: int) -> list:
    """Creation of each frame of the animation."""
    t = timesteps[frame]
    updated_lines = []

    # Iterate over both models/axes
    for i, ax in enumerate(axes_gen):
        # Get the FDataGrid for the current model at the current frame
        current_fdata = f_data_evolutions[i][frame]
        y_data = current_fdata.data_matrix[..., 0]

        # Update the lines for this specific subplot
        for j, line in enumerate(lines_per_ax[i]):
            line.set_data(x_data, y_data[j])
            updated_lines.append(line)

        ax.set_title(f"{model_names[i]} t={t:.3f}", fontsize=18)

    return updated_lines

# Create the animation
anim = FuncAnimation(fig_gen, update, frames=n_frames, interval=100)

plt.close()
anim

# %%
# Generating Samples
# ^^^^^^^^^^^^^^^^^^
#
# After visualizing the reverse evolution, we can generate a larger number of
# samples (e.g., `n_samples=100`) to perform a more comprehensive evaluation of
# the generated data. This will allow us to compare the quality of samples
# produced by both the standard VP and the circulant processes using
# quantitative metrics.

# %%
n_gen_samples = 100

non_diag_gen_generated = non_diag_gen.generate(n_gen_samples)
diag_gen_generated = diag_gen.generate(n_gen_samples)

# %%

fig_final, axes_final = plt.subplots(
    1, 3, figsize=(18, 5), sharey=True, layout="constrained",
)
n_gen_to_plot = 6

fd[:n_gen_to_plot].plot(axes=axes_final[0], lw=3.5)
diag_gen_generated[:n_gen_to_plot].plot(axes=axes_final[1], lw=3.5)
non_diag_gen_generated[:n_gen_to_plot].plot(axes=axes_final[2], lw=3.5)

axes_final[0].set_title("Original Data\n", fontsize=25, fontweight="bold")
axes_final[1].set_title("VP\n", fontsize=25, fontweight="bold")
axes_final[2].set_title("Circulant\n", fontsize=25, fontweight="bold")

for ax in axes_final:
    # Hide the tick labels: the scale is shared and not meaningful here
    ax.tick_params(axis="both", labelbottom=False, labelleft=False)

plt.show()



# %%
# While visual inspection is helpful, quantitative metrics provide a more
# rigorous way to compare the samples generated by both methods.

# %%
# Evaluation Metrics for Constant Functions
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#
# Let :math:`\{X_i\}_{i=1}^{N}` be the generated samples evaluated on a
# grid of :math:`M` points,
# so :math:`X_i = (X_i(t_1), \ldots, X_i(t_M)) \in \mathbb{R}^M`.
# Each generated function should approximate a constant, so we summarise it
# by its empirical mean
# :math:`\hat{c}_i = \frac{1}{M}\sum_{j=1}^{M} X_i(t_j)`.
#
# Metric 1: Value Diversity (Wasserstein Distance)
# """""""""""""""""""""""""""""""""""""""""""""""""
#
# A well-trained model should produce constants :math:`\hat{c}_i` that cover
# the theoretical range :math:`[-1, 1]` uniformly. We quantify this by the
# Wasserstein-1 distance between the empirical distribution of
# :math:`\{\hat{c}_i\}_{i=1}^N` and the ideal uniform distribution
# :math:`\mathcal{U}(-1,1)`:
#
# .. math::
#
#    W_1\!\left(\hat{\mu},\, \mathcal{U}(-1,1)\right)
#    = \int_{\mathbb{R}} \left|F_{\hat{\mu}}(x) - F_{\mathcal{U}}(x)\right| dx
#
# where :math:`F_{\hat{\mu}}` and :math:`F_{\mathcal{U}}` are the cumulative
# distribution functions of the empirical and uniform distributions,
# respectively. A value close to :math:`0` indicates that the model generates
# constants with the correct diversity. We use the
# ``scipy.stats.wasserstein_distance`` function to compute this metric.
#
# Metric 2: Internal Noise (MSE Distribution)
# """""""""""""""""""""""""""""""""""""""""""
#
# Since each generated function should be *exactly* constant, any variation
# along the grid is spurious noise introduced by the model. For each sample we
# compute the Mean Square Error with respect to its own mean:
#
# .. math::
#
#    \mathrm{MSE}_i
#    = \frac{1}{M} \sum_{j=1}^{M} \bigl(X_i(t_j) - \hat{c}_i\bigr)^2.
#
# We report the distribution of :math:`\{\mathrm{MSE}_i\}_{i=1}^N`,
# summarised by its median. Values close to :math:`0` indicate that the
# generated functions are smooth and free of noise.

# %%
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import wasserstein_distance


def evaluate_constant_model(fd_gen_dict, theoretical_range=(-1, 1)):
    """Plot and compute the constant-function metrics for each model.

    For each entry of ``fd_gen_dict`` (model name -> FDataGrid), plots the
    distribution of the per-sample means against the target uniform
    distribution and the distribution of the internal noise (MSE), and
    returns both metrics in a dictionary.
    """
    n_models = len(fd_gen_dict)
    model_names = list(fd_gen_dict.keys())

    fig, axes = plt.subplots(
        n_models, 2,
        figsize=(15, 4 * n_models),
        sharey="col", sharex="col",
        squeeze=False,
    )

    # ── Column titles (row 0 only, bold, large) ──────────────────────────────
    col_titles = ["Constants Distribution", "Internal Noise (MSE)"]
    for col, col_title in enumerate(col_titles):
        axes[0, col].set_title(
            col_title, fontsize=20, fontweight="bold", pad=10,
        )

    distribution_bins = np.linspace(
        theoretical_range[0] - 0.5,
        theoretical_range[1] + 0.5,
        30,
    )

    error_bins  = np.linspace(0, 0.0005, 30)
    result_dict = {}

    for row, (model_name, fd_gen) in enumerate(fd_gen_dict.items()):
        y_gen = fd_gen.data_matrix.squeeze(-1)
        n_funcs = y_gen.shape[0]

        # ── Metric 1: Value Distribution (Wasserstein) ───────────────────────
        empirical_means = np.mean(y_gen, axis=1)
        ideal_means     = np.linspace(
            theoretical_range[0],
            theoretical_range[1],
            n_funcs,
        )

        w1= wasserstein_distance(empirical_means, ideal_means)

        # ── Metric 2: Internal Noise (MSE) ───────────────────────────────────
        residuals    = y_gen - empirical_means[:, None]
        internal_mse = np.mean(residuals**2, axis=1)
        median_mse   = np.median(internal_mse)

        # ── Col 0 — Histogram of means ───────────────────────────────────────
        theoretical_height = (1.0 /
                              (theoretical_range[1] - theoretical_range[0])
                             )
        x_unif = [
            theoretical_range[0] - 0.5,
            theoretical_range[0],
            theoretical_range[0],
            theoretical_range[1],
            theoretical_range[1],
            theoretical_range[1] + 0.5,
        ]

        y_unif = [0, 0, theoretical_height, theoretical_height, 0, 0]

        ax0 = axes[row, 0]
        ax0.hist(
            empirical_means, bins=distribution_bins, density=True,
            alpha=0.75, color="steelblue", edgecolor="black", label=model_name,
                )
        ax0.plot(
            x_unif, y_unif, color="red", linestyle="--",
            linewidth=2, label="Uniform",
        )

        ax0.set_xlim(theoretical_range[0] - 0.5, theoretical_range[1] + 0.5)
        ax0.set_ylabel("Density", fontsize=16)
        ax0.legend(fontsize=14)
        ax0.tick_params(axis="both", labelsize=14)

        # Row 0: bold column name + metric; other rows: metric only
        title_w = (
            r"$\bf{Distribution\ of\ \hat{c_i}}$" + f"\n$W_1$ = {w1:.5f}"
            if row == 0 else
            f"$W_1$ = {w1:.5f}"
        )
        ax0.set_title(title_w, fontsize=20)

        # ── Col 1 — Histogram of MSE ─────────────────────────────────────────
        ax1 = axes[row, 1]
        ax1.hist(internal_mse, bins=error_bins, density=True,
                 alpha=0.75, color="mediumpurple", edgecolor="black")
        ax1.axvline(median_mse, color="red", linestyle="--", linewidth=2,
                    label=f"Median: {median_mse:.2e}")
        ax1.set_ylabel("Density", fontsize=16)
        ax1.legend(fontsize=14)
        ax1.tick_params(axis="both", labelsize=14)
        ax1.ticklabel_format(style="sci", axis="y", scilimits=(0, 0))
        title_mse = (
            r"$\bf{Distribution\ of\ MSE_i}$"
            f"\nMedian MSE = {median_mse:.2e}"
            if row == 0 else
            f"Median MSE = {median_mse:.2e}"
        )
        ax1.set_title(title_mse, fontsize=20)

        result_dict[model_name] = {
            "wasserstein_distance": w1,
            "median_mse":           median_mse,
        }

    # ── Row labels on the left ───────────────────────────────────────────────
    fig.subplots_adjust(left=0.12, top=0.95, hspace=0.45)
    fig.canvas.draw()  # force layout before computing positions

    for row, name in enumerate(model_names):
        row_y = (
            axes[row, 0].get_position().y0 + axes[row, 0].get_position().y1
        ) / 2
        fig.text(0.02, row_y, name,
                 va="center", ha="center",
                 fontsize=20, fontweight="bold", rotation=90)

    plt.show()
    return result_dict


# %%
# We now evaluate the original data alongside the samples generated by both
# processes.

# %%
data_dict = {
    "Original Data": fd[:n_gen_samples],
    "VP": diag_gen_generated,
    "Circulant": non_diag_gen_generated,
}

dict_results = evaluate_constant_model(data_dict)

print("Constant dataset results:")
for k, v in dict_results.items():
    print(
        f"  {k}: W : {v['wasserstein_distance']:.6f} | "
        f"Median MSE: {v['median_mse']:.2e}",
    )


# %%
# The mean values of the generated samples show similar distributions for both
# methods: the mass concentrates near its mean rather than following the target
# uniform distribution, which reflects the limited capacity of the score
# network for this small training run.
#
# For the internal noise metric, the circulant process tends to produce
# slightly smoother functions (lower intra-sample MSE) than the standard VP,
# consistent with the spatial prior introduced by :math:`\Delta\mathbf{L}`.
# However, the difference is modest.
