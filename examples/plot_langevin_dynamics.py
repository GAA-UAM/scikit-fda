"""
SDE simulation: Langevin dynamics
=======================================================================

This example shows how to use numeric SDE solvers to simulate solutions of
Stochastic Differential Equations (SDEs).
"""

# Author: Pablo Soto Martín
# License: MIT
# sphinx_gallery_thumbnail_number = 1

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import rc
from matplotlib.animation import FuncAnimation
from scipy.stats import multivariate_normal

from skfda.datasets import make_sde_trajectories

# %%
# Langevin dynamics is a mathematical model used to describe the behaviour of
# particles in a fluid, particularly in the context of statistical mechanic
# and molecular dynamics. It was initally formulated by French physicist Paul
# Langevin. In Langevin dynamics, the motion of particles is influenced by both
# determinist  and stochastic forces. The ideas presented by Paul Langevin can
# be applied in various disciplines to simulate the behaviour of particles in
# complex environments. In our case, we will use them to produce samples from
# a probability distribution: a 2-d Gaussian mixture.
#
# Langevin dynamics enable us to sample from probability distributions from
# which a non-normalised pdf is known. This is possible thanks to the use
# of the score function. Given a probability density function
# :math:`p(\mathbf{x}),` the score function is defined as the gradient of its
# logarithm
#
# .. math::
#
#   \nabla_\mathbf{x} \log p(\mathbf{x}).
#
# For example, if :math:`p(\mathbf{x}) = \frac{q(\mathbf{x})}{Z}`, where
# :math:`q(\mathbf{x}) \geq 0` is known but :math:`Z` is a not known
# normalising constant, then the score of :math:`p` is
#
# .. math::
#
#   \nabla_\mathbf{x} \log p(\mathbf{x}) = \nabla_\mathbf{x} \log q(\mathbf{x})
#   - \nabla_\mathbf{x} \log Z =  \nabla_\mathbf{x} \log q(\mathbf{x}),
#
# which is known.
#
# Once we know the score function, we can sample from the
# probability distribution :math:`p(\mathbf{x})` using a dynamic driven by
# SDEs. The idea is to define an SDE whose stationary distribution is
# :math:`p(\mathbf{x})`. If we evolve the SDE
#
# .. math::
#
#   d\mathbf{X}(t) = \nabla_\mathbf{x} \log p(\mathbf{X}(t))dt +
#   \sqrt{2}d\mathbf{W}(t)
#
# from an arbitrary, sufficiently smooth initial distribution
# :math:`\mathbf{X}(0) \sim \pi_0(\mathbf{x})`, we get that for
# :math:`t \rightarrow \infty`, the marginal probability distribution of the
# process converges to the distribution :math:`p(\mathbf{x})`. The initial
# distribution :math:`\pi_0(\mathbf{x})` could be any sufficiently smooth
# distribution.
#
# We will use scikit-fda to simulate this process. We will exemplify this
# use case by sampling from a 2-d Gaussian mixture, but the same steps can be
# applied to other distributions.
#
# We will start by defining some notation. The Gaussian mixture is composed
# of :math:`N` Gaussians of mean :math:`\mu_n` and covariance matrix
# :math:`\Sigma_n`. For the sake of simplicity, we will suppose the covariance
# matrices are diagonal. Let :math:`\sigma_n` then be the corresponding vector
# of standard deviations. Each Gaussian will be weighted by :math:`\omega_n`,
# such that :math:`\sum_{n=1}^N \omega_n = 1`. So, if :math:`p_n(x)` is the pdf
# for the n-th Gaussian, then the pdf of the mixture is
# :math:`p(x) = \sum_{n=1}^{N}\omega_n p_n(x)`.
#
# To compute the score, we can use the chain rule:
#
# .. math::
#
#   \nabla_x \log p(x) =  \frac{\nabla_x p(x)}{p(x)} =
#   \frac{\sum_{n=1}^{N}\omega_n\nabla_x p_n(x)}{\sum_{n=1}^{N}\omega_n p_n(x)}
#   = \frac{\sum_{n=1}^{N}\omega_n p_n(x) \frac{x - \mu_n}{\sigma_n}}
#   {\sum_{n=1}^{N}\omega_n p_n(x)}.
#
# We start by defining functions that compute the pdf, log_pdf and score of the
# distribution.

means = np.array([[-1, -1], [3, 2], [0, 2]])
cov_matrices = np.array(
    [
        [[0.4, 0], [0, 0.4]],
        [[0.5, 0], [0, 0.5]],
        [[0.2, 0], [0, 0.2]],
    ],
)

probabilities = np.array([0.3, 0.6, 0.1])


def pdf_gaussian_mixture(
    x: np.ndarray,
    weight: np.ndarray,
    mean: np.ndarray,
    cov: np.ndarray,
) -> np.ndarray:
    """Pdf of a 2-d Gaussian distribution of N Gaussians."""
    n_gaussians, dim = np.shape(means)
    return np.sum(
        [weight[n] * multivariate_normal.pdf(x, mean[n], cov[n])
         for n in range(n_gaussians)
         ],
        axis=0,
    )


def log_pdf_gaussian_mixture(
    x: np.ndarray,
    weight: np.ndarray,
    mean: np.ndarray,
    cov: np.ndarray,
) -> np.ndarray:
    """Log-pdf of a 2-d Gaussian distribution of N Gaussians."""
    return np.log(pdf_gaussian_mixture(x, weight, mean, cov))


def score_gaussian_mixture(
    x: np.ndarray,
    weight: np.ndarray,
    mean: np.ndarray,
    cov: np.ndarray,
) -> np.ndarray:
    """Score of a 2-d Gaussian distribution of N Gaussians."""
    n_gaussians, dim = np.shape(means)
    score = np.zeros_like(x)
    pdf = pdf_gaussian_mixture(x, weight, mean, cov)

    for n in range(n_gaussians):
        score_weight = weight[n] * (x - mean[n]) / np.sqrt(np.diag(cov[n]))
        score += (
            score_weight
            * multivariate_normal.pdf(x, mean[n], cov[n])[:, np.newaxis]
        )

    return -score / pdf[:, np.newaxis]

# %%
# Once we have defined the pdf and the score of the distribution, we can
# visualize them with a contour plot of the logprobability and the vector
# field given by the score.


x_range = np.linspace(-4, 6, 100)
y_range = np.linspace(-4, 6, 100)
x_score_range = np.linspace(-4, 6, 15)
y_score_range = np.linspace(-4, 6, 15)

X, Y = np.meshgrid(x_range, y_range)
coords = np.c_[X.ravel(), Y.ravel()]

Z = log_pdf_gaussian_mixture(coords, probabilities, means, cov_matrices)
Z = Z.reshape(X.shape)

X_score, Y_score = np.meshgrid(x_score_range, y_score_range)
coords_score = np.c_[X_score.ravel(), Y_score.ravel()]

score = score_gaussian_mixture(
    coords_score,
    probabilities,
    means,
    cov_matrices,
)
score = score.reshape(X_score.shape + (2,))
score_x_coord = score[:, :, 0]
score_y_coord = score[:, :, 1]

plt.contour(X, Y, Z, levels=25, cmap='autumn')
plt.quiver(X_score, Y_score, score_x_coord, score_y_coord, scale=200)
plt.xticks([])
plt.yticks([])
plt.title("Score of a Gaussian mixture", y=1.02)
plt.show()

# %%
# As we can see in the image, the score function is a vector which points
# in the direction in which :math:`\log p(\mathbf{x})` grows faster. Also, if
# :math:`\mathbf{X}(t)` is in an low probability region,
# :math:`\nabla_\mathbf{x} \log p(\mathbf{X}(t))` will have a big norm, which
# means that points which are far away from the "common" areas will tend
# faster towards more probable ones. In regions with high probability,
# :math:`\nabla_\mathbf{x} \log p(\mathbf{X}(t))` will have a small norm,
# which means that the majority of the samples will remain in that area.
#
# We can now proceed to define the parameters for the SDE simulation. For
# this example we have chosen than the starting data follow a uniform
# distribution, but any other distribution is equally valid.


def langevin_drift(
    t: float,
    x: np.ndarray,
) -> np.ndarray:
    """Drift term of the Langevin dynamics."""
    return score_gaussian_mixture(x, probabilities, means, cov_matrices)


def langevin_diffusion(
    t: float,
    x: np.ndarray,
) -> np.ndarray:
    """Diffusion term of the Langevin dynamics."""
    return np.sqrt(2) * np.eye(x.shape[-1])


rnd_state = np.random.RandomState(1)


def initial_distribution(
    size: int,
    random_state: np.random.RandomState,
) -> np.ndarray:
    """Uniform initial distribution"""
    return random_state.uniform(-4, 6, (size, 2))


n_samples = 400
n_grid_points = 200
grid_points_per_frame = 10
frames = 20
# %%
# We use :func:`skfda.datasets.make_sde_trajectories` method of the datasets
# module to simulate solutions of the SDE. More information on how to use it
# can be found in the example
# :ref:`sphx_glr_auto_examples_plot_sde_simulation.py`.
t_0 = 0
t_n = 3.0

fd = make_sde_trajectories(
    initial_condition=initial_distribution,
    n_grid_points=n_grid_points,
    n_samples=n_samples,
    start=t_0,
    stop=t_n,
    drift=langevin_drift,
    diffusion=langevin_diffusion,
    random_state=rnd_state,
)

# %%
# We can visualize how the samples start from a random distribution and
# gradually move to regions of higher mass probability pushed by the score
# drift. The final result is an approximate sample from the target
# distribution.

points = fd.data_matrix
fig, ax = plt.subplots()

plt.contour(X, Y, Z, levels=25, cmap='autumn')
plt.quiver(X_score, Y_score, score_x_coord, score_y_coord, scale=200)
rc('animation', html='jshmtl')
scatter = None


def update(frame: int) -> None:
    """Creation of each frame of the animation."""
    global scatter

    if scatter:
        scatter.remove()

    ax.set_xlim(-4, 6)
    ax.set_ylim(-4, 6)
    ax.set_xticks([])
    ax.set_yticks([])
    x = points[:, grid_points_per_frame * frame, 0]
    y = points[:, grid_points_per_frame * frame, 1]
    scatter = ax.scatter(x, y, s=5, c='dodgerblue')


animation = FuncAnimation(fig, update, frames=frames, interval=500)
plt.close()
animation
