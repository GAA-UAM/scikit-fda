"""
Magnitude-Shape Plot synthetic example
======================================

Shows the use of the MS-Plot applied to a synthetic dataset.
"""

# Author: Carlos Ramos Carreño
# License: MIT

# sphinx_gallery_thumbnail_number = 3

# sphinx_gallery_start_ignore
# ruff: noqa: PLR2004
# ruff: noqa: SIM300
# sphinx_gallery_end_ignore

# %%
# First, we generate a synthetic dataset following [DaWe18]_

import numpy as np

from skfda.datasets import make_gaussian_process
from skfda.misc.covariances import Exponential

random_state = np.random.RandomState(0)
n_samples = 200

fd = make_gaussian_process(
    n_samples=n_samples,
    n_features=100,
    cov=Exponential(),
    mean=lambda t: 4 * t,
    random_state=random_state,
)

# %%
# We now add the outliers

magnitude_outlier = make_gaussian_process(
    n_samples=1,
    n_features=100,
    cov=Exponential(),
    mean=lambda t: 4 * t + 20,
    random_state=random_state,
)

shape_outlier_shift = make_gaussian_process(
    n_samples=1,
    n_features=100,
    cov=Exponential(),
    mean=lambda t: 4 * t + 10 * (t > 0.4),
    random_state=random_state,
)

shape_outlier_peak = make_gaussian_process(
    n_samples=1,
    n_features=100,
    cov=Exponential(),
    mean=lambda t: 4 * t - 10 * ((0.25 < t) & (t < 0.3)),
    random_state=random_state,
)

shape_outlier_sin = make_gaussian_process(
    n_samples=1,
    n_features=100,
    cov=Exponential(),
    mean=lambda t: 4 * t + 2 * np.sin(18 * t),
    random_state=random_state,
)

shape_outlier_slope = make_gaussian_process(
    n_samples=1,
    n_features=100,
    cov=Exponential(),
    mean=lambda t: 10 * t,
    random_state=random_state,
)

magnitude_shape_outlier = make_gaussian_process(
    n_samples=1,
    n_features=100,
    cov=Exponential(),
    mean=lambda t: 4 * t + 2 * np.sin(18 * t) - 20,
    random_state=random_state,
)


fd = fd.concatenate(
    magnitude_outlier,
    shape_outlier_shift,
    shape_outlier_peak,
    shape_outlier_sin,
    shape_outlier_slope,
    magnitude_shape_outlier,
)

# %%
# The data is plotted to show the curves we are working with.

import matplotlib.pyplot as plt

labels = [0] * n_samples + [1] * 6

fd.plot(
    group=labels,
    group_colors=["lightgrey", "black"],
)
plt.show()

# %%
# The MS-Plot is generated. In order to show the results, the
# :func:`~skfda.exploratory.visualization.MagnitudeShapePlot.plot`
# method is used.

from skfda.exploratory.visualization import MagnitudeShapePlot

msplot = MagnitudeShapePlot(fd)

msplot.plot()
plt.show()

# %%
# To show the utility of the plot, the curves are plotted showing each outlier
# in a different color

labels = [0] * n_samples + [1, 2, 3, 4, 5, 6]
colors = [
    "lightgrey",
    "orange",
    "blue",
    "black",
    "green",
    "brown",
    "lightblue",
]

fd.plot(
    group=labels,
    group_colors=colors,
)
plt.show()

# %%
# We now show the points in the MS-plot using the same colors

fig, ax = plt.subplots()
ax.scatter(
    msplot.points[:, 0].ravel(),
    msplot.points[:, 1].ravel(),
    c=colors[:1] * n_samples + colors[1:],
)
ax.set_title("MS-Plot")
ax.set_xlabel("magnitude outlyingness")
ax.set_ylabel("shape outlyingness")
plt.show()

# %%
# .. rubric:: References
# .. [DaWe18] Dai, Wenlin, and Genton, Marc G. "Multivariate functional data
#    visualization and outlier detection." Journal of Computational and
#    Graphical Statistics 27.4 (2018): 923-934.
