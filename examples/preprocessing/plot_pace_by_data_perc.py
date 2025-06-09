"""
Functional Principal Component Analysis through Conditional Expectation
=======================================================================

Explores an alternative way to do functional principal component analysis for
irregularly sampled data.
"""

# Author: Alejandro Arias Gomez
# License: MIT

# sphinx_gallery_thumbnail_number = 2

# %%
import matplotlib.pyplot as plt
import numpy as np

from skfda.datasets._real_datasets import fetch_weather
from skfda.datasets._sample_from_fdata import irregular_sample
from skfda.preprocessing.dim_reduction import FPCA, PACE
from skfda.representation import FDataGrid
from skfda.typing._numpy import NDArrayInt

# %%
canadian = fetch_weather().data.coordinates[0]
assert isinstance(canadian, FDataGrid), "Expected an FDataGrid object"

# %%

def generate_num_measurements(
    percentage: int,
    size: int = 35,
    min_val: int = 1,
    max_val: int = 365,
    random_state: int | None = None,
) -> NDArrayInt:
    """
    Calculate the number of points of each observation.

    Calculate the number of points of each observation to transform a regular
    dataset into an irregular one, based on a percentage of measurement from
    the whole dataset.

    Args:
        percentage: percentage of measurements to retain.
        size: number of observations.
        min_val: minimum number of measurements to retain per observations.
        max_val: number of measurements per observation.
        random_state: random state.

    Returns:
        Array of integers with the number of time points to retain.
    """
    target_sum = round((percentage / 100) * size * max_val)

    base = np.full(size, min_val)
    remainder = target_sum - base.sum()

    rng = np.random.default_rng(random_state)

    if remainder > 0:
        increments = rng.multinomial(remainder, [1/size] * size)
        result = base + increments
    else:
        result = base.copy()

    return result.tolist()

# %%
n_components = 3
components_all = []
reconstructed_all = []
mse_all = []

random_state = 11
data_percentages = np.array([10, 20, 35])

for perc in data_percentages:
    measurements_per_obs = generate_num_measurements(
        perc,
        random_state=random_state,
    )

    irregular_canadian = irregular_sample(
        canadian,
        measurements_per_obs,
        random_state=random_state,
    )
    irregular_canadian.coordinate_names = canadian.coordinate_names
    irregular_canadian.argument_names = canadian.argument_names

    pace = PACE(
        n_components=n_components,
        bandwidth_mean=np.array([0.1, 10]),
        bandwidth_cov=10.0,
        n_grid_points=25,
    )
    pace_scores = pace.fit_transform(irregular_canadian)
    reconstructed_pace = pace.inverse_transform(pace_scores)
    components_all.append(pace.components_)
    reconstructed_all.append(reconstructed_pace)

    grid1 = canadian.grid_points[0]
    grid2 = reconstructed_pace.grid_points[0]

    common_grid = np.intersect1d(grid1, grid2)
    idx1 = np.where(np.isin(grid1, common_grid))[0]
    idx2 = np.where(np.isin(grid2, common_grid))[0]

    aligned_true = canadian.data_matrix[:, idx1]
    aligned_pred = reconstructed_pace.data_matrix[:, idx2]

    mse_per_curve = np.mean((aligned_true - aligned_pred) ** 2, axis=1)
    average_mse = np.mean(mse_per_curve)

    mse_all.append(average_mse)

print(mse_all)

# %%
fpca = FPCA(n_components=n_components)
fpca_scores = fpca.fit_transform(canadian)
reconstructed_fpca = fpca.inverse_transform(fpca_scores)
components_all.append(fpca.components_)
reconstructed_all.append(reconstructed_fpca)

grid1 = canadian.grid_points[0]
grid2 = reconstructed_fpca.grid_points[0]

common_grid = np.intersect1d(grid1, grid2)
idx1 = np.where(np.isin(grid1, common_grid))[0]
idx2 = np.where(np.isin(grid2, common_grid))[0]

aligned_true = canadian.data_matrix[:, idx1]
aligned_pred = reconstructed_fpca.data_matrix[:, idx2]

mse_per_curve = np.mean((aligned_true - aligned_pred) ** 2, axis=1)
average_mse = np.mean(mse_per_curve)

print(average_mse)

# %%
fig, axes = plt.subplots(2, 2, figsize=(12, 8))
titles = ["PACE (10%)", "PACE (20%)", "PACE (35%)", "Regular FPCA"]
for i, ax in enumerate(axes.flat):
    components_all[i].plot(axes=ax)
    ax.set_title(titles[i])

plt.tight_layout()
plt.show()

# %%
fig, axes = plt.subplots(2, 2, figsize=(12, 8))
for i, ax in enumerate(axes.flat):
    reconstructed_all[i].plot(axes=ax, color="gray")
    reconstructed_all[i].mean().plot(axes=ax, color="darkblue", linewidth=3)
    ax.set_title(titles[i])

plt.tight_layout()
plt.show()

# %%
