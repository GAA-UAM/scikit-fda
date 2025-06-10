"""
PACE analysis based on the total percentage of data available
=======================================================================

Explores the effect of the sparseness of the data in the reconstruction of
subject trajectories via the PACE algorithm for irregularly sampled data.
"""

# Author: Alejandro Arias Gomez
# License: MIT

# sphinx_gallery_thumbnail_number = 3

# %%
import matplotlib.pyplot as plt
import numpy as np

from skfda.datasets._real_datasets import fetch_weather
from skfda.datasets._sample_from_fdata import irregular_sample
from skfda.preprocessing.dim_reduction import FPCA, PACE
from skfda.representation import FDataGrid
from skfda.typing._numpy import NDArrayInt

# %%
# This example explores the effect of the sparseness of the data in the
# reconstruction of subject trajectories via the PACE algorithm for irregularly
# sampled data. We simulate different sparsity levels of a regular dataset and
# evaluate the reconstruction accuracy of PACE vs. classical FPCA.
#
# For this experiment we will use the Canadian weather dataset, more precisely
# the first coordinate, which corresponds to the temperatures over each day of
# the year averaged from 1960 to 1994 at 35 different locations.
canadian = fetch_weather().data.coordinates[0]
assert isinstance(canadian, FDataGrid), "Expected an FDataGrid object"

canadian.plot()
plt.show()

# %%
# We will artificially sparsify the data based on a percentage with the
# following function.
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
# In order to generate sufficient data for our analysis, we will perform the
# PACE algorithm for each of the different data percentages: 10, 20 and 35.
# Our interest resides in analysing the principal components, full
# reconstructions of the data and calculating the average mean squared error
# of the reconstructed curves, compared to the original ones.
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
# For more accurate analysis, we will compare the obtained results with those
# derived from applying FPCA to the full dataset.
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
# We will display the first three principal components extracted under each
# reconstruction setting: using PACE with 10%, 20%, and 35% data retention
# respectively, and one using regular FPCA on the full dataset. Because PACE
# components are estimated from incomplete and irregular data, some of the
# qualitative interpretability is lost in the 10% and 20% cases. Nevertheless,
# the components remain reasonable approximations, consistent in shape and
# orientation (up to sign) with those from full FPCA.
#
# As sparsity decreases in the 35% case, the components estimated via PACE
# visually converge to those obtained from regular FPCA, bearing in mind that
# principal components are identifiable only up to a sign. These results
# confirm that, with enough irregular data, PACE not only reconstructs
# trajectories accurately, but also recovers meaningful modes of variation —
# enabling interpretable decompositions of functional datasets even in sparse
# settings.
fig, axes = plt.subplots(2, 2, figsize=(12, 8))
titles = ["PACE (10%)", "PACE (20%)", "PACE (35%)", "Regular FPCA"]
for i, ax in enumerate(axes.flat):
    components_all[i].plot(axes=ax)
    ax.set_title(titles[i])

plt.tight_layout()
plt.show()

# %%
# We will now plot the reconstructred trajectories, along with the mean curve
# on each case. As expected, the reconstruction captures most of the small
# variations in the dataset, as well as the general structure of the
# temperature curves over the year, with peaks and troughs that correspond to
# summer and winter months, respectively. The reconstruction aligns well with
# the known periodic nature of climate data, especially in high-latitude
# regions like Canada.
#
# Despite the progressive reduction in data density, PACE is able to recover
# the key trends in the functional data with remarkable consistency. At the 10%
# level, the reconstruction is the noisiest, with more visible deviations in
# individual trajectories. Nevertheless, the mean curve and general trends of
# information remain largely consistent with that of the regular FPCA. The
# average MSE at this level is approximately 4.37, confirming a relatively good
# approximation given the extreme sparsity. As the percentage increases to 20%,
# the MSE decreases to 2.93, reflecting the better estimation accuracy. At 35%,
# the reconstruction becomes more precise, with the MSE dropping further to
# approximately 1.41 which, compared to the MSE of 0.68 obtained by FPCA,
# confirms the convergence of PACE to the dense-data solution as more
# observations become available and reaffirming PACE as a powerful and flexible
# alternative for real-world scenarios where dense, regularly sampled data is
# rarely available.
fig, axes = plt.subplots(2, 2, figsize=(12, 8))
for i, ax in enumerate(axes.flat):
    reconstructed_all[i].plot(axes=ax, color="gray")
    reconstructed_all[i].mean().plot(axes=ax, color="darkblue", linewidth=3)
    ax.set_title(titles[i])

plt.tight_layout()
plt.show()

# %%
