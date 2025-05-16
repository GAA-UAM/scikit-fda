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
from sklearn.utils import Bunch

import skfda
from skfda.datasets._real_datasets import fetch_tecator
from skfda.datasets._sample_from_fdata import irregular_sample
from skfda.preprocessing.dim_reduction import FPCA, PACE
from skfda.representation import FDataGrid
from skfda.typing._numpy import NDArrayInt

# %%
tecator_bunch: Bunch = fetch_tecator()
tecator: FDataGrid = tecator_bunch.data
assert isinstance(tecator, FDataGrid), "Expected an FDataGrid object"

# %%

def generate_num_measurements(
    percentage: int,
    size: int = 215,
    min_val: int = 1,
    max_val: int = 100,
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

# Store results for plotting
components_all = []
reconstructed_all = []

fpca = FPCA(n_components=n_components)
fpca_scores = fpca.fit_transform(tecator)
reconstructed_fpca = fpca.inverse_transform(fpca_scores)
components_all.append(fpca.components_)
reconstructed_all.append(reconstructed_fpca)



# %%
random_state = 43
data_percentages = np.array([5, 10, 15])

for perc in data_percentages:
    measurements_per_obs = generate_num_measurements(
        perc,
        random_state=random_state,
    )

    irregular_tecator = irregular_sample(
        tecator,
        measurements_per_obs,
        random_state=random_state,
    )
    irregular_tecator.coordinate_names = tecator.coordinate_names
    irregular_tecator.argument_names = tecator.argument_names
    irregular_tecator.dataset_name = f"Tecator {perc}%"

    pace = PACE(
        n_components=n_components,
        bandwidth_mean=np.array([0.1, 10]),
        bandwidth_cov=10.0,
    )
    pace_scores = pace.fit_transform(irregular_tecator)
    reconstructed_pace = pace.inverse_transform(pace_scores)
    components_all.append(pace.components_)
    reconstructed_all.append(reconstructed_pace)

# %%
fig, axes = plt.subplots(2, 2, figsize=(12, 8))
titles = ["Regular FPCA", "PACE (5%)", "PACE (10%)", "PACE (15%)"]
for i, ax in enumerate(axes.flat):
    components_all[i].plot(axes=ax)
    ax.set_title(titles[i])

plt.tight_layout()
plt.show()

# %%
fig, axes = plt.subplots(2, 2, figsize=(12, 8))
for i, ax in enumerate(axes.flat):
    reconstructed_all[i].plot(axes=ax, color="lightgrey")
    reconstructed_all[i].mean().plot(axes=ax, color="darkblue", linewidth=2)
    ax.set_title(titles[i])

plt.tight_layout()
plt.show()

# %%
