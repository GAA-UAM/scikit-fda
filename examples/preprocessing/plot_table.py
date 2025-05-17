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
irregular_tecator = irregular_sample(
    tecator[:100],
    35,
    random_state=1,
)

print(irregular_tecator.start_indices[:5])


# %%
pace = PACE(
    n_components=3,
    bandwidth_mean=np.array([0.1, 50]),
    bandwidth_cov=np.array([0.1, 50]),
)
pace_scores = pace.fit_transform(irregular_tecator)



# %%
import matplotlib.pyplot as plt
import pandas as pd

# First table: increasing number of observations
table_obs = pd.DataFrame({
    "Nº of obs.": [10, 25, 50, 100, 150, 215],
    "Nº of points per obs.": [10] * 6,
    "Execution time (s)": [1.6, 5.2, 10.5, 17.2, 24.9, 43.4],
})

# Second table: increasing number of measurements per observation
table_meas = pd.DataFrame({
    "Nº of obs.": [100] * 7,
    "Nº of points per obs.": [2, 5, 10, 15, 20, 25, 35],
    "Execution time (s)": [0.3, 3.7, 17.2, 50.2, 96.2, 164.8, 356.7],
})

# Plot
fig, ax = plt.subplots(1, 2, figsize=(8, 4), sharex=False)

# First plot: Execution time vs Nº of obs.
ax[0].plot(table_obs["Nº of obs."], table_obs["Execution time (s)"], marker='o')
ax[0].set_title("Execution Time vs Nº of Observations")
ax[0].set_xlabel("Nº of Observations")
ax[0].set_ylabel("Execution Time (s)")
ax[0].grid(True)

# Second plot: Execution time vs Nº of points per obs.
ax[1].plot(table_meas["Nº of points per obs."], table_meas["Execution time (s)"], marker='o', color='orange')
ax[1].set_title("Execution Time vs Nº of Points per Observation")
ax[1].set_xlabel("Nº of Points per Observation")
ax[1].set_ylabel("Execution Time (s)")
ax[1].grid(True)

plt.tight_layout()
plt.show()

# %%
