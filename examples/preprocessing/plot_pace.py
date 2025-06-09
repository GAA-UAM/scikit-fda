"""
Functional Principal Component Analysis through Conditional Expectation
=======================================================================

Explores an alternative way to do functional principal component analysis for
irregularly sampled data.
"""

# Author: Alejandro Arias Gomez
# License: MIT

# sphinx_gallery_thumbnail_number = 4

# %%
from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np
from sklearn.utils import Bunch

import skfda
from skfda.datasets._real_datasets import fetch_growth
from skfda.datasets._sample_from_fdata import irregular_sample
from skfda.preprocessing.dim_reduction import FPCA, PACE
from skfda.representation import FDataGrid, FDataIrregular
from skfda.typing._numpy import NDArrayInt

# %%
# In this example we are going to use functional principal component analysis
# through conditional expectation to explore datasets and obtain conclusions
# about said dataset using this technique.
#
# The PACE algorithm is an alternative to FPCA that is specifically designed
# for irregularly sampled data. It uses local linear smoothing to estimate
# the model components of the data, and then performs the conditional
# expectation step to obtain the principal components. These components are
# the directions that capture the main modes of variation across the function.
# PACE shares the same objectives as FPCA, it is a dimensionality
# reduction method for functional data that aims to reduce the complexity of
# studying observations by expressing the data in terms of a basis of K
# components that explain most of the variation in the data.
#
# The PACE algorithm was introduced in :footcite:ts`yao+muller+wang_2005_pace`,
# and in this example we will use the implementation of the algorithm that
# follows the same steps as the original algorithm.
#
dataset: Bunch = fetch_growth()
fd: FDataIrregular = dataset.data
assert isinstance(fd, FDataGrid), "Expected an FDataGrid object"

fd[:20].plot()
plt.show()

# print(fd.data_matrix.shape)
# print(fd.data_matrix[3])


# %%
random_state = 12

irregular_fd = irregular_sample(
    fd,
    8,
    random_state=random_state,
)
irregular_fd.coordinate_names = fd.coordinate_names
irregular_fd.argument_names = fd.argument_names
irregular_fd.dataset_name = fd.dataset_name

irregular_fd[:5].plot()
plt.show()


# %%
plt.figure()
for t, v in zip(irregular_fd.points, irregular_fd.values, strict=True):
    t_i = np.asarray(t)
    v_i = np.asarray(v)
    plt.scatter(t_i, v_i, alpha=0.7, s=10, color="black")

plt.xlabel("age")
plt.ylabel("height")
plt.title("Berkeley Growth Study", pad=20)
# plt.tight_layout()
plt.show()

# %%
n_components = 2
fpca = FPCA(n_components=n_components)
scores = fpca.fit_transform(fd)
fpca_rec = fpca.inverse_transform(scores)
fig, ax = plt.subplots()
for i, sample in enumerate(fpca.components_):
    label = fpca.components_.sample_names[i]
    sample.plot(axes=ax, label=label)

ax.set_ylim(-0.4, 0.6)
ax.set_xlabel(fpca.components_.argument_names[0] or "Domain")
ax.set_ylabel(fpca.components_.coordinate_names[0] or "Value")
plt.show()

print(fpca.explained_variance_ratio_)

# %%
# We can now apply the PACE method to the dataset.
pace = PACE(
    n_components=n_components,
    bandwidth_mean=np.array([0.1, 50]),
    # bandwidth_cov=np.array([0.1, 50]),
    bandwidth_cov=1.5,
    # n_grid_points=31,
)
pace.fit(irregular_fd)

fpc_scores = pace.transform(irregular_fd)

# %%
# Let's further understand the correlation of the data, by plotting the
# covariance function, which measures how the data varies together over time.
# First, we inspect the time point pairs by subject: although the data per
# subject is sparse, the assembled data fill the domain of the covariance
# surface quite densely.
t_mean = np.asarray(pace.mean_.grid_points[0])
pair_counts: defaultdict[tuple[int, int], int] = defaultdict(int)

for i, start in enumerate(irregular_fd.start_indices):
    end = (
        irregular_fd.start_indices[i + 1]
        if i + 1 < len(irregular_fd.start_indices)
        else len(irregular_fd.points)
    )
    t_i = np.asarray(irregular_fd.points[start:end])

    for s in t_i:
        for t in t_i:
            idx_s = np.argmin(np.abs(t_mean - s))
            idx_t = np.argmin(np.abs(t_mean - t))
            pair_counts[(int(idx_s), int(idx_t))] += 1

x, y, c = [], [], []

for (i, j), count in pair_counts.items():
    x.append(t_mean[j])
    y.append(t_mean[i])
    c.append(min(count, 5))

plt.figure()
scatter = plt.scatter(x, y, c=c, cmap="Blues", s=8, vmin=0, vmax=5)

cbar = plt.colorbar(scatter, label="Subjects")
cbar.set_ticks([0, 1, 2, 3, 4, 5])
cbar.set_ticklabels(["0", "1", "2", "3", "4", "5+"])

plt.xlabel("T_im")
plt.ylabel("T_il")
plt.title("Observed (r, s) Pairs by Subject Frequency")
plt.show()

# %%
# We can see that, before seroconversion, the data is very correlated, meaning
# that the subject's behaviour is similar. However, as time passes, the data
# becomes less correlated. This is because the patients undergo different
# treatments and have different responses to the disease. Another increase in
# the correlation is observed at the end of the time interval, which is due to
# the fact that patients reach a steady state (treatment) and the CD4 cell
# counts are similar.
#
# The early-late stages high correlation is likely related to the fact that
# patients who started with higher CD4 counts before seroconversion tend to
# also end with higher CD4 counts after 40 months — and those who started low,
# stay low. This indicates a strong individual-level persistence: patients
# maintain their relative immune status (i.e., high or low CD4) across the
# entire time range.
covariance_x, covariance_y = np.meshgrid(
    pace.t_covariance_,
    pace.t_covariance_,
    indexing="ij",
)
covariance = pace.covariance_.squeeze()

fig = plt.figure()
ax = fig.add_subplot(111, projection="3d")
ax.plot_surface(
    covariance_x,
    covariance_y,
    covariance,
    cmap="viridis",
    alpha=0.7,
)
ax.set_title("Smoothed Covariance Surface via PACE")
plt.tight_layout()
plt.show()

# %%
# To take the analysis further, we can now plot the first three components of
# the PACE method to develop our understanding of the data.
fig, ax = plt.subplots()
for i, sample in enumerate(pace.components_):
    label = pace.components_.sample_names[i]
    sample.plot(axes=ax, label=label)

ax.set_ylim(-0.4, 0.6)
ax.set_xlabel(pace.components_.argument_names[0] or "Domain")
ax.set_ylabel(pace.components_.coordinate_names[0] or "Value")
ax.legend()
plt.show()

# %%
# Analysing the components

fve_percent = pace.explained_variance_ratio_ * 100
fve_percent = np.insert(fve_percent, 0, 0)

n_components = np.arange(len(fve_percent))
max_components = 15
fve_percent = fve_percent[: max_components + 1]
n_components = n_components[: max_components + 1]

k = int(pace.n_components)
fve_k = fve_percent[k]

plt.figure(figsize=(6, 5))
plt.plot(n_components, fve_percent, "ro--", label="FVE curve")
plt.axvline(x=k, color="blue", linestyle="-", linewidth=1)
plt.scatter([k], [fve_k], color="black", zorder=5)

plt.text(k + 0.3, fve_k - 5, f"k = {k}, FVE = {fve_k:.3f}%", fontsize=8)
plt.title("Fraction of variance explained by No. of PC")
plt.xlabel("No. of Principal Components")
plt.ylabel("FVE (%)")
plt.ylim(0, 105)
plt.show()

# %%
# Lastly, we can also plot the reconstructed curves using the first three
# components to see how well they adjust to the original data. Selecting two
# arbitrary subjects, we will now plot the reconstructed curves against the
# original time points.
reconstructed = pace.inverse_transform(fpc_scores)

subject_indices = [6, 15]

t_mean = np.asarray(pace.mean_.grid_points[0])
mean_values = pace.mean_.data_matrix[0, :, 0]

fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True)
for ax, i in zip(axes, subject_indices, strict=True):
    reconstructed[i].plot(
        axes=ax, label="Reconstructed", color="C0", linestyle="-",
    )
    irregular_fd[i].scatter(
        axes=ax, label="Original (irregular)", color="red", marker="o",
    )

    t_all = np.asarray(fd.grid_points[0])
    v_all = fd.data_matrix[i, :, 0]

    t_obs = np.asarray(irregular_fd[i].points)

    mask_unobserved = ~np.isin(t_all, t_obs)
    t_unobserved = t_all[mask_unobserved]
    v_unobserved = v_all[mask_unobserved]

    # Plot unused points
    ax.scatter(
        t_unobserved,
        v_unobserved,
        color="salmon",
        marker=".",
    )

    ax.plot(t_mean, mean_values, linestyle="--", color="gray", label="Mean")

    ax.set_title(f"Subject {i}")
    ax.set_xlabel(irregular_fd.argument_names[0] or "Domain")
    ax.set_ylabel(irregular_fd.coordinate_names[0] or "Value")
    ax.legend()

plt.tight_layout()
plt.show()

# In these two cases, we can see that the first subject has a very high
# influence by the second component, whereas the second subject's trajectory is
# mostly influenced by the first component.

# %%
# We can finalise the analysis by plotting the reconstructed curves of the
# whole dataset, to showcase its similarity to the mean function, as well as
# highlight the new ways of analysis that this FDataGrid object allows us to
# perform.
reconstructed.plot()
plt.show()

grid1 = fd.grid_points[0]
grid2 = fpca_rec.grid_points[0]

common_grid = np.intersect1d(grid1, grid2)
idx1 = np.where(np.isin(grid1, common_grid))[0]
idx2 = np.where(np.isin(grid2, common_grid))[0]

aligned_true = fd.data_matrix[:, idx1]
aligned_pred = fpca_rec.data_matrix[:, idx2]

mse_per_curve = np.mean((aligned_true - aligned_pred) ** 2, axis=1)
average_mse = np.mean(mse_per_curve)

print(average_mse)


grid1 = fd.grid_points[0]
grid2 = reconstructed.grid_points[0]

common_grid = np.intersect1d(grid1, grid2)
idx1 = np.where(np.isin(grid1, common_grid))[0]
idx2 = np.where(np.isin(grid2, common_grid))[0]

aligned_true = fd.data_matrix[:, idx1]
aligned_pred = reconstructed.data_matrix[:, idx2]

mse_per_curve = np.mean((aligned_true - aligned_pred) ** 2, axis=1)
average_mse = np.mean(mse_per_curve)

print(average_mse)

# In conclusion, this analysis demonstrates how Functional Principal Component
# Analysis through Conditional Expectation (PACE) can be effectively applied to
# irregularly sampled longitudinal data, such as the CD4 dataset. By leveraging
# local smoothing and conditional expectation, PACE provides a principled way
# to estimate the mean, covariance surface, and principal components of sparse
# functional data.
#
# The results reveal meaningful biological patterns, such as the persistence of
# immune status across time and the existence of distinct trajectories of CD4
# decline among patients. The extracted components help disentangle population-
# level trends from subject-specific variations, and the reconstruction process
# confirms that just a few components can adequately capture the main structure
# in the data.
#
# Overall, PACE proves to be a powerful tool for exploratory analysis and
# dimensionality reduction in the context of sparse functional datasets. Its
# flexibility, interpretability, and ability to recover latent dynamics make it
# particularly suitable for longitudinal studies in biomedical research and
# beyond.

# %%
# References
# ----------
#
# .. footbibliography::
