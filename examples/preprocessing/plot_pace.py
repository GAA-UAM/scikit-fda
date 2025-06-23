"""
Functional Principal Component Analysis through Conditional Expectation
=======================================================================

Explores an alternative way to do functional principal component analysis for
irregularly sampled data.
"""

# Author: Alejandro Arias Gomez
# License: MIT

# sphinx_gallery_thumbnail_number = 6

# %%
from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np
from sklearn.utils import Bunch

from skfda.datasets._real_datasets import fetch_growth
from skfda.datasets._sample_from_fdata import irregular_sample
from skfda.preprocessing.dim_reduction import FPCA, PACE
from skfda.representation import FDataGrid

# %%
# In this example we are going to use functional principal component analysis
# through conditional expectation to explore the Berkeley Growth Study and
# obtain conclusions about said dataset using this technique, comparing the
# results to those obtained with the FPCA method.
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
# The Berkeley Growth Study dataset consists of height measurements over time
# for a cohort of children. It is a classic dataset in Functional Data
# Analysis, and here it is initially provided in a dense, regular format
# (FDataGrid).
dataset: Bunch = fetch_growth()
fd: FDataGrid = dataset.data
assert isinstance(fd, FDataGrid), "Expected an FDataGrid object"

fd[:20].plot()
plt.show()

# %%
# To evaluate the performance of PACE, we transform the dense dataset into an
# irregular one. Each subject is now sampled at 8 randomly selected time
# points. This mirrors the sparse and irregular setting often found in
# real-world longitudinal data.
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
# We continue by plotting all raw (t_i, X_i) points from the irregularly
# sampled dataset. This gives insight into the data sparsity and non-uniform
# time coverage across individuals.
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
# We apply classical FPCA (intended for regularly sampled data) to the original
# dense dataset. This gives us a reference point to later compare with PACE.
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

fpca_rec.plot()
plt.show()
print(fpca.explained_variance_ratio_)

# %%
# We now apply the PACE algorithm to the irregularly sampled dataset. The
# bandwidth parameters for mean and covariance estimation are specified
# manually.
pace = PACE(
    n_components=n_components,
    bandwidth_mean=np.array([0.1, 50]),
    bandwidth_cov=1.5,
)
pace.fit(irregular_fd)

fpc_scores = pace.transform(irregular_fd)

# %%
# Let's further understand the correlation of the data, by plotting the
# covariance function, which measures how the data varies together over time.
# First, we inspect the time point pairs by subject: although the data per
# subject is sparse, the assembled data fill the domain of the covariance
# surface quite densely. This visualization is relevant to justify the
# effectiveness of the kernel smoother.
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
# The kernel-smoothed covariance surface Ĝ(t, s) is visualized here. It
# captures how height values co-vary over time, averaged over the population.
# The surface portrays how the correlation of the measurements increases over
# time, verifying the theoretical results where we are aware of the disparity
# in growth curves at adolescence, only to reach a more stable increase
# towards the age of 16-17.
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
# To take the analysis further, we can now plot the first two components of
# the PACE method to develop our understanding of the data. What stands out
# is the strong similarity with the components extracted from the FPCA method,
# taking into account that only 8/31 time points are kept from each subject.
# Variations in their trajectories appear more smoothed due to the effect of
# the kernel smoothers used for the mean and covariance surface.
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
# Another graph of interest is the fraction of variance explained by number of
# principal components, which can help decide how many components to maintain
# for further stages of analysis. For this example, we draw the line in >95%
# and, because of this, we use the first two principal components.
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

print(pace.explained_variance_ratio_[:3])

# %%
# Lastly, we can also plot the reconstructed curves using the first two
# components to see how well they adjust to the original data. Selecting two
# arbitrary subjects, we will now plot the reconstructed curves against the
# original time points. Plotted in red with bigger circles are the points used
# for the analysis, and in a smaller orange form are the discarded points, used
# in the case of FPCA analysis.
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

# In these two cases, we can see that the reconstructions are remarkably
# accurate. However, in subject 6 we see that, because no points from the last
# stages are included, the reconstruction does not capture the slight increase.
# This stresses the importance of utilising a densely populated dataset for
# more accurate results.

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

# %%
# In conclusion, this analysis demonstrates how Functional Principal Component
# Analysis through Conditional Expectation (PACE) can be effectively applied to
# irregularly sampled longitudinal data. By leveraging local smoothing and
# conditional expectation, PACE provides a principled way to estimate the mean,
# covariance surface, and principal components of sparse functional data.
#
# The extracted components help disentangle population-level trends from
# subject-specific variations, and the reconstruction process confirms that a
# small number of components suffice to adequately capture the main structure
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
