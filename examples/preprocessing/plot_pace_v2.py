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
from skfda.datasets._real_datasets import fetch_cd4
from skfda.preprocessing.dim_reduction import PACE
from skfda.representation import FDataIrregular

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
# We will analyse the CD4 dataset. This dataset contains the CD4 cell counts of
# 349 HIV patients measured in between months 0 and 42 since seroconversion.
# To better understand the data, we will plot the first 20 subjects. Each
# subject contains a different number of measurements (less than 12), and the
# time points are different for each subject.
cd4_bunch: Bunch = fetch_cd4()
cd4: FDataIrregular = cd4_bunch.data
assert isinstance(cd4, FDataIrregular), "Expected an FDataIrregular object"

cd4[:20].plot()
plt.show()

# %%
# Continuing with the analysis of the dataset, we will plot the total data
# across all subjects, where we can further identify the sparsity of the data.
# The data is spread across all the domain, with more frequent measurements
# every trimester, especially the first one before and after seroconversion.
plt.figure()
for t, v in zip(cd4.points, cd4.values, strict=True):
    t_i = np.asarray(t)
    v_i = np.asarray(v)
    plt.scatter(t_i, v_i, alpha=0.7, s=10, color="black")

plt.xlabel("months since seroconversion")
plt.ylabel("CD4 cell count")
plt.title("All observed CD4 values across subjects")
plt.tight_layout()
plt.show()

# %%
# We can now apply the PACE method to the dataset.
pace = PACE(
    n_components=3,
    bandwidth_mean=np.array([0.1, 50.0]),
    bandwidth_cov=np.array([0.1, 50.0]),
    # boundary_effect_interval=(0.0, 0.95),
)
pace.fit(cd4)

fpc_scores = pace.transform(cd4)

# %%
# Let's further understand the correlation of the data, by plotting the
# covariance function, which measures how the data varies together over time.
# First, we inspect the time point pairs by subject: although the data per
# subject is sparse, the assembled data fill the domain of the covariance
# surface quite densely.
t_mean = np.asarray(pace.mean_.grid_points[0])
pair_counts: defaultdict[tuple[int, int], int] = defaultdict(int)

for i, start in enumerate(cd4.start_indices):
    end = (
        cd4.start_indices[i + 1]
        if i + 1 < len(cd4.start_indices)
        else len(cd4.points)
    )
    t_i = np.asarray(cd4.points[start:end])

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
# We can see that, as time passes, the data becomes more correlated. This is
# because the patients undergo different treatments and have different
# responses to the disease but, over time, patients reach a steady state
# (treatment) the CD4 cell count normalises.
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

ax.set_xlabel(pace.components_.argument_names[0] or "Domain")
ax.set_ylabel(pace.components_.coordinate_names[0] or "Value")
ax.legend()
plt.show()

print(pace.explained_variance_ratio[:3])

# %%
# Analysing the components, we see that the first component is the one that
# explains the most variance in the data (96.364%), and it is a smooth
# horizontal curve that indicates that the curves have a similar structure to
# the mean of the data.
#
# The second component captures variations in the rate of
# change of CD4 counts over time. Specifically, it reflects differences in how
# rapidly patients' CD4 counts decline or recover post-seroconversion. For
# instance, a positive score on this component may indicate a patient whose CD4
# count decreases more slowly, while a negative score may correspond to a more
# rapid decline.
#
# The third component accounts for more localized and transient deviations in
# CD4 counts. It highlights short-term fluctuations—such as brief increases or
# dips—particularly near the time of seroconversion. These may reflect
# individual responses to treatment, transient infections, or other
# patient-specific factors. Because this component explains a relatively small
# portion of the total variance, such patterns are less consistent across the
# broader population.
#
# Overall, these components reflect progressively more localized and less
# dominant variations in the data. The first component mostly models the
# general progression of CD4 cell count, while the second and third provide
# subject-specific refinements. This analysis further supports the conclusions
# drawn from the covariance surface.
#
# Additionally, we can also obtain the graph that shows the fraction of
# variance explained by the number of components. This graph is useful to
# determine whether the number of components to use in the analysis is
# appropriate.

fve_percent = pace.explained_variance_ratio * 100
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

subject_indices = [107, 236]

t_mean = np.asarray(pace.mean_.grid_points[0])
mean_values = pace.mean_.data_matrix[0, :, 0]

fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True)
for ax, i in zip(axes, subject_indices, strict=True):
    reconstructed[i].plot(
        axes=ax, label="Reconstructed", color="C0", linestyle="-",
    )
    cd4[i].scatter(
        axes=ax, label="Original (irregular)", color="red", marker="o"
    )
    ax.plot(t_mean, mean_values, linestyle="--", color="gray", label="Mean")

    ax.set_title(f"Subject {i}")
    ax.set_xlabel(cd4.argument_names[0] or "Domain")
    ax.set_ylabel(cd4.coordinate_names[0] or "Value")
    ax.legend()

plt.tight_layout()
plt.show()

# In these two cases, we can see that the first subject has a higher
# influence by the second component, whereas the second subject's trajectory is
# mostly influenced by the first component.

# %%
# We can finalise the analysis by plotting the reconstructed curves of the
# whole dataset, to showcase its similarity to the mean function, as well as
# highlight the new ways of analysis that this FDataGrid object allows us to
# perform.
reconstructed.plot()
plt.show()

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
