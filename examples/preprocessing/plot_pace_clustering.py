"""
PACE algorithm for cluster analysis
=======================================================================

Explores the possibility to apply clustering techniques to sparse,
irregularly sampled data using PACE.
"""

# Author: Alejandro Arias Gomez
# License: MIT

# sphinx_gallery_thumbnail_number = 8

# %%
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import linear_sum_assignment
from sklearn.metrics import confusion_matrix
from sklearn.utils import Bunch

from skfda.datasets._real_datasets import fetch_country_height
from skfda.exploratory.visualization.clustering import (
    ClusterPlot,
)
from skfda.ml.clustering import FuzzyCMeans
from skfda.preprocessing.dim_reduction import PACE
from skfda.representation import FDataIrregular

# %%
# In this example, we will use the Country Height dataset, which contains the
# average height of 144 countries grouped by decades, from 1700 to 2000.
# The dataset is irregularly sampled, meaning that not all countries have
# measurements for all decades.
#
# The dataset is available in the `skfda.datasets` module.
#
# Our goal is to analyse the relationship between the countries and their
# continent using clustering techniques. However, because the dataset is
# irregularly sampled, we will use the PACE algorithm to perform FPCA analysis
# and reconstruct the underlying curves for each country and convert the data
# into a regular grid before applying clustering, in order to be able to use
# the package's clustering algorithms.
#
# We will first load the dataset and plot each country's height curve, divided
# by continent.
country_height_bunch: Bunch = fetch_country_height()
country_height: FDataIrregular = country_height_bunch.data
assert isinstance(
    country_height, FDataIrregular
), "Expected an FDataIrregular object"

target = country_height_bunch.target

country_categories = country_height_bunch.target_names.categories
n_groups = len(country_categories)
cmap = plt.get_cmap("Set2")
country_colors = [cmap(i / (n_groups - 1)) for i in range(n_groups)]

country_height.plot(
    group=country_height_bunch.target,
    group_colors=country_colors,
    group_names=country_categories,
)
plt.show()


# %%
# To reinforce the irregularity of the data, we will plot the data points
# (years and heights) for all countries, where it can be seen that ever since
# the 1850s, the measurements are much more frequent.
plt.figure()
for t, v in zip(country_height.points, country_height.values, strict=True):
    t_i = np.asarray(t)
    v_i = np.asarray(v)
    plt.scatter(t_i, v_i, alpha=0.7, s=10, color="black")

plt.xlabel("year")
plt.ylabel("height_cm")
plt.title("Average male height by country")
plt.tight_layout()
plt.show()


# %%
# We can now apply the PACE method to the dataset.
pace = PACE(
    n_components=0.95,
    bandwidth_mean=np.array([12, 17]),
    bandwidth_cov=np.array([18, 22]),
)
pace.fit(country_height)

fpc_scores = pace.transform(country_height)

# %%
# Plotting the covariance surface of the data, we can see that, in the stages
# where the measurements are more sparse, the covariance is quite low,
# contrasting with the more recent years, where the covariance is much higher.
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
# We can further inspect the principal components of the data, where we can see
# that the first component takes the form of a steady decrease over the years,
# while the second component takes the shape of a slow increase, with peaks
# around 1750 and 1890. The third component is a more complex shape, with a
# decrease until the 1800s, only to spike until 1890 and decrease again.
# Combined, the last two components explain the more subtle variations in the
# data, while the first component explains the main trend.
fig, ax = plt.subplots()
for i, sample in enumerate(pace.components_):
    label = pace.components_.sample_names[i]
    sample.plot(axes=ax, label=label)

ax.set_xlabel(pace.components_.argument_names[0] or "Domain")
ax.set_ylabel(pace.components_.coordinate_names[0] or "Value")
ax.legend()
plt.show()

# %%
# From the FPC scores, we can reconstruct the whole dataset, allowing us to
# view the data in a regular grid, which is a necessary tool for clustering
# algorithms.
reconstructed = pace.inverse_transform(fpc_scores)

reconstructed.plot()
plt.show()

# %%
# Now we can apply clustering techniques available in the package. More
# specifically, we will use the Fuzzy C-Means algorithm, which is a
# probabilistic clustering algorithm that allows each data point to belong to
# multiple clusters with different degrees of membership.
n_clusters = n_groups
seed = 2

cluster: FuzzyCMeans = FuzzyCMeans(
    n_clusters=n_clusters,
    max_iter=200,
    random_state=seed,
    fuzzifier=1.1,
)
cluster.fit(reconstructed)
predicted = cluster.predict(reconstructed)

confusion = confusion_matrix(target, predicted)
row_ind, col_ind = linear_sum_assignment(-confusion)

permutation = np.zeros(confusion.shape[1], dtype=int)
permutation[col_ind] = row_ind

remapped_predicted = permutation[predicted]

matches = np.sum(remapped_predicted == target)
percentage = (matches / len(target)) * 100

print(f"Percentage of correct predictions: {percentage:.2f}%")


# %%
# As a result of the clustering algorithm, we can see that the percentage of
# correct predictions is rather low, with around 50% of the countries being
# correctly classified. Although this may be influenced by the fact that the
# data is sparsely and irregularly sampled, especially in the earlier
# decades, such a low percentage suggests that the average height by country
# is not a good indicator of the continent to which a country belongs.
#
# Further studies could be done about the relation between the height of a
# country's population and the geographical location they inhabit, but it may
# be necessary to restrict the areas of classification to a smaller subsets,
# given that the size of the continents is quite large and populations of
# diverse ethnogeographic origins coexist in the same continent.
#
# In addition, restricting the analysis to a smaller time frame may also
# improve the results.
#
# We will now plot the clustering results.
cluster_colors = [
    country_colors[permutation[i]] for i in range(len(permutation))
]
cluster_labels = [
    country_categories[permutation[i]] for i in range(len(permutation))
]

ClusterPlot(
    cluster,
    reconstructed,
    cluster_colors=cluster_colors,
    cluster_labels=cluster_labels,
).plot()
plt.show()

# %%
# References
# ----------
#
# .. footbibliography::
