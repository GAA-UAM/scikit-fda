"""
Clustering
==========

In this example, the use of the clustering methods is shown applied to the Canadian Weather dataset.
"""

# Author: Amanda Hernando Bernabé
# License: MIT

# sphinx_gallery_thumbnail_number = 6

from fda import datasets
from fda.clustering import *
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.patches as mpatches
from cycler import cycler
import matplotlib

_repr_svg:_ BytesIO
##################################################################################
# First, the Canadian Weather dataset is downloaded from the package 'fda' in CRAN.
# It contains a FDataGrid with daily temperatures and precipitations, that is, it
# has a 2-dimensional image. We are interested only in the daily average temperatures,
# so another FDataGrid is constructed with the desired values.

dataset = datasets.fetch_weather()
fd = dataset["data"]
fd_temperatures = FDataGrid(data_matrix=fd.data_matrix[:, :, 0], sample_points=fd.sample_points,
                            dataset_label=fd.dataset_label, axes_labels=fd.axes_labels[0:2])

# The desired FDataGrid only contains 10 random samples, so that the example provides
# clearer plots.
indices_samples = np.array([1, 3, 5, 10, 14, 17, 21, 25, 27, 30])
fd = fd_temperatures[indices_samples]

############################################################################################
# The data is plotted to show the curves we are working with. They are divided according to the
# target. In this case, it includes the different climates to which the weather stations belong to.

climate_by_sample = [dataset["target"][i] for i in indices_samples]

# Note that the samples chosen belong to three of the four possible target groups. By
# coincidence, these three groups correspond to indices 1, 2, 3, that is why the indices
# (´climate_by_sample´) are decremented in 1. In case of reproducing the example with other
# ´indices_samples´ and the four groups are not present in the sample, changes should be
# made in order ´indexer´ contains numbers in the interval [0, n_target_groups) and at
# least, an occurrence of each one.
indexer = np.asarray(climate_by_sample) - 1
indices_target_groups = np.unique(climate_by_sample)
climates = dataset["target_names"][indices_target_groups]

n_climates = len(climates)

# Assigning the color to each of the samples.
colormap = plt.cm.get_cmap('rainbow')
colors_by_climate = colormap(indexer / (n_climates - 1))
climate_colors = colormap(np.arange(n_climates) / (n_climates - 1))

# Plotting the legend
patches = []
for i in range(n_climates):
    patches.append(mpatches.Patch(color=climate_colors[i], label=climates[i]))

matplotlib.rcParams['axes.prop_cycle'] = cycler(color=colors_by_climate)
plt.figure()
fig, ax = fd.plot()
ax[0].legend(handles=patches)

############################################################################################
# The number of clusters is set with the number of climates, in order to see the performance
# of the clustering methods.

n_clusters = n_climates

############################################################################################
# If call the :func:`clustering method <fda.clustering.clustering>` with the data, a tuple
# with two arrays is returned. The first one contains the number of cluster each sample belongs
# to and the second one, the centroids of each cluster.

clustering_values, centers = clustering(fd, n_clusters)

print(clustering_values)
print(centers)

############################################################################################
# If call the :func:`fuzzy_clustering method <fda.clustering.fuzzy_clustering>` with the data,
# a tuple with two arrays is also returned. The first one contains ´n_clusters´ elements for
# each sample and dimension. They denote the degree of membership of each sample to each cluster.
# The second array contains the centroids of each cluster.

clustering_values, centers = fuzzy_clustering(fd, n_clusters)

print(clustering_values)
print(centers)

############################################################################################
# Other option includes the possibility of calling directly to the :func:`plot_clustering method
# <fda.clustering.plot_clustering>` which also returns the above information plus a plot showing
# to which cluster each sample belongs to.

#Customization of cluster colors and labels in order to match the first image of raw data.
cluster_colors = climate_colors[np.array([2, 0, 1])]
cluster_labels = climates[np.array([2, 0, 1])]

plt.figure()
fig, ax, labels, centers = plot_clustering(fd, n_clusters, cluster_colors=cluster_colors,
                                           cluster_labels=cluster_labels)

############################################################################################
# In the above method, the :func:`fuzzy_clustering method <fda.clustering.fuzzy_clustering>`
# can also be used. It assigns each sample to the cluster whose membership value is the
# greatest.

#Customization of cluster colors and labels in order to match the first image of raw data.
cluster_colors_fuzzy = climate_colors[np.array([1, 2, 0])]
cluster_labels_fuzzy = climates[np.array([1, 2, 0])]

plt.figure()
fig, ax, labels, centers = plot_clustering(fd, n_clusters, method=fuzzy_clustering,
                                           cluster_colors=cluster_colors_fuzzy,
                                           cluster_labels=cluster_labels_fuzzy)

############################################################################################
# Another plot implemented to show the results of the the :func:`fuzzy_clustering method
# <fda.clustering.fuzzy_clustering>` is the below one, which is similar to parallel coordinates.
# It is recommended to assign colors to each of the samples in order to identify them. In this
# example, the colors are the ones of the first plot, dividing the samples by climate.

plt.figure()
fig, ax, labels = plot_fuzzy_clustering_values(fd, n_clusters, cluster_labels=cluster_labels_fuzzy,
                                               sample_colors=colors_by_climate.reshape(fd.nsamples, fd.ndim_image, 4))

############################################################################################
# Lastly, the func:`plot_fuzzy_clustering_bars <fda.clustering.plot_fuzzy_clustering_bars>`
# method, returns a barplot. Each sample is designated with a bar which is filled proportionally
# to the membership values with the color of each cluster.

plt.figure()
fig, ax, labels = plot_fuzzy_clustering_bars(fd, n_clusters, cluster_colors=cluster_colors_fuzzy,
                                             cluster_labels=cluster_labels_fuzzy)

############################################################################################
# The possibility of sorting the bars according to a cluster is given specifying the number of
# cluster, which belongs to the interval [0, n_clusters).

plt.figure()
fig, ax, labels = plot_fuzzy_clustering_bars(fd, n_clusters, sort=0,
                                             cluster_colors=cluster_colors_fuzzy,
                                             cluster_labels=cluster_labels_fuzzy)

plt.figure()
fig, ax, labels = plot_fuzzy_clustering_bars(fd, n_clusters, sort=1,
                                             cluster_colors=cluster_colors_fuzzy,
                                             cluster_labels=cluster_labels_fuzzy)

plt.figure()
fig, ax, labels = plot_fuzzy_clustering_bars(fd, n_clusters, sort=2,
                                             cluster_colors=cluster_colors_fuzzy,
                                             cluster_labels=cluster_labels_fuzzy)
