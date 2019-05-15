"""
Clustering
==========

In this example, the use of the clustering methods is shown applied to the
Canadian Weather dataset.
"""

# Author: Amanda Hernando Bernabé
# License: MIT

# sphinx_gallery_thumbnail_number = 6

from skfda import datasets
from skfda.representation.grid import FDataGrid
from skfda.exploratory.visualization.clustering_plots import *

##################################################################################
# First, the Canadian Weather dataset is downloaded from the package 'fda' in CRAN.
# It contains a FDataGrid with daily temperatures and precipitations, that is, it
# has a 2-dimensional image. We are interested only in the daily average temperatures,
# so another FDataGrid is constructed with the desired values.

dataset = datasets.fetch_weather()
fd = dataset["data"]
fd_temperatures = FDataGrid(data_matrix=fd.data_matrix[:, :, 0],
                            sample_points=fd.sample_points,
                            dataset_label=fd.dataset_label,
                            axes_labels=fd.axes_labels[0:2])

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

# Assigning the color to each of the groups.
colormap = plt.cm.get_cmap('tab20b')
n_climates = len(climates)
climate_colors = colormap(np.arange(n_climates) / (n_climates - 1))

plt.figure()
fd.plot(sample_labels=indexer, label_colors=climate_colors, label_names=climates)

############################################################################################
# The number of clusters is set with the number of climates, in order to see the performance
# of the clustering methods and the seed is set to one in order to obatain always the same
# result for the example.

n_clusters = n_climates
seed = 2

############################################################################################
# First, the class :class:`K-Means <fskfda.ml.clustering.base_kmeans.KMeans>`
# is instantiated. Its :func:`fit method <skfda.ml.clustering.base_kmeans.KMeans.fit>`
# is called with the desired. data,
# resulting in the calculation of the clustering values, the number of cluster
# each sample belongs to, and the centroids of each cluster.

kmeans = KMeans(n_clusters=n_clusters, random_state=seed)
kmeans.fit(fd)
print(kmeans.predict(fd))
print(kmeans.cluster_centers_)

############################################################################################
# To see the information in a graphic way, the :func:`plot method
# <fda.clustering.KMeans.plot>` can be used.

# Customization of cluster colors and labels in order to match the first image
# of raw data.
cluster_colors = climate_colors[np.array([1, 2, 0])]
cluster_labels = climates[np.array([1, 2, 0])]

plot_clusters(kmeans, fd, cluster_colors=cluster_colors,
              cluster_labels=cluster_labels)

############################################################################################
# Other clustering algorithm implemented is the Fuzzy K-Means found in the
# class :class:`FuzzyKMeans <skfda.ml.clustering.base_kmeans.FuzzyKMeans>`. Following the above
# procedure, an object of this type is instantiated and then, the
# :func:`fit method <skfda.ml.clustering.base_kmeans.KMeans.fit>` is called with the desired. data.
# Internally, the membership_values are calculated, which contains ´n_clusters´
# elements for each sample and dimension, denoting the degree of membership of
# each sample to each cluster. Also, the centroids of each cluster are obtained.

fuzzy_kmeans = FuzzyKMeans(n_clusters=n_clusters, random_state=seed)
fuzzy_kmeans.fit(fd)
print(fuzzy_kmeans.predict(fd))
print(fuzzy_kmeans.cluster_centers_)

############################################################################################
# To see the information in a graphic way, the :func:`plot method
# <fda.clustering.FuzzyKMeans.plot>` can be used. It assigns each sample to the
# cluster whose membership value is the greatest.

plot_clusters(fuzzy_kmeans, fd, cluster_colors=cluster_colors,
              cluster_labels=cluster_labels)

############################################################################################
# Another plot implemented to show the results in the class :class:`Fuzzy K-Means
# <skfda.ml.clustering.base_kmeans.FuzzyKMeans>` is the below one, which is similar to parallel coordinates.
# It is recommended to assign colors to each of the samples in order to identify them. In this
# example, the colors are the ones of the first plot, dividing the samples by climate.

colors_by_climate = colormap(indexer / (n_climates - 1))

plt.figure()
plot_cluster_lines(fuzzy_kmeans, fd, cluster_labels=cluster_labels,
                   sample_colors=colors_by_climate.reshape(fd.nsamples,
                                                          fd.ndim_image, 4))

############################################################################################
# Lastly, the function :func:`plot_bars <fda.clustering.FuzzyKMeans.plot_bars>`
# found in the :class:`Fuzzy K-Means <skfda.ml.clustering.base_kmeans.FuzzyKMeans>` class,
# returns a barplot. Each sample is designated with a bar which is filled proportionally
# to the membership values with the color of each cluster.

plt.figure()
plot_cluster_bars(fuzzy_kmeans, fd, cluster_colors=cluster_colors,
                       cluster_labels=cluster_labels)

############################################################################################
# The possibility of sorting the bars according to a cluster is given specifying the number of
# cluster, which belongs to the interval [0, n_clusters).

plt.figure()
plot_cluster_bars(fuzzy_kmeans, fd, sort=0, cluster_colors=cluster_colors,
                       cluster_labels=cluster_labels)

plt.figure()
plot_cluster_bars(fuzzy_kmeans, fd, sort=1, cluster_colors=cluster_colors,
                       cluster_labels=cluster_labels)

plt.figure()
plot_cluster_bars(fuzzy_kmeans, fd, sort=2, cluster_colors=cluster_colors,
                       cluster_labels=cluster_labels)
