"""
Functional Boxplot
====================

Shows the use of the functional boxplot.
"""

# Author: Amanda Hernando Bernabé <amanda.hernando@estudiante.uam.es>
# License: MIT

# sphinx_gallery_thumbnail_number = 2

from fda import datasets
from fda.grid import FDataGrid
from fda.magnitude_shape_plot import magnitude_shape_plot
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import matplotlib.patches as mpatches

##################################################################################
# First, the Canadian Weather dataset is downloaded from the package 'fda' in CRAN.
# It contains a FDataGrid with daily temperatures and precipitations, that is, it
# has a 2-dimensional image. We are interested only in the daily average temperatures,
# so another FDataGrid is constructed with the desired values.

dataset = datasets.fetch_weather()
fd = dataset["data"]
fd_temperatures = FDataGrid(data_matrix=fd.data_matrix[:, :, 0], sample_points=fd.sample_points,
                            dataset_label=fd.dataset_label, axes_labels=fd.axes_labels[0:2])

############################################################################################
# The data is plotted to show the curves we are working with. They are divided according to the
# target. In this case, it includes the different climates to which the weather stations belong to.

regions = dataset["target"]
indexer, uniques = pd.factorize(regions, sort=True)

#Assigning the color to each of the samples.
colormap = plt.cm.get_cmap('seismic')
colors = colormap(indexer / (len(uniques) - 1))

#Plotting the curves.
plt.figure()
for i in range(fd_temperatures.nsamples):
    plt.plot(fd_temperatures.sample_points[0], fd_temperatures.data_matrix[i], c=colors[i])

#Plotting the legend
indices = range(len(uniques))
d = dict(zip(uniques, indices))
patches = []
for label, index in d.items():
    patches.append(mpatches.Patch(color=colormap(index / (len(uniques) - 1)), label=label))
plt.legend(handles=patches)

#############################################################################################
# The MS-Plot is generated specifying the tones of the colors defined in the default colormap.
# The function returns the points plotted and an array indicating which ones are outliers.

color = 0.3
outliercol = 0.7
plt.figure()
points, outliers = magnitude_shape_plot(fd_temperatures, alpha = 0.99, color = color,
                                        outliercol = outliercol)

############################################################################################
# To show the utility of the plot, the curves are plotted according to the distinction
# made by the MS-Plot (outliers or not) with the same colors.

outliers[outliers == 0] = color
outliers[outliers == 1] = outliercol

colors = np.zeros((fd_temperatures.nsamples, 4))
colors = colormap(outliers)

plt.figure()
for i in range(fd_temperatures.nsamples):
    plt.plot(fd_temperatures.sample_points[0], fd_temperatures.data_matrix[i], c=colors[i])


#######################################################################################
# We can observe that the curves belonging to the Pacific and Arctic climates are
# the ones which are pointed as outliers since they are a minority. The Pacific temperatures
# are much smoother and the Arctic ones much lower, differing from the rest in shape and
# magnitude respectively.



