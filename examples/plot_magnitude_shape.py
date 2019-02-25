"""
Magnitude-Shape Plot
====================

Shows the use of the MS-Plot applied to the Canadian Weather dataset.
"""

# Author: Amanda Hernando Bernabé
# License: MIT

# sphinx_gallery_thumbnail_number = 2

from fda import datasets
from fda.grid import FDataGrid
from fda.depth_measures import fraiman_muniz_depth
from fda.magnitude_shape_plot import magnitude_shape_plot
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.patches as mpatches
from cycler import cycler
import matplotlib

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

climates = dataset["target_names"]
n_climates = len(climates)
# indexer, uniques = pd.factorize(regions, sort=True)

#Assigning the color to each of the samples.
colormap = plt.cm.get_cmap('seismic')
colors_by_climate = colormap(np.asarray(dataset["target"]) / (n_climates- 1))
climate_colors = colormap(np.arange(n_climates) / (n_climates- 1))

#Building the legend
patches = []
for i in range(n_climates):
    patches.append(mpatches.Patch(color=climate_colors[i], label=climates[i]))

#Plotting the curves.
matplotlib.rcParams['axes.prop_cycle'] = cycler(color=colors_by_climate)
plt.figure()
fig, ax = fd_temperatures.plot()
ax[0].legend(handles=patches)

#############################################################################################
# The MS-Plot is generated specifying the tones of the colors defined in the default colormap.
# The function returns the points plotted and an array indicating which ones are outliers.

color = 0.3
outliercol = 0.7
plt.figure()
points, outliers = magnitude_shape_plot(fd_temperatures, color = color, outliercol = outliercol)

############################################################################################
# To show the utility of the plot, the curves are plotted according to the distinction
# made by the MS-Plot (outliers or not) with the same colors.

outliers[outliers == 0] = color
outliers[outliers == 1] = outliercol

colors = colormap(outliers)
matplotlib.rcParams['axes.prop_cycle'] = cycler(color=colors)

plt.figure()
fd_temperatures.plot()

#######################################################################################
# We can observe that most of the curves  pointed as outliers belong either to the Pacific or
# Arctic climates which are not the common ones found in Canada. The Pacific temperatures
# are much smoother and the Arctic ones much lower, differing from the rest in shape and
# magnitude respectively.
#
# There are two curves from the Arctic climate which are not pointed as
# outliers but in the MS-Plot, they appear further left from the central points. This behaviour
# can be modified specifying the parameter alpha.
#
# Now we use the :func:`Fraiman and Muniz depth measure <fda.depth_measures.fraiman_muniz_depth>`
# in the MS-Plot.

plt.figure()
points, outliers = magnitude_shape_plot(fd_temperatures, depth_method = fraiman_muniz_depth,
                                        color = color, outliercol = outliercol)
#######################################################################################
# We can observe that none of the samples are pointed as outliers. Nevertheless, if we group them
# in three groups according to their position in the MS-Plot, the result is the expected one.
# Those samples at the left (larger deviation in the mean directional outlyingness) correspond
# to the Arctic climate, which has lower temperatures, and those on top (larger deviation in the
# directional outlyingness) to the Pacific one, which has smoother curves.


outliers[:] = color
outliers[np.where(points[:, 0] < -0.6)] = outliercol
outliers[np.where(points[:, 1] > 0.12)] = 0.9
colors_groups = colormap(outliers)

plt.figure()
plt.scatter(points[:, 0], points[:, 1], c=colors_groups)
fd_temperatures.set_labels(fig)

matplotlib.rcParams['axes.prop_cycle'] = cycler(color=colors_groups)
plt.figure()
fd_temperatures.plot()