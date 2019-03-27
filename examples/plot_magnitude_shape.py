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

# Each climate is assigned a color. Defaults to grey.
colormap = plt.cm.get_cmap('seismic')
label_names = dataset["target_names"]
nlabels = len(label_names)
label_colors = colormap( np.arange(nlabels) / (nlabels - 1))

plt.figure()
fd_temperatures.plot(sample_labels=dataset["target"], label_colors=label_colors,
                     label_names=label_names)

#############################################################################################
# The MS-Plot is generated specifying the tones of the colors defined in the default colormap.
# The function returns the points plotted and an array indicating which ones are outliers.

color = 0.3
outliercol = 0.7

plt.figure()
points, outliers = magnitude_shape_plot(fd_temperatures, color = color,
                                        outliercol = outliercol)

############################################################################################
# To show the utility of the plot, the curves are plotted according to the distinction
# made by the MS-Plot (outliers or not) with the same colors.

colormap = plt.cm.get_cmap('seismic')

plt.figure()
fd_temperatures.plot(sample_labels=outliers.astype(int),
                     label_colors=colormap([color, outliercol]))

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

group1 = np.where(points[:, 0] < -0.6)
group2 = np.where(points[:, 1] > 0.12)

colors = np.copy(outliers)
colors[:] = color
colors[group1] = outliercol
colors[group2] = 0.9

plt.figure()
plt.scatter(points[:, 0], points[:, 1], c=colormap(colors))

labels = np.copy(outliers).astype(int)
labels[group1] = 1
labels[group2] = 2

plt.figure()
fd_temperatures.plot(sample_labels=labels,
                     label_colors=colormap([color, outliercol, 0.9]))
