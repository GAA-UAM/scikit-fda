"""
Functional Boxplot
====================

Shows the use of the functional Boxplot applied to the Canadian Weather dataset.
"""

# Author: Amanda Hernando Bernabé
# License: MIT

# sphinx_gallery_thumbnail_number = 4


from fda import datasets
from fda.grid import FDataGrid
from fda.boxplot import fdboxplot
from fda.depth_measures import band_depth, fraiman_muniz_depth
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

#Assigning the color to each of the samples according to the target.
colormap = plt.cm.get_cmap('seismic')
colors_by_climate = colormap(np.asarray(dataset["target"]) / (n_climates- 1))
climate_colors = colormap(np.arange(n_climates) / (n_climates- 1))

#Building the legend
patches = []
for i in range(n_climates):
    patches.append(mpatches.Patch(color=climate_colors[i], label=climates[i]))

#Plotting the curves and the legend with the desired colors.
matplotlib.rcParams['axes.prop_cycle'] = cycler(color=colors_by_climate)
plt.figure()
fig, ax = fd_temperatures.plot()
ax[0].legend(handles=patches)

############################################################################################
# We call the :func:`functional boxplot method <fda.boxplot.fdboxplot>` with the data.

plt.figure()
fdBoxploInfo = fdboxplot(fd_temperatures)

############################################################################################
# We can observe in the boxplot the median in black, the central region (where the 50% of the
# most centered samples reside) in pink and the envelopes and vertical lines in blue. The
# outliers detected, those samples with at least a point outside the outlying envelope, are
# represented with a red dashed line. The colors can be customized.
#
# The outliers are shown below with respect to the other samples..

color = 0.3
outliercol = 0.7

outliers = np.copy(fdBoxploInfo.outliers[0])
outliers[outliers == 0] = color
outliers[outliers == 1] = outliercol

colors = np.zeros((fd_temperatures.nsamples, 4))
colors = colormap(outliers)

plt.figure()
for i in range(fd_temperatures.nsamples):
    plt.plot(fd_temperatures.sample_points[0], fd_temperatures.data_matrix[i], c=colors[i])

plt.xlabel(fd_temperatures.axes_labels[0])
plt.ylabel(fd_temperatures.axes_labels[1])
plt.title(fd_temperatures.dataset_label)

############################################################################################
# The curves pointed as outliers are are those curves with significantly lower values to the
# rest. This is the expected result due to the depth measure used, the :func:`modified band
# depth <fda.boxplot.depth_measures.fraiman_muniz_depth>` which rank the samples according to
# their magnitude.
#
# We can retrieve the plot from the value returned by the method, a :func:`FDataBoxplotInfo
# object <fda.boxplot.FDataBoxplotInfo>`.

plt.figure()
fdBoxploInfo.plot()

############################################################################################
# The :func:`functional boxplot method <fda.boxplot.fdboxplot>` admits any :ref:`depth measure
# <depth-measures>` defined. Now the call is done with the :func:`band depth measure
# <fda.boxplot.depth_measures.band_depth>` and the factor is reduced in order to designate
# some samples as outliers (otherwise, with this measure and the default factor, none of the
# curves are pointed out as outliers). We can see that the outliers detected belong to the
# Pacific and Arctic climates which are less common to find in Canada. As a consequence, this
# measure detects better shape outliers compared to the previous one.

plt.figure()
fdBoxploInfo = fdboxplot(fd_temperatures, method=band_depth, factor = 0.4)

############################################################################################
# Another functionality implemented in this method is the enhanced functional boxplot,
# which can include other central regions, apart from the central or 50% one.
#
# In the following call, the :func:`Fraiman and Muniz depth measure
# <fda.boxplot.depth_measures.fraiman_muniz_depth>` is used and the 25% and 75% central regions
# are specified. Since the plot may be a little overloaded, only the part of the outlier curves
# which falls out of the central regions is plotted (fullout parameter).

plt.figure()
fdBoxploInfo = fdboxplot(fd_temperatures,  method=fraiman_muniz_depth,
                         prob = [0.75, 0.5, 0.25], fullout = True)