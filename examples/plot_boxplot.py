"""
Boxplot
=======

Shows the use of the functional Boxplot applied to the Canadian Weather dataset.
"""

# Author: Amanda Hernando Bernabé
# License: MIT

# sphinx_gallery_thumbnail_number = 2

from skfda import datasets
from skfda.grid import FDataGrid
from skfda.depth_measures import band_depth, fraiman_muniz_depth
import matplotlib.pyplot as plt
from skfda.exploratory.visualization.boxplot import Boxplot
import numpy

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
############################################################################################
# The data is plotted to show the curves we are working with. They are divided according to the
# target. In this case, it includes the different climates to which the weather stations belong to.

# Each climate is assigned a color. Defaults to grey.
colormap = plt.cm.get_cmap('seismic')
label_names = dataset["target_names"]
nlabels = len(label_names)
label_colors = colormap( numpy.arange(nlabels) / (nlabels - 1))

plt.figure()
fd_temperatures.plot(sample_labels=dataset["target"], label_colors=label_colors,
                     label_names=label_names)


############################################################################################
# We instantiate a :func:`functional boxplot object <skfda.boxplot.Boxplot>` with the data,
# and we call its :func:`plot function <skfda.boxplot.Boxplot.plot>` to show the graph.
#
# By default, only the part of the outlier curves which falls out of the central regions
# is plotted. We want the entire curve to be shown, that is why the show_full_outliers parameter is
# set to True.

fdBoxplot = Boxplot(fd_temperatures)
fdBoxplot.show_full_outliers = True

plt.figure()
fdBoxplot.plot()

############################################################################################
# We can observe in the boxplot the median in black, the central region (where the 50% of the
# most centered samples reside) in pink and the envelopes and vertical lines in blue. The
# outliers detected, those samples with at least a point outside the outlying envelope, are
# represented with a red dashed line. The colors can be customized.
#
# The outliers are shown below with respect to the other samples.

color = 0.3
outliercol = 0.7

plt.figure()
fd_temperatures.plot(sample_labels=fdBoxplot.outliers[0].astype(int),
                     label_colors=colormap([color, outliercol]),
                     label_names=["nonoutliers", "outliers"])

############################################################################################
# The curves pointed as outliers are are those curves with significantly lower values to the
# rest. This is the expected result due to the depth measure used, the :func:`modified band
# depth <skfda.boxplot.depth_measures.fraiman_muniz_depth>` which rank the samples according to
# their magnitude.
#
# The :func:`functional boxplot object <skfda.boxplot.Boxplot>` admits any :ref:`depth measure
# <depth-measures>` defined or customized by the user. Now the call is done with the
# :func:`band depth measure <skfda.boxplot.depth_measures.band_depth>` and the factor is reduced
# in order to designate some samples as outliers (otherwise, with this measure and the default
# factor, none of the curves are pointed out as outliers). We can see that the outliers detected
# belong to the Pacific and Arctic climates which are less common to find in Canada. As a
# consequence, this measure detects better shape outliers compared to the previous one.

fdBoxplot = Boxplot(fd_temperatures, method=band_depth, factor = 0.4)
fdBoxplot.show_full_outliers = True

plt.figure()
fdBoxplot.plot()

############################################################################################
# Another functionality implemented in this object is the enhanced functional boxplot,
# which can include other central regions, apart from the central or 50% one.
#
# In the following instantiation, the :func:`Fraiman and Muniz depth measure
# <skfda.boxplot.depth_measures.fraiman_muniz_depth>` is used and the 25% and 75% central regions
# are specified.

fdBoxplot = Boxplot(fd_temperatures,  method=fraiman_muniz_depth,
                    prob = [0.75, 0.5, 0.25])
plt.figure()
fdBoxplot.plot()

#############################################################################################
# The above two lines could be replaced just by fdBoxplot since the default representation of
# the :func:`boxplot object <skfda.boxplot.Boxplot>` is the image of the plot. However, due to
# generation of this notebook it does not show the image and that is why the plot method is
# called.
