"""
Magnitude-Shape Plot
===================================

Shows the use of the MS-Plot applied to the Canadian Weather dataset.
"""

# Author: Amanda Hernando Bernabé <amanda.hernando@estudiante.uam.es>
# License: MIT

# sphinx_gallery_thumbnail_number = 1

from fda import datasets
from fda.magnitude_shape_plot import *


###############################################################################
# First, the Canadian Weather dataset is downloaded from the package 'fda' in CRAN.
# We are interested only in the daily average temperatures to construct the grid.

dataset = datasets.fetch_cran("CanadianWeather", "fda")
fd = FDataGrid(np.asarray(dataset["CanadianWeather"]['dailyAv'][:, :, 0]).T,
               dataset_label="Canadian Weather",
               axes_labels=["Day", "Temperature (ºC)"])
###############################################################################
# The FDataGrid is plotted to show the data in a visual way.
fd.plot()

###############################################################################
# The FDataGrid class is used for datasets containing discretized functions
# that are measured at the same points.

#outliers = magnitude_shape_plot(fd)
plt.show()


