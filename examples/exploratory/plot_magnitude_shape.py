"""
Magnitude-Shape Plot
====================

Shows the use of the MS-Plot applied to the Canadian Weather dataset.
"""

# Author: Amanda Hernando Bernabé
# License: MIT

# sphinx_gallery_thumbnail_number = 2

# sphinx_gallery_start_ignore
# ruff: noqa: PLR2004
# sphinx_gallery_end_ignore

# %%
# First, the Canadian Weather dataset is downloaded from the package 'fda' in
# CRAN. It contains a FDataGrid with daily temperatures and precipitations,
# that is, it has a 2-dimensional image. We are interested only in the daily
# average temperatures, so we extract the first coordinate.

from skfda.datasets import fetch_weather

X, y = fetch_weather(return_X_y=True, as_frame=True)
fd = X.iloc[:, 0].array
target = y.array
# sphinx_gallery_start_ignore
from pandas import Categorical

from skfda import FDataGrid

assert isinstance(fd, FDataGrid)
assert isinstance(target, Categorical)
# sphinx_gallery_end_ignore
fd_temperatures = fd.coordinates[0]

# %%
# The data is plotted to show the curves we are working with. They are divided
# according to the target. In this case, it includes the different climates to
# which the weather stations belong.

# Each climate is assigned a color. Defaults to grey.

import matplotlib.pyplot as plt
import numpy as np

group_names = target.categories

fd_temperatures.plot(
    group=target.codes,
    group_names=group_names,
)
plt.show()

# %%
# The MS-Plot is generated. In order to show the results, the
# :func:`~skfda.exploratory.visualization.MagnitudeShapePlot.plot` method
# is used. Note that the colors have been specified before to distinguish
# between outliers or not. In particular the tones of the default colormap,
# (which is 'seismic' and can be customized), are assigned.

from skfda.exploratory.depth.multivariate import SimplicialDepth
from skfda.exploratory.visualization import MagnitudeShapePlot

msplot = MagnitudeShapePlot(
    fd_temperatures,
    multivariate_depth=SimplicialDepth(),
)

msplot.plot()
plt.show()

# %%
# To show the utility of the plot, the curves are plotted according to the
# distinction made by the MS-Plot (outliers or not) with the same colors.

fd_temperatures.plot(
    group=msplot.outliers.astype(int),
    group_colors=["blue", "red"],
    group_names=["nonoutliers", "outliers"],
)
plt.show()

# %%
# We can observe that most of the curves  pointed as outliers belong either to
# the Pacific or Arctic climates which are not the common ones found in
# Canada. The Pacific temperatures are much smoother and the Arctic ones much
# lower, differing from the rest in shape and magnitude respectively.
#
# There are two curves from the Arctic climate which are not pointed as
# outliers but in the MS-Plot, they appear further left from the central
# points. This behaviour can be modified specifying the parameter alpha.
#
# Now we use the default multivariate depth from
# :func:`~skfda.exploratory.depth.IntegratedDepth` in the
# MS-Plot.

from skfda.exploratory.depth import IntegratedDepth

msplot = MagnitudeShapePlot(
    fd_temperatures,
    multivariate_depth=IntegratedDepth().multivariate_depth,
)

fig = msplot.plot()
plt.show()

# %%
# We can observe that almost none of the samples are pointed as outliers.
# Nevertheless, if we group them in three groups according to their position
# in the MS-Plot, the result is the expected one. Those samples at the left
# (larger deviation in the mean directional outlyingness) correspond to the
# Arctic climate, which has lower temperatures, and those on top (larger
# deviation in the directional outlyingness) to the Pacific one, which has
# smoother curves.

colors = np.array("C") + target.codes.astype(str)

ax = fig.axes[0]
xlim = ax.get_xlim()
ylim = ax.get_ylim()

fig, ax = plt.subplots()
ax.scatter(msplot.points[:, 0], msplot.points[:, 1], c=colors)
ax.set_title("MS-Plot")
ax.set_xlabel("magnitude outlyingness")
ax.set_xlim(*xlim)
ax.set_ylabel("shape outlyingness")
ax.set_ylim(*ylim)

plt.show()
