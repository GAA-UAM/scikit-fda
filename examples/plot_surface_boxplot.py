"""
Surface Boxplot
====================

Shows the use of the surface boxplot, which is a generalization of the functional boxplot
for FDataGrid whose domain dimension is 2.
"""

# Author: Amanda Hernando Bernabé
# License: MIT

# sphinx_gallery_thumbnail_number = 3

import numpy as np
import matplotlib.pyplot as plt
from fda.grid import FDataGrid
from fda.boxplot import surface_boxplot, fdboxplot
from fda.datasets import make_sinusoidal_process, make_gaussian_process

##################################################################################
# In order to call the :func:`surface_boxplot method <fda.boxplot.surface_boxplot>`
# a functional data object with bidimensional domain must be generated. In this example,
# a FDataGrid representing a function :math:`f : \mathbb{R}^2\longmapsto\mathbb{R}^2` is
# constructed to show also the support of a multivariate dimensional image. The first
# dimension of the image contains sinusoidal processes and the second dimension,
# gaussian ones.
#
# First, the values are generated for each dimension with a function
# :math:`f : \mathbb{R}\longmapsto\mathbb{R}` implemented in the
# :func:`make_sinusoidal_process method <fda.datasets.make_sinusoidal_process>` and in the
# :func:`make_gaussian_process method <fda.datasets.make_gaussian_process>`, respectively.
# Those functions return FDataGrid objects whose 'data_matrix' store the values needed.

n_samples = 10
n_features = 10

fd1 = make_sinusoidal_process(n_samples = n_samples, n_features=n_features)
fd2 = make_gaussian_process(n_samples = n_samples, n_features=n_features)

##################################################################################
# After, those values generated for one dimension on the domain are propagated along
# another dimension, obtaining a three-dimensional matrix or cube (two-dimensional domain
# and one-dimensional image). This is done with both data matrices from the above FDataGrids.

cube1 = np.repeat(fd1.data_matrix, n_features).reshape(
    (n_samples, n_features, n_features))
cube2 = np.repeat(fd2.data_matrix, n_features).reshape(
    (n_samples, n_features, n_features))

##################################################################################
# Finally, both three-dimensional matrices are merged together and the FDataGrid desired
# is obtained. The data is plotted.

cube_2 = np.empty((n_samples, n_features, n_features, 2))
cube_2[:, :, :, 0] = cube1
cube_2[:, :, :, 1] = cube2

fd_2 = FDataGrid(data_matrix=cube_2, sample_points=np.tile(fd1.sample_points, (2,1)))

plt.figure()
fd_2.plot()

##################################################################################
# Since matplotlib was initially designed with only two-dimensional plotting in mind,
# the three-dimensional plotting utilities were built on top of matplotlib's two-dimensional
# display, and the result is a convenient (if somewhat limited) set of tools for
# three-dimensional data visualization as we can observe.
#
# For this reason, the profiles of the surfaces, which are contained in the first two
# generated functional data objects, are plot below, to help to visualize the data.

fig, ax = plt.subplots(1,2)
fd1.plot(ax=[ax[0]])
fd2.plot(ax=[ax[1]])

##################################################################################
# To terminate the example, the call to the target function of the example is made,
# showing the surface boxplot which corresponds to our FDataGrid representing a
# function :math:` f : \mathbb{R}^2\longmapsto\mathbb{R}^2` with a sinusoidal process in the
# first dimension of the image and a gaussian one in the second one.

plt.figure()
fdBoxplotInfo = surface_boxplot(fd_2)

##################################################################################
# It returns a :func:`FDataBoxplotInfo object <fda.boxplot.FDataBoxplotInfo>` from
# which the graph can be retrieved.
#
# The surface boxplot contains the median, the central envelope and the outlying envelope
# plotted from darker to lighter colors, although they can be customized.

plt.figure()
fdBoxplotInfo.plot()

##################################################################################
# Analogous to the procedure followed before of plotting the three-dimensional data
# and their correponding profiles, we can obtain also the functional boxplot for
# one-dimensional data with the :func:`fdboxplot function <fda.boxplot.fdboxplot>`
# passing as arguments the first two FdataGrid objects. The profile of the surface
# boxplot is obtained.

fig, ax = plt.subplots(1,2)
fdboxplot(fd1, ax=[ax[0]])
fdboxplot(fd2, ax=[ax[1]])