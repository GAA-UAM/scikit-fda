"""
Function composition
====================

This example shows the composition of multidimensional FDataGrids.
"""

# Author: Pablo Marcos Manchón
# License: MIT

# sphinx_gallery_thumbnail_number = 3

import fda
import matplotlib.pyplot as plt
import numpy as np

from mpl_toolkits.mplot3d import axes3d




###############################################################################
# Function composition can be applied to our data once is in functional
# form using the method :func:`compose`.
#
# Let :math:`f: X \rightarrow Y` and :math:`g: Y \rightarrow Z`, the composition
# will produce a third function :math:`g \circ f: X \rightarrow Z` which maps
# :math:`x \in X` to :math:`g(f(x))` [1].
#
# In `Landmark Registration <plot_landmark_registration.html>`_ it is shown the
# simplest case, where it is used to apply a transformation of the time scale of
# unidimensional data to register its features.
#
# The following example shows the basic usage applied to a surface and a curve,
# although the method will work for data with arbitrary dimensions to.
#
# Firstly we will create a data object containing a surface
# :math:`g: \mathbb{R}^2 \rightarrow \mathbb{R}`.
#

# Constructs example surface
X, Y, Z = axes3d.get_test_data(1.2)
data_matrix = [Z.T]
sample_points = [X[0,:], Y[:, 0]]

g = fda.FDataGrid(data_matrix, sample_points)

# Sets cubic interpolation
g.interpolator = fda.grid.GridSplineInterpolator(interpolation_order=3)

# Plots the surface
g.plot()

###############################################################################
# We will create a parametric curve :math:`f(t)=(10 \, \cos(t), 10 \, sin(t))`.
# The result of the composition, :math:`g \circ f:\mathbb{R} \rightarrow
# \mathbb{R}`
# will be another functional object with the values of :math:`g` along the path
# given by :math:`f`.
#

# Creation of circunference in parametric form
t = np.linspace(0, 2*np.pi, 100)

data_matrix = [10 * np.array([np.cos(t), np.sin(t)]).T]
f = fda.FDataGrid(data_matrix, t)

# Composition of function
gof = g.compose(f)

plt.figure()

gof.plot()

###############################################################################
# In the following chart it is plotted the curve
# :math:`(10 \, \cos(t), 10 \, sin(t), g \circ f (t))` and the surface.
#

# Plots surface
fig, ax = g.plot(alpha=.8)

# Plots path along the surface
path = f(t)[0]
ax[0].plot(path[:,0], path[:,1], gof(t)[0], color="orange")

plt.show()
###############################################################################
# [1] Function composition `https://en.wikipedia.org/wiki/Function_composition
# <https://en.wikipedia.org/wiki/Function_composition>`_.
#
