"""
Function composition
====================

This example shows the composition of multidimensional FDataGrids.
"""

# Author: Pablo Marcos Manchón
# License: MIT

# sphinx_gallery_thumbnail_number = 3

# %%
# Function composition can be applied to our data once is in functional
# form using the method :func:`~skfda.representation.FData.compose`.
#
# Let :math:`f: X \rightarrow Y` and :math:`g: Y \rightarrow Z`, the
# composition will produce a third function :math:`g \circ f: X \rightarrow Z`
# which maps :math:`x \in X` to :math:`g(f(x))` [1].
#
# In :ref:`sphx_glr_auto_examples_plot_landmark_registration.py` it is shown
# the simplest case, where it is used to apply a transformation of the time
# scale of unidimensional data to register its features.
#
# The following example shows the basic usage applied to a surface and a
# curve, although the method will work for data with arbitrary dimensions to.
#
# Firstly we will create a data object containing a surface
# :math:`g: \mathbb{R}^2 \rightarrow \mathbb{R}`.
#
# Constructs example surface
import numpy as np
from mpl_toolkits.mplot3d import axes3d

import skfda

X, Y, Z = axes3d.get_test_data(1.2)
data_matrix = [Z.T]
grid_points = [X[0, :], Y[:, 0]]

g = skfda.FDataGrid(data_matrix, grid_points)

# Plots the surface
g.plot()

# %%
# We will create a parametric curve
# :math:`f(t)=(10 \, \cos(t), 10 \, sin(t))`. The result of the composition,
# :math:`g \circ f:\mathbb{R} \rightarrow \mathbb{R}` will be another
# functional object with the values of :math:`g` along the path given by
# :math:`f`.


# Creation of circunference in parametric form
t = np.linspace(0, 2 * np.pi, 100)

data_matrix = [10 * np.array([np.cos(t), np.sin(t)]).T]
f = skfda.FDataGrid(data_matrix, t)

# Composition of function
gof = g.compose(f)

gof.plot()

# %%
# In the following chart it is plotted the curve
# :math:`(10 \, \cos(t), 10 \, sin(t), g \circ f (t))` and the surface.

# Plots surface
fig = g.plot(alpha=0.8)

# Plots path along the surface
path = f(t)[0]
fig.axes[0].plot(path[:, 0], path[:, 1], gof(t)[0, ..., 0], color="orange")

fig

# %%
# [1] Function composition `https://en.wikipedia.org/wiki/Function_composition
# <https://en.wikipedia.org/wiki/Function_composition>`_.
