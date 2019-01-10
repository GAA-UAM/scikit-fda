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


#TODO: Delete auxiliar plot functions after merge the graphics branch
def plot_aux(fd, derivative=0, ax=None, **kwargs):
    """Temporal function. Plots using interpolation"""

    if ax is None:
        ax = plt.gca()

    t = np.linspace(*fd.domain_range[0], 200)
    plt.plot(t, fd(t, derivative=derivative).T, **kwargs)

def plot_3d(fd, derivative=0):
    """Temporal function. Plots a surface."""
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    t = np.linspace(*fd.domain_range[0], 30)
    s = np.linspace(*fd.domain_range[1], 30)

    X, Y = np.meshgrid(t, s, indexing='ij')

    # Evaluation of the functional
    Z =  fd((t,s), derivative=derivative, grid=True)

    for i in range(fd.nsamples):
        ax.plot_wireframe(X, Y, Z[i], color=f"C{i}", alpha=0.6)

    return ax


###############################################################################
# Function composition can be applied to our data once is in functional
# form using the method :func:`compose`.
#
# Let :math:`f: X \rightarrow Y` and :math:`g: Y \rightarrow Z`, the composition
# will produce a third function :math:`f \circ g: X \rightarrow Z` which maps
# :math:`x \in X` to :math:`g(f(x))` [1].
#
# In `Landmark Registration <plot_landmark_registration>` it is shown the
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
g.interpolator = fda.grid.GridSplineInterpolator(interpolation_order=3)

plot_3d(g)

###############################################################################
# We will create a parametric curve :math:`f(t)=(10 \, \cos(t), 10 \, sin(t))`.
# The result of the composition, :math:`f \circ g:\mathbb{R} \rightarrow
# \mathbb{R}`
# will be another functional object with the values of :math:`g` along the path
# given by :math:`f`.
#

# Creation of circunference in parametric form
t = np.linspace(0, 2*np.pi, 100)

data_matrix = [10 * np.array([np.cos(t), np.sin(t)]).T]
f = fda.FDataGrid(data_matrix, t)

# Composition of function
fog = g.compose(f)

plt.figure()

fog.plot()

###############################################################################
# In the following chart it is plotted the curve
# :math:`(10 \, \cos(t), 10 \, sin(t), f \circ g (t))` and the surface.
#

ax = plot_3d(g)

path = f(t)[0]
ax.plot(path[:,0], path[:,1], fog(t)[0], color="orange")

###############################################################################
# [1] Function composition `https://en.wikipedia.org/wiki/Function_composition
# <https://en.wikipedia.org/wiki/Function_composition>`_.
#
