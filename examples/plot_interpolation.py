"""
Interpolation
=====================

This example shows the types of interpolation used in the evaluation of
FDataGrids.
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

def scatter_3d(fd, ax=None):

    if ax is None:
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

    t = fd.sample_points[0]
    s = fd.sample_points[1]
    X, Y = np.meshgrid(t, s, indexing='ij')

    for i in range(fd.nsamples):
        ax.scatter(X, Y, fd.data_matrix[i,...,0], color=f"C{i}")
    return ax


###############################################################################
# The :class:`FDataGrid` class is used for datasets containing discretized
# functions. For the evaluation between the points of discretization, or sample
# points, is necessary to interpolate.
#
# We will construct an example dataset with two curves with 6 points of
# discretization.
#

fd = fda.datasets.make_sinusoidal_process(n_samples=2, n_features=6,
                                          random_state=1)
fd.scatter()
plt.legend(["Sample 1", "Sample 2"])

###############################################################################
# By default it is used linear interpolation, which is one of the simplest
# methods of interpolation and therefore one of the least computationally
# expensive, but has the disadvantage that the interpolant is not differentiable
# at the points of discretization.
#

fd.plot()
fd.scatter()

################################################################################
# The interpolation method of the FDataGrid could be changed setting the
# attribute `interpolator`. Once we have set an interpolator it is used for
# the evaluation of the object.
#
# Polynomial spline interpolation could be performed using the interpolator
# :class:`GridSplineInterpolator`. In the following example a cubic interpolator
# is set.

fd.interpolator = fda.grid.GridSplineInterpolator(interpolation_order=3)

plot_aux(fd)
fd.scatter()


###############################################################################
# Smooth interpolation could be performed with the attribute
# `smoothness_parameter` of the spline interpolator.
#

# Sample with noise
fd_smooth = fda.datasets.make_sinusoidal_process(n_samples=1, n_features=30,
                                                 random_state=1, error_std=.3)

# Cubic interpolator
fd_smooth.interpolator = fda.grid.GridSplineInterpolator(interpolation_order=3)

plot_aux(fd_smooth, label="Cubic")

# Smooth interpolation
fd_smooth.interpolator = fda.grid.GridSplineInterpolator(interpolation_order=3,
                                                         smoothness_parameter=1.5)

plot_aux(fd_smooth, label="Cubic smoothed")

fd_smooth.scatter()
plt.legend()


###############################################################################
# It is possible to evaluate derivatives of the FDatagrid,
# but due to the fact that interpolation is performed first, the interpolation
# loses one degree for each order of derivation. In the next example, it is
# shown the first derivative of a sample using interpolation with different
# degrees.
#

fd = fd[1]

for i in range(1, 4):
    fd.interpolator = fda.grid.GridSplineInterpolator(interpolation_order=i)
    plot_aux(fd, derivative=1, label=f"Degree {i}")

plt.legend()

###############################################################################
# FDataGrids can be differentiate using lagged differences with the
# method :func:`derivative`, creating another FDataGrid which could be
# interpolated in order to avoid interpolating before differentiating.
#

fd_derivative = fd.derivative()

plot_aux(fd_derivative, label="Differentiation first")
fd_derivative.scatter()

plot_aux(fd, derivative=1, label="Interpolation first")

plt.legend()

###############################################################################
# Sometimes our samples are required to be monotone, in these cases it is
# possible to use monotone cubic interpolation with the attribute `monotone`.
# A piecewise cubic hermite interpolating polynomial (PCHIP) will be used.
#


fd_monotone = fd.copy(data_matrix=np.sort(fd.data_matrix, axis=1))


plot_aux(fd_monotone, linestyle='--', label="cubic")



fd_monotone.interpolator = fda.grid.GridSplineInterpolator(interpolation_order=3,
                                                           monotone=True)
plot_aux(fd_monotone, label="PCHIP")

fd_monotone.scatter(c='C1')
plt.legend()

###############################################################################
# All the interpolators will work regardless of the dimension of the image, but
# depending on the domain dimension some methods will not be available.
#
# For the next examples it is constructed a surface, :math:`x_i: \mathbb{R}^2
# \longmapsto \mathbb{R}`. By default, as in unidimensional samples, it is used
# linear interpolation.
#

X, Y, Z = axes3d.get_test_data(1.2)
data_matrix = [Z.T]
sample_points = [X[0,:], Y[:, 0]]


fd = fda.FDataGrid(data_matrix, sample_points)

ax = plot_3d(fd)
scatter_3d(fd, ax)

###############################################################################
# In the following figure it is shown the result of the cubic interpolation
# applied to the surface.
#

fd.interpolator = fda.grid.GridSplineInterpolator(interpolation_order=3)

ax = plot_3d(fd)
scatter_3d(fd, ax)

plt.show()

###############################################################################
# In case of surface derivatives could be taked in two directions, for this
# reason a tuple with the order of derivates in each direction could be passed.
# Let :math:`x(t,s)` be the surface, in the following example it is shown the
# derivative with respect to the second coordinate, :math:`\frac{\partial}
# {\partial s}x(t,s)`.

plot_3d(fd, derivative=(0, 1))

plt.show()

###############################################################################
# The following table shows the interpolation methods available by the class
# :class:`GridSplineInterpolator` depending on the domain dimension.
#
# +------------------+--------+----------------+----------+-------------+-------------+
# | Domain dimension | Linear | Up to degree 5 | Monotone | Derivatives |  Smoothing  |
# +==================+========+================+==========+=============+=============+
# |         1        |   ✔    |       ✔        |    ✔     |      ✔      |      ✔      |
# +------------------+--------+----------------+----------+-------------+-------------+
# |         2        |   ✔    |       ✔        |    ✖     |      ✔      |      ✔      |
# +------------------+--------+----------------+----------+-------------+-------------+
# |     3 or more    |   ✔    |       ✖        |    ✖     |      ✖      |      ✖      |
# +------------------+--------+----------------+----------+-------------+-------------+
#
