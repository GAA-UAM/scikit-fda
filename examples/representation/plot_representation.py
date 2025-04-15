"""
Representation of functional data
=================================

Explores the different representations of functional data.
"""

# Author: Carlos Ramos Carreño
# License: MIT

# %%
# In this example we are going to show the different representations of
# functional data available in scikit-fda.
#
# First we are going to fetch a functional data dataset, such as the Berkeley
# Growth Study. This dataset correspond to the height of several boys and
# girls measured until the 18 years of age. The number and times of the
# measurements are the same for each individual.

import skfda

dataset = skfda.datasets.fetch_growth()
fd = dataset["data"]
y = dataset["target"]

print(repr(fd))

fd.plot(group=y, group_colors=["red", "blue"])

# %%
# This kind of representation is a discretized representation, in which the
# measurement points are shared between samples.
print(fd.grid_points)

# %%
# In this representation, the data can be arranged as a matrix.
print(fd.data_matrix)

# %%
# By default, the data points are interpolated using a linear interpolation,
# but this is configurable.
dataset = skfda.datasets.fetch_medflies()
fd = dataset["data"]

first_curve = fd[0]
first_curve.plot()

# %%
# The interpolation used can however be changed. Here, we will use an
# interpolation with degree 3 splines.

from skfda.representation.interpolation import SplineInterpolation

first_curve.interpolation = SplineInterpolation(3)
first_curve.plot()

# %%
# This representation allows also functions with arbitrary dimensions of the
# domain and codomain.
fd = skfda.datasets.make_multimodal_samples(
    n_samples=1,
    dim_domain=2,
    dim_codomain=2,
)

print(fd.dim_domain)
print(fd.dim_codomain)

fd.plot()

# %%
# Another possible representation is a decomposition in a basis of functions.
# $$
# f(t) = \\sum_{i=1}^N a_i \\phi_i(t)
# $$
# It is possible to transform between both representations. Let us use again
# the Berkeley Growth dataset.
dataset = skfda.datasets.fetch_growth()
fd = dataset["data"]
y = dataset["target"]

fd.plot()

# %%
# We will represent it using a basis of B-splines.

from skfda.representation import basis

fd_basis = fd.to_basis(basis.BSplineBasis(n_basis=4))

fd_basis.plot()

# %%
# We can increase the number of elements in the basis to try to reproduce the
# original data with more fidelity.

fd_basis_big = fd.to_basis(basis.BSplineBasis(n_basis=7))

fd_basis_big.plot()

#############################################################################
# Lets compare the diferent representations in the same plot, for the same
# curve

fig = fd[0].plot()
fd_basis[0].plot(fig=fig)
fd_basis_big[0].plot(fig=fig)

fig.axes[0].legend(["Original", "4 elements", "7 elements"])

# %%
# We can also see the effect of changing the basis.
# For example, in the Fourier basis the functions start and end at the same
# points if the period is equal to the domain range, so this basis is clearly
# non suitable for the Growth dataset.

fd_basis = fd.to_basis(basis.FourierBasis(n_basis=7))

fd_basis.plot()

# %%
# The data is now represented as the coefficients in the basis expansion.

print(fd_basis)
