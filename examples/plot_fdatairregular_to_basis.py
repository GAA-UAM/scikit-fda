"""
Irregular data to basis representation
=======================================================================

Convert irregular data to a basis representation using the ``to_basis``
method of the :class:`skfda.representation.irregular.FDataIrregular` class.
"""
# Author: Pablo Cuesta Sierra
# License: MIT

# sphinx_gallery_thumbnail_number = -1

import matplotlib.pyplot as plt
import numpy as np

from skfda.datasets import fetch_weather, irregular_sample
from skfda.representation.basis import FourierBasis
from skfda.misc.scoring import r2_score

np.random.seed(439472)  # set the seed for reproducibility

# %%
# First, the Canadian Weather dataset is downloaded from the package 'fda' in
# CRAN. It contains a FDataGrid with daily temperatures and precipitations,
# that is, it has a 2-dimensional image. We are interested only in the daily
# average temperatures, so we will use the first coordinate.
#
# As we want to ilustrate the conversion of irregular data to basis,
# representation, we will take an irregular sample of the temperatures dataset
# containing only 8 points per curve.
fd_temperatures = fetch_weather().data.coordinates[0]
irregular_temperatures = irregular_sample(
    fdata=fd_temperatures, n_points_per_curve=8,
)

# %%
# To get an idea of the irregular data we will be working with, 6 of the
# irregular curves are plotted, along with the original curves
# that they come from.
fig = plt.figure()
irregular_temperatures[-6:].scatter(fig=fig)
fd_temperatures[-6:].plot(fig=fig, alpha=0.1)
plt.show()

# %%
# Now, we will convert the irregularly sampled temperature curves to basis
# representation. Due to the periodicity of the data, we will be using a
# Fourier basis.
basis = FourierBasis(n_basis=5, domain_range=fd_temperatures.domain_range)
irregular_temperatures_converted = irregular_temperatures.to_basis(
    basis, conversion_type="mixed_effects",
)

# %%
# To visualize the conversion, we will now plot 6 of the converted
# curves (smooth basis representation) along with the original temperatures
# (non-smooth) and the irregular points that we sampled.
fig = plt.figure(figsize=(10, 14))
for k in range(6):
    axes = plt.subplot(3, 2, k + 1)
    fd_temperatures.plot(axes=axes, alpha=0.05, color="black")
    fd_temperatures[k].plot(axes=axes, color=f"C{k}")
    irregular_temperatures_converted[k].plot(axes=axes, color=f"C{k}")
    irregular_temperatures[k].scatter(axes=axes, color=f"C{k}")
plt.show()

# %%
# Finally, we will get a score of the quality of the conversion by comparing
# the obtained basis representation (``irregular_temperatures_converted``)
# with the original data (``fd_temperatures``) from the CRAN dataset. We will
# be using the :func:`skfda.misc.scoring.r2_score`.
#
# Note that, to compare the original data and the basis representation (which
# have different :class:`FData` types), we have to evaluate the latter at
# the grid points of the former.
r2 = r2_score(
    fd_temperatures,
    irregular_temperatures_converted.to_grid(fd_temperatures.grid_points),
)
print(f"R2 score: {r2:.2f}")

# %%
# References
# ----------
#
# .. footbibliography::
