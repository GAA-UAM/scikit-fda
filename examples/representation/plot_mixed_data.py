"""
Mixed data structure and visualization.
======================================

This example demonstrates how to build and visualize mixed datasets
containing both scalar and functional elements using the Canadian weather
dataset. In particular, we show how to include both a function and its
derivative, either jointly in a vector-valued functional object, or
separately as independent entries. We also explore how to structure and
visualize this data using `pandas` and scikit-fda's visualization tools.
"""

# Author: Luis Hebrero Garicano
# License: MIT
# sphinx_gallery_thumbnail_number = 1


# %%
# We load the Canadian weather dataset. This dataset includes daily
# temperature and precipitation curves for 35 weather stations in Canada,
# along with a scalar variable: the climate zone of each station.

from skfda import datasets

X, y = datasets.fetch_weather(return_X_y=True)
fd_temperatures = X.coordinates[0]
fd_precipitations = X.coordinates[1]

argument_names = ("t (day)",)

fd_temperatures.argument_names = argument_names
fd_temperatures.coordinate_names = ("T(t) (ºC)",)

fd_precipitations.argument_names = argument_names
fd_precipitations.coordinate_names = ("P(t) (mm.)",)

# %%
# We visualize the two functional components separately.
import matplotlib.pyplot as plt

fig, axes = plt.subplots(1, 2, figsize=(8, 4))

fd_temperatures.plot(axes=axes[0])
fd_precipitations.plot(axes=axes[1])
fig.tight_layout()
plt.show()

# %%
# To enrich the data with information about temporal changes, we compute the
# first derivative of both temperature and precipitation curves. These
# derivatives can capture local variation patterns such as rising or falling
# trends.
#
# We will compute the derivatives over the smoothed curves using a basis
# expansion. This makes less likely that the noise of the observations is
# translated to unreliable derivative curves.
# Given that the data is of periodic nature we use a Fourier basis
# representation with 5 elements for this purpose.

import skfda

time_range = (
    fd_temperatures.grid_points[0][0],
    fd_temperatures.grid_points[0][-1],
)

basis = skfda.representation.basis.FourierBasis(
    domain_range=time_range,
    n_basis=5,
)

fd_1st_temperatures_smooth = fd_temperatures.derivative(
    method=basis,
)
fd_1st_precipitations_smooth = fd_precipitations.derivative(
    method=basis,
)

fd_1st_temperatures_smooth.coordinate_names = ("T'(t) (ºC/days)",)
fd_1st_precipitations_smooth.coordinate_names = ("P'(t) (mm./days)",)

# %%
# Let's take a look at the smoothed derivatives.

fig, axes = plt.subplots(1, 2, figsize=(8, 4))

axes[0].set_title("Temperatures smoothed first derivative")
axes[1].set_title("Precipitations smoothed first derivative")

fd_1st_temperatures_smooth.plot(axes=axes[0])
fd_1st_precipitations_smooth.plot(axes=axes[1])

fig.tight_layout()
plt.show()
# %%
# Now we build a vector-valued functional object that combines the original
# temperature and its derivative. This type of structure is useful when you
# want to treat them as a single feature with multiple components.

from skfda import concatenate

fd_vector = concatenate(
    [
        fd_precipitations,
        fd_1st_precipitations_smooth,
    ],
    as_coordinates=True,
)

fig, axes = plt.subplots(1, 2, figsize=(8, 3))

fd_vector.plot(axes=axes)
fig.tight_layout()
plt.show()

# %%
# We now create a mixed data object using a :class:`pandas.DataFrame`. This
# includes:
#
# - a scalar variable: the climate zone (weather type),
# - functional variables: the temperature :math:`T(t)` and its derivative
#   :math:`T'(t)`,
# - and the vector-valued function combining both precipitation and its
#   derivative: :math:`P_{\text{vec}}(t) = (P(t), P'(t))`.
#
# This illustrates two valid ways to include a function and its derivative
# as part of the same observation.
import pandas as pd

mixed_fd = pd.DataFrame(
    {
        "weather type": y,
        "T(t)": fd_temperatures,
        "T'(t)": fd_1st_temperatures_smooth,
        "P(t)_vec": fd_vector,
    },
)

# %%
# Finally, we use
# :func:`~skfda.exploratory.visualization.representation.plot_mixed_data`
# to visualize the full mixed dataset. Each column is visualized with an
# appropriate method, helping us explore the structure in both scalar and
# functional components.

from skfda.exploratory.visualization.representation import plot_mixed_data

fig, axes = plt.subplots(1, 4, figsize=(28, 7))

with plt.rc_context({"axes.titlesize": 18}):
    plot_mixed_data(mixed_fd, axes=axes)
    fig.suptitle("Canadian Weather", fontsize=24)

plt.show()

# %%
