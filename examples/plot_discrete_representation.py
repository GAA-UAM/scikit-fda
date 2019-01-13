"""
Discretized function representation
===================================

Shows how to make a discretized representation of a function.
"""

# Author: Carlos Ramos Carreño <vnmabus@gmail.com>
# License: MIT

# sphinx_gallery_thumbnail_number = 2

from fda import FDataGrid
import numpy as np

###############################################################################
# We will construct a dataset containing several sinusoidal functions with
# random displacements.

random_state = np.random.RandomState(0)

sample_points = np.linspace(0, 1)
data = np.array([np.sin((sample_points + random_state.randn())
                        * 2 * np.pi) for _ in range(5)])

###############################################################################
# The FDataGrid class is used for datasets containing discretized functions
# that are measured at the same points.

fd = FDataGrid(data, sample_points,
               dataset_label='Sinusoidal curves',
               axes_labels=['t', 'x(t)'])

fd = fd[:5]

###############################################################################
# We can plot the measured values of each function in a scatter plot.

fd.scatter(s=0.5)

###############################################################################
# We can also plot the interpolated functions.

fd.plot()
