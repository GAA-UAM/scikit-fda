"""
Getting the data
================

How to get data to use in scikit-fda.

"""

# Author: Carlos Ramos Carreño
# License: MIT

##############################################################################
# The FDataGrid class
# -------------------
#
# In order to use scikit-fda, first we need functional data to analyze.
# A common case is to have each functional observation measured at the same
# points.
# This kind of functional data is easily representable in scikit-fda using
# the :class:`~skfda.representation.grid.FDataGrid` class.
#
# The :class:`~skfda.representation.grid.FDataGrid` has two important
# attributes: ``data_matrix`` and ``grid_points``. The attribute
# ``grid_points`` is a tuple with the same length as the number of domain
# dimensions (that is, one for curves, two for surfaces...). Each of its
# elements is a 1D numpy :class:`~numpy.ndarray` containing the measurement
# points for that particular dimension. The attribute ``data_matrix`` is a
# numpy :class:`~numpy.ndarray` containing the measured values of the
# functions in the grid spanned by the grid points. For functions
# :math:`\{f_i: \mathbb{R}^p \to \mathbb{R}^q\}_{i=1}^N` this is a tensor
# with dimensions :math:`N \times M_1 \times \ldots \times M_p \times q`,
# where :math:`M_i` is the number of measurement points for the domain
# dimension :math:`i`.

##############################################################################
# In order to create a :class:`~skfda.representation.grid.FDataGrid`, these
# attributes may be provided. The attributes are converted to
# :class:`~numpy.ndarray` when necessary.

##############################################################################
# .. note::
#
#     The grid points can be omitted,
#     and in that case their number is inferred from the dimensions of
#     ``data_matrix`` and they are automatically assigned as equispaced points
#     in the unitary cube in the domain set.
#
#     In the common case of functions with domain dimension of 1, the list of
#     grid points can be passed directly as ``grid_points``.
#
#     If the codomain dimension is 1, the last dimension of ``data_matrix``
#     can be dropped.

##############################################################################
# In order to better understand the FDataGrid structure, consider the
# following example:

import skfda

grid_points = [
    [0.2, 0.5, 0.7],      # Measurement points in first domain dimension
    [0, 1],               # Measurement points in second domain dimension
]

data_matrix = [
    [                # First observation
        [            # 0.2
            [        # Value at (0.2, 0)
                [1, 2, 3, 4],
            ],
            [        # Value at (0.2, 1)
                [0, 1, -1.3, 2],
            ],
        ],
        [            # 0.5
            [        # Value at (0.5, 0)
                [-2, 0, 5.5, 7],
            ],
            [        # Value at (0.5, 1)
                [2, 1.1, -1, -2],
            ],
        ],
        [            # 0.7
            [        # Value at (0.7, 0)
                [0, 0, 1, 1],
            ],
            [        # Value at (0.7, 1)
                [-3, 5, -0.5, -2],
            ],
        ],
    ],
    # This example has only one observation. Next observations would be
    # added here.
]

fd = skfda.FDataGrid(
    data_matrix=data_matrix,
    grid_points=grid_points,
)

fd.plot()
