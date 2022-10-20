"""
This module contains the definition of the constants used in the package.
The following constants are defined:
.. data:: N_POINTS_FINE_MESH
    Constant used in the discretization of a basis object, by default de
    number of points used are the maximum between
    BASIS_MIN_FACTOR * n_basis + 1 and N_POINTS_FINE_MESH.
.. data:: N_POINTS_COARSE_MESH
    Constant used in the default discretization of a basis in some methods.
.. data:: N_POINTS_UNIDIMENSIONAL_PLOT_MESH
    Number of points used in the evaluation of a function to be plotted.
.. data:: N_POINTS_SURFACE_PLOT_AX
    Number of points per axis used in the evaluation of a surface to be
    plotted.
"""

N_POINTS_COARSE_MESH = 201

N_POINTS_FINE_MESH = 501

N_POINTS_SURFACE_PLOT_AX = 30

N_POINTS_UNIDIMENSIONAL_PLOT_MESH = 501
