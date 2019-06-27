"""
This module contains the definition of the constants used in the package.

The following constants are defined:

.. data:: BASIS_MIN_FACTOR

    constant used in the discretization of a basis object, by default de
    number of points used are the maximun between BASIS_MIN_FACTOR * nbasis +1
    and N_POINTS_FINE_MESH.

.. data:: N_POINTS_FINE_MESH

    Constant used in the discretization of a basis object, by default de
    number of points used are the maximun between BASIS_MIN_FACTOR * nbasis +1
    and N_POINTS_FINE_MESH.

.. data:: N_POINTS_COARSE_MESH

    Constant used in the default discretization of a basis in some methods.

.. data:: N_POINS_UNIDIMENSIONAL_PLOT_MESH

    Number of points per axis used in the evaluation of a surface to be plotted.

.. data:: N_POINS_SURFACE_PLOT_AX

    Number of points used in the evaluation of a function to be plotted.

"""

BASIS_MIN_FACTOR = 10

N_POINTS_COARSE_MESH = 201

N_POINTS_FINE_MESH = 501

N_POINS_SURFACE_PLOT_AX = 30

N_POINS_UNIDIMENSIONAL_PLOT_MESH = 501



def _list_constants():
    r"""Return a dict containing all the constants with their values.

    Returns:
        (dict): Dictionary with the name of the constants as keys and their
        respectives values.

    """
    g = globals()

    # All the constants must be written in upper case and dont start with _
    return {c:g[c] for c in g if not c.startswith("_") and c.isupper()}
