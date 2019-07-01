"""Module with generic methods"""

import numpy as np


def _list_of_arrays(original_array):
    """Convert to a list of arrays.

    If the original list is one-dimensional (e.g. [1, 2, 3]), return list to
    array (in this case [array([1, 2, 3])]).

    If the original list is two-dimensional (e.g. [[1, 2, 3], [4, 5]]), return
    a list containing other one-dimensional arrays (in this case
    [array([1, 2, 3]), array([4, 5, 6])]).

    In any other case the behaviour is unespecified.

    """
    new_array = np.array([np.asarray(i) for i in
                          np.atleast_1d(original_array)])

    # Special case: Only one array, expand dimension
    if len(new_array.shape) == 1 and not any(isinstance(s, np.ndarray)
                                             for s in new_array):
        new_array = np.atleast_2d(new_array)

    return list(new_array)


def _coordinate_list(axes):
    """Convert a list with axes in a list with coordinates.

    Computes the cartesian product of the axes and returns a numpy array of
    1 dimension with all the possible combinations, for an arbitrary number of
    dimensions.

    Args:
        Axes (array_like): List with axes.

    Return:
        (np.ndarray): Numpy 2-D array with all the possible combinations.
        The entry (i,j) represent the j-th coordinate of the i-th point.

    Examples:

        >>> from skfda.representation._functional_data import _coordinate_list
        >>> axes = [[0,1],[2,3]]
        >>> _coordinate_list(axes)
        array([[0, 2],
               [0, 3],
               [1, 2],
               [1, 3]])

        >>> axes = [[0,1],[2,3],[4]]
        >>> _coordinate_list(axes)
        array([[0, 2, 4],
               [0, 3, 4],
               [1, 2, 4],
               [1, 3, 4]])

        >>> axes = [[0,1]]
        >>> _coordinate_list(axes)
        array([[0],
               [1]])

    """
    return np.vstack(list(map(np.ravel, np.meshgrid(*axes, indexing='ij')))).T
