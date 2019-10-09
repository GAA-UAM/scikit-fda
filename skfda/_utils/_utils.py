"""Module with generic methods"""

import functools

import types

import numpy as np

def _check_univariate(fd):
    """Checks if an FData is univariate and raises an error"""

    if fd.dim_domain != 1 or fd.dim_codomain != 1:
        raise ValueError(f"The functional data must be univariate, i.e., " +
                         f"with dim_domain=1 " +
                         (f"" if fd.dim_domain==1
                          else f"(currently is {fd.dim_domain}) ") +
                         f"and dim_codomain=1 " +
                         (f"" if fd.dim_codomain==1 else
                          f"(currently is  {fd.dim_codomain})"))

def _to_grid(X, y, eval_points=None):
    """Transforms the functional data in grids to perform calculations."""

    from .. import FDataGrid
    x_is_grid = isinstance(X, FDataGrid)
    y_is_grid = isinstance(y, FDataGrid)

    if eval_points is not None:
        X = X.to_grid(eval_points)
        y = y.to_grid(eval_points)
    elif x_is_grid and not y_is_grid:
        y = y.to_grid(X.sample_points[0])
    elif not x_is_grid and y_is_grid:
        X = X.to_grid(y.sample_points[0])
    elif not x_is_grid and not y_is_grid:
        X = X.to_grid()
        y = y.to_grid()

    return X, y


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


def parameter_aliases(**alias_assignments):
    """Allows using aliases for parameters"""
    def decorator(f):

        if isinstance(f, (types.FunctionType, types.LambdaType)):
            # f is a function
            @functools.wraps(f)
            def aliasing_function(*args, **kwargs):
                nonlocal alias_assignments
                for parameter_name, aliases in alias_assignments.items():
                    aliases = tuple(aliases)
                    aliases_used = [a for a in kwargs
                                    if a in aliases + (parameter_name,)]
                    if len(aliases_used) > 1:
                        raise ValueError(
                            f"Several arguments with the same meaning used: " +
                            str(aliases_used))

                    elif len(aliases_used) == 1:
                        arg = kwargs.pop(aliases_used[0])
                        kwargs[parameter_name] = arg

                return f(*args, **kwargs)
            return aliasing_function

        else:
            # f is a class

            class cls(f):
                pass

            nonlocal alias_assignments
            init = cls.__init__
            cls.__init__ = parameter_aliases(**alias_assignments)(init)

            set_params = getattr(cls, "set_params", None)
            if set_params is not None:  # For estimators
                cls.set_params = parameter_aliases(
                    **alias_assignments)(set_params)

            for key, value in alias_assignments.items():
                def getter(self):
                    return getattr(self, key)

                def setter(self, new_value):
                    return setattr(self, key, new_value)

                for alias in value:
                    setattr(cls, alias, property(getter, setter))

            cls.__name__ = f.__name__
            cls.__doc__ = f.__doc__
            cls.__module__ = f.__module__

            return cls

    return decorator


def _check_estimator(estimator):
    from sklearn.utils.estimator_checks import (
        check_get_params_invariance, check_set_params)

    name = estimator.__name__
    instance = estimator()
    check_get_params_invariance(name, instance)
    check_set_params(name, instance)
