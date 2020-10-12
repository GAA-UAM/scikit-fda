import abc

import scipy.stats

import numpy as np


class Depth(abc.ABC):
    """
    Abstract class representing a depth function.

    Usually it will accept a distribution in the initializer.

    """

    @abc.abstractmethod
    def __init__(self, distribution):
        pass

    @abc.abstractmethod
    def __call__(self, data_points):
        """
        Evaluate the depth over a different set of points.

        """
        pass

    @property
    def max(self):
        """
        Maximum (or supremum if there is no maximum) of the depth values.

        """
        return 1

    @property
    def min(self):
        """
        Minimum (or infimum if there is no maximum)  of the depth values.

        """
        return 0


def _cumulative_one_dim(array, values):
    searched_index = np.searchsorted(array, values, side='right')

    return searched_index / len(array)


_cumulative_distribution_ordered = np.vectorize(
    _cumulative_one_dim,
    signature='(n),(m)->(m)')


def _cumulative_distribution(column):
    """Calculates the cumulative distribution function of the values passed to
    the function and evaluates it at each point.

    Args:
        column (numpy.darray): Array containing the values over which the
            distribution function is calculated.

    Returns:
        numpy.darray: Array containing the evaluation at each point of the
            distribution function.

    Examples:
        >>> _cumulative_distribution(np.array([1, 4, 5, 1, 2, 2, 4, 1, 1, 3]))
        array([ 0.4,  0.9,  1. ,  0.4,  0.6,  0.6,  0.9,  0.4,  0.4,  0.7])

    """
    return _cumulative_distribution_ordered(np.sort(column), column)


class _UnivariateFraimanMuniz(Depth):
    """
    Univariate depth used to compute the Fraiman an Muniz depth.
    """

    def __init__(self, distribution):
        self.sorted_values = np.sort(distribution, axis=0)

    def __call__(self, data_points):
        cum_dist = _cumulative_distribution_ordered(
            np.moveaxis(self.sorted_values, 0, -1),
            np.moveaxis(data_points, 0, -1))

        assert cum_dist.shape[-2] == 1
        return 1 - np.abs(0.5 - np.moveaxis(cum_dist, -1, 0)[..., 0])

    @property
    def min(self):
        return 1 / 2


def _stagel_donoho_outlyingness(X, *, pointwise=False):

    if pointwise is False:
        raise NotImplementedError("Only implemented pointwise")

    if X.dim_codomain == 1:
        # Special case, can be computed exactly
        m = X.data_matrix[..., 0]

        return (np.abs(m - np.median(m, axis=0)) /
                scipy.stats.median_abs_deviation(m, axis=0, scale=1 / 1.4826))

    else:
        raise NotImplementedError("Only implemented for one dimension")


def projection_depth(X, *, pointwise=False):
    """Returns the projection depth.

    The projection depth is the depth function associated with the
    Stagel-Donoho outlyingness.
    """
    from . import outlyingness_to_depth

    depth = outlyingness_to_depth(_stagel_donoho_outlyingness)

    return depth(X, pointwise=pointwise)
