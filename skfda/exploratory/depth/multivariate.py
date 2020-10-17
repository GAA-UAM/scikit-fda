import abc
from scipy.special import comb

import scipy.stats
import sklearn

import numpy as np


class Depth(abc.ABC, sklearn.base.BaseEstimator):
    """
    Abstract class representing a depth function.

    Usually it will accept a distribution in the initializer.

    """

    def fit(self, X, y=None):
        """
        Learn the distribution from the observations.

        Args:
            X: Functional dataset from which the distribution of the data is
               inferred.
            y: Unused. Kept only for convention.

        Returns:
            self: Fitted estimator.

        """
        return self

    @abc.abstractmethod
    def predict(self, X):
        """
        Compute the depth inside the learned distribution.

        Args:
            X: Points whose depth is going to be evaluated.

        """
        pass

    def fit_predict(self, X, y=None):
        """
        Compute the depth of each observation with respect to the whole
        dataset.

        Args:
            X: Dataset.
            y: Unused. Kept only for convention.

        """
        return self.fit(X).predict(X)

    def __call__(self, X, distribution=None):
        """
        Allows the depth to be used as a function.

        Args:
            X: Points whose depth is going to be evaluated.
            distribution: Functional dataset from which the distribution of
                the data is inferred. If ``None`` it is the same as ``X``.

        """
        copy = sklearn.base.clone(self)

        if distribution is None:
            return copy.fit_predict(X)
        else:
            return copy.fit(distribution).predict(X)

    @property
    def max(self):
        """
        Maximum (or supremum if there is no maximum) of the depth values.

        """
        return 1

    @property
    def min(self):
        """
        Minimum (or infimum if there is no maximum) of the depth values.

        """
        return 0


def _searchsorted_one_dim(array, values, *, side='left'):
    searched_index = np.searchsorted(array, values, side=side)

    return searched_index


_searchsorted_vectorized = np.vectorize(
    _searchsorted_one_dim,
    signature='(n),(m),()->(m)',
    excluded='side')


def _searchsorted_ordered(array, values, *, side='left'):
    return _searchsorted_vectorized(array, values, side=side)


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
    return _searchsorted_ordered(np.sort(column), column,
                                 side='right') / len(column)


class _UnivariateFraimanMuniz(Depth):
    """
    Univariate depth used to compute the Fraiman an Muniz depth.

    """

    def fit(self, X, y=None):
        self._sorted_values = np.sort(X, axis=0)
        return self

    def predict(self, X):
        cum_dist = _searchsorted_ordered(
            np.moveaxis(self._sorted_values, 0, -1),
            np.moveaxis(X, 0, -1), side='right') / len(self._sorted_values)

        assert cum_dist.shape[-2] == 1
        return 1 - np.abs(0.5 - np.moveaxis(cum_dist, -1, 0)[..., 0])

    @property
    def min(self):
        return 1 / 2


class SimplicialDepth(Depth):
    """
    Simplicial depth.

    """

    def fit(self, X, y=None):
        self._dim = X.shape[-1]

        if self._dim == 1:
            self.sorted_values = np.sort(X, axis=0)
        else:
            raise NotImplementedError("SimplicialDepth is currently only "
                                      "implemented for one-dimensional data.")

        return self

    def predict(self, X):

        assert self._dim == X.shape[-1]

        if self._dim == 1:
            positions_left = _searchsorted_ordered(
                np.moveaxis(self.sorted_values, 0, -1),
                np.moveaxis(X, 0, -1))

            positions_left = np.moveaxis(positions_left, -1, 0)[..., 0]

            positions_right = _searchsorted_ordered(
                np.moveaxis(self.sorted_values, 0, -1),
                np.moveaxis(X, 0, -1), side='right')

            positions_right = np.moveaxis(positions_right, -1, 0)[..., 0]

            num_strictly_below = positions_left
            num_strictly_above = len(self.sorted_values) - positions_right

            total_pairs = comb(len(self.sorted_values), 2)

        return (total_pairs - comb(num_strictly_below, 2)
                - comb(num_strictly_above, 2)) / total_pairs

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
