import dcor

import scipy.signal
import sklearn.base
import sklearn.utils

import numpy as np

from ....representation import FDataGrid


def _compute_dependence(X, y, *, dependence_measure):
    '''
    Computes the dependence of each point in each trajectory in X with the
    corresponding class label in Y.
    '''

    # Move n_samples to the end
    # The shape is now input_shape + n_samples + n_output
    X = np.moveaxis(X, 0, -2)

    input_shape = X.shape[:-2]

    # Join input in a list for rowwise
    X = X.reshape(-1, X.shape[-2], X.shape[-1])

    if y.ndim == 1:
        y = np.atleast_2d(y).T
    Y = np.array([y] * len(X))

    dependence_results = dcor.rowwise(dependence_measure, X, Y)

    return dependence_results.reshape(input_shape)


def select_local_maxima(X, *, order: int=1):
    r'''
    Compute local maxima of an array.

    Points near the boundary are considered maxima looking only at one side.

    For flat regions only the boundary points of the flat region could be
    considered maxima.

    Parameters:

        X (numpy array): Where to compute the local maxima.
        order (callable): How many points on each side to look, to check if
            a point is a maximum in that interval.

    Examples:

        >>> from skfda.preprocessing.dim_reduction.variable_selection.\
        ...     maxima_hunting import select_local_maxima
        >>> import numpy as np

        >>> x = np.array([2, 1, 1, 1, 2, 3, 3, 3, 2, 3, 4, 3, 2])
        >>> select_local_maxima(x).astype(np.int_)
        array([ 0,  5,  7, 10])

        The ``order`` parameter can be used to check a larger interval to see
        if a point is still a maxima, effectively eliminating small local
        maxima.

        >>> x = np.array([2, 1, 1, 1, 2, 3, 3, 3, 2, 3, 4, 3, 2])
        >>> select_local_maxima(x, order=3).astype(np.int_)
        array([ 0,  5, 10])

    '''
    indexes = scipy.signal.argrelextrema(
        X, comparator=np.greater_equal, order=order)[0]

    # Discard flat
    maxima = X[indexes]

    left_points = np.take(X, indexes - 1, mode='clip')
    right_points = np.take(X, indexes + 1, mode='clip')

    is_not_flat = (maxima > left_points) | (maxima > right_points)

    return indexes[is_not_flat]


class MaximaHunting(sklearn.base.BaseEstimator, sklearn.base.TransformerMixin):
    r'''
    Maxima Hunting variable selection.

    This is a filter variable selection method for problems with a target
    variable. It evaluates a dependence measure between each point of the
    function and the target variable, and keeps those points in which this
    dependence is a local maximum.

    Selecting the local maxima serves two purposes. First, it ensures that
    the points that are relevant in isolation are selected, as they must
    maximice their dependence with the target variable. Second, the points
    that are relevant only because they are near a relevant point (and are
    thus highly correlated with it) are NOT selected, as only local maxima
    are selected, minimizing the redundancy of the selected variables.

    For a longer explanation about the method, and comparison with other
    functional variable selection methods, we refer the reader to the
    original article [1]_.

    Parameters:

        dependence_measure (callable): Dependence measure to use. By default,
            it uses the bias corrected squared distance correlation.
        local_maxima_selector (callable): Function to detect local maxima. The
            default is :func:`select_local_maxima` with ``order`` parameter
            equal to one. The original article used a similar function testing
            different values of ``order``.

    Examples:

        >>> from skfda.preprocessing.dim_reduction import variable_selection
        >>> from skfda.preprocessing.dim_reduction.variable_selection.\
        ...     maxima_hunting import select_local_maxima
        >>> from skfda.datasets import make_gaussian_process
        >>> from functools import partial
        >>> import skfda
        >>> import numpy as np

        We create trajectories from two classes, one with zero mean and the
        other with a peak-like mean. Both have Brownian covariance.

        >>> n_samples = 10000
        >>> n_features = 100
        >>>
        >>> def mean_1(t):
        ...     return (np.abs(t - 0.25)
        ...             - 2 * np.abs(t - 0.5)
        ...             + np.abs(t - 0.75))
        >>>
        >>> X_0 = make_gaussian_process(n_samples=n_samples // 2,
        ...                             n_features=n_features,
        ...                             random_state=0)
        >>> X_1 = make_gaussian_process(n_samples=n_samples // 2,
        ...                             n_features=n_features,
        ...                             mean=mean_1,
        ...                             random_state=1)
        >>> X = skfda.concatenate((X_0, X_1))
        >>>
        >>> y = np.zeros(n_samples)
        >>> y [n_samples // 2:] = 1

        Select the relevant points to distinguish the two classes

        >>> local_maxima_selector = partial(select_local_maxima, order=10)
        >>> mh = variable_selection.MaximaHunting(
        ...            local_maxima_selector=local_maxima_selector)
        >>> _ = mh.fit(X, y)
        >>> point_mask = mh.get_support()
        >>> points = X.grid_points[0][point_mask]
        >>> np.allclose(points, [0.5], rtol=0.1)
        True

        Apply the learned dimensionality reduction

        >>> X_dimred = mh.transform(X)
        >>> len(X.grid_points[0])
        100
        >>> X_dimred.shape
        (10000, 1)

    References:

        .. [1] J. R. Berrendero, A. Cuevas, and J. L. Torrecilla, “Variable
               selection in functional data classification: a maxima-hunting
               proposal,” STAT SINICA, vol. 26, no. 2, pp. 619–638, 2016,
               doi: 10.5705/ss.202014.0014.

    '''

    def __init__(self,
                 dependence_measure=dcor.u_distance_correlation_sqr,
                 local_maxima_selector=select_local_maxima):
        self.dependence_measure = dependence_measure
        self.local_maxima_selector = local_maxima_selector

    def fit(self, X: FDataGrid, y):

        self.features_shape_ = X.data_matrix.shape[1:]
        self.dependence_ = _compute_dependence(
            X.data_matrix, y,
            dependence_measure=self.dependence_measure)

        self.indexes_ = self.local_maxima_selector(self.dependence_)

        sorting_indexes = np.argsort(self.dependence_[self.indexes_])[::-1]
        self.sorted_indexes_ = self.indexes_[sorting_indexes]

        return self

    def get_support(self, indices: bool=False):
        if indices:
            return self.indexes_
        else:
            mask = np.zeros(self.features_shape_[0:-1], dtype=bool)
            mask[self.indexes_] = True
            return mask

    def transform(self, X, y=None):

        sklearn.utils.validation.check_is_fitted(self)

        if X.data_matrix.shape[1:] != self.features_shape_:
            raise ValueError("The trajectories have a different number of "
                             "points than the ones fitted")

        return X.data_matrix[:, self.sorted_indexes_].reshape(X.n_samples, -1)
