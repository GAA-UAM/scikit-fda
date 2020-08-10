import dcor

import sklearn.base
import sklearn.utils
import numpy as np
from ....representation import FDataGrid


def dependency(X, Y, dependency_measure=dcor.u_distance_correlation_sqr):
    '''
    Computes the dependency of each point in each trajectory in X with the
    corresponding class label in Y.
    '''

    def vectorial_dependency_measure(x):
        x = np.atleast_2d(x).transpose()

        return dependency_measure(x, Y)

    vectorial_dependency_measure = np.vectorize(
        vectorial_dependency_measure,
        otypes=[float],
        signature="(m,n)->()"
    )

    X_view = np.rollaxis(X, 0, len(X.shape))

    return vectorial_dependency_measure(X_view)


def select_local_maxima(curve, smoothing: int=1):

    selected_features = []
    scores = []

    # Grow the curve at left and right with non maxima points so that points
    # near the extremes can be processed in the same way.
    extra_1 = np.repeat(curve[0], smoothing)
    extra_2 = np.repeat(curve[-1], smoothing)
    new_curve = np.concatenate([extra_1, curve, extra_2])

    for i in range(0, len(curve)):
        interval = new_curve[i:i + 2 * smoothing + 1]
        candidate_maximum = interval[smoothing]
        assert candidate_maximum == curve[i]

        is_maxima_in_interval = np.all(interval <= candidate_maximum)
        is_not_flat = (candidate_maximum > interval[smoothing - 1] or
                       candidate_maximum > interval[smoothing + 1])

        # If the maximum is the point in the middle of the interval, it is
        # selected.
        if is_maxima_in_interval and is_not_flat:
            selected_features.append(i)
            scores.append(candidate_maximum)

    return np.array(selected_features), np.array(scores)


def maxima_hunting(X, y,
                   dependency_measure=dcor.u_distance_correlation_sqr,
                   smoothing=1):
    r'''
    Maxima Hunting variable selection.

    This is a filter variable selection method for problems with a target
    variable. It evaluates a dependency measure between each point of the
    function and the target variable, and keeps those points in which this
    dependency is a local maximum.

    Selecting the local maxima serves two purposes. First, it ensures that
    the points that are relevant in isolation are selected, as they must
    maximice their dependency with the target variable. Second, the points
    that are relevant only because they are near a relevant point (and are
    thus highly correlated with it) are NOT selected, as only local maxima
    are selected, minimizing the redundancy of the selected variables.

    For a longer explanation about the method, and comparison with other
    functional variable selection methods, we refer the reader to the
    original article [1]_.

    Parameters:

        dependency_measure (callable): dependency measure to use. By default,
            it uses the bias corrected squared distance correlation.

    Examples:

        >>> from skfda.preprocessing.dim_reduction import variable_selection
        >>> from skfda.datasets import make_gaussian_process
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

        >>> rkvs = variable_selection.MaximaHunting(smoothing=10)
        >>> _ = rkvs.fit(X, y)
        >>> point_mask = rkvs.get_support()
        >>> points = X.sample_points[0][point_mask]
        >>> np.allclose(points, [0.5], rtol=0.1)
        True

        Apply the learned dimensionality reduction

        >>> X_dimred = rkvs.transform(X)
        >>> len(X.sample_points[0])
        100
        >>> X_dimred.shape
        (10000, 1)

    References:

        .. [1] J. R. Berrendero, A. Cuevas, and J. L. Torrecilla, “Variable
               selection in functional data classification: a maxima-hunting
               proposal,” STAT SINICA, vol. 26, no. 2, pp. 619–638, 2016,
               doi: 10.5705/ss.202014.0014.

    '''

    curve = dependency(X[..., None], y, dependency_measure)

    return select_local_maxima(curve, smoothing)


class MaximaHunting(sklearn.base.BaseEstimator, sklearn.base.TransformerMixin):

    def __init__(self,
                 dependency_measure=dcor.u_distance_correlation_sqr,
                 smoothing=1):
        self.dependency_measure = dependency_measure
        self.smoothing = smoothing

    def fit(self, X: FDataGrid, y):

        X, y = sklearn.utils.validation.check_X_y(X.data_matrix[..., 0], y)

        self.features_shape_ = X.shape[1:]

        self.results_ = maxima_hunting(
            X=X,
            y=y,
            dependency_measure=self.dependency_measure,
            smoothing=self.smoothing)

        indexes = np.argsort(self.results_[1])[::-1]
        self.sorted_indexes_ = self.results_[0][indexes]

        return self

    def get_support(self, indices: bool=False):
        indexes_unraveled = self.results_[0]
        if indices:
            return indexes_unraveled
        else:
            mask = np.zeros(self.features_shape_[0], dtype=bool)
            mask[self.results_[0]] = True
            return mask

    def transform(self, X, y=None):

        sklearn.utils.validation.check_is_fitted(self)

        X = sklearn.utils.validation.check_array(X.data_matrix[..., 0])

        if X.shape[1:] != self.features_shape_:
            raise ValueError("The trajectories have a different number of "
                             "points than the ones fitted")

        return X[:, self.sorted_indexes_]
