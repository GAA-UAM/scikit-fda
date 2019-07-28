

from sklearn.neighbors import KNeighborsRegressor as _KNeighborsRegressor
from sklearn.neighbors import (RadiusNeighborsRegressor as
                               _RadiusNeighborsRegressor)
from sklearn.base import RegressorMixin

from .base import (NeighborsBase, NeighborsMixin,
                   KNeighborsMixin, RadiusNeighborsMixin,
                   NeighborsScalarRegresorMixin,
                   NeighborsFunctionalRegressorMixin,
                   NearestNeighborsMixinInit)

from ..exploratory.stats import mean
from ..misc.metrics import lp_distance


class KNeighborsScalarRegressor(NeighborsBase, NeighborsMixin,
                                KNeighborsMixin, RegressorMixin,
                                NeighborsScalarRegresorMixin):
    """Regression based on k-nearest neighbors with scalar response.

    The target is predicted by local interpolation of the targets associated of
    the nearest neighbors in the training set.

    Parameters
    ----------
    n_neighbors : int, optional (default = 5)
        Number of neighbors to use by default for :meth:`kneighbors` queries.
    weights : str or callable, optional (default = 'uniform')
        weight function used in prediction.  Possible values:

        - 'uniform' : uniform weights.  All points in each neighborhood
          are weighted equally.
        - 'distance' : weight points by the inverse of their distance.
          in this case, closer neighbors of a query point will have a
          greater influence than neighbors which are further away.
        - [callable] : a user-defined function which accepts an
          array of distances, and returns an array of the same shape
          containing the weights.

    algorithm : {'auto', 'ball_tree', 'brute'}, optional
        Algorithm used to compute the nearest neighbors:

        - 'ball_tree' will use :class:`sklearn.neighbors.BallTree`.
        - 'brute' will use a brute-force search.
        - 'auto' will attempt to decide the most appropriate algorithm based on
          the values passed to :meth:`fit` method.

    leaf_size : int, optional (default = 30)
        Leaf size passed to BallTree or KDTree.  This can affect the
        speed of the construction and query, as well as the memory
        required to store the tree.  The optimal value depends on the
        nature of the problem.
    metric : string or callable, (default
        :func:`lp_distance <skfda.misc.metrics.lp_distance>`)
        the distance metric to use for the tree.  The default metric is
        the Lp distance. See the documentation of the metrics module
        for a list of available metrics.
    metric_params : dict, optional (default = None)
        Additional keyword arguments for the metric function.
    n_jobs : int or None, optional (default=None)
        The number of parallel jobs to run for neighbors search.
        ``None`` means 1 unless in a :obj:`joblib.parallel_backend` context.
        ``-1`` means using all processors.
        Doesn't affect :meth:`fit` method.
    sklearn_metric : boolean, optional (default = False)
        Indicates if the metric used is a sklearn distance between vectors (see
        :class:`sklearn.neighbors.DistanceMetric`) or a functional metric of
        the module :mod:`skfda.misc.metrics`.
    Examples
    --------
    Firstly, we will create a toy dataset with gaussian-like samples shifted.

    >>> from skfda.datasets import make_multimodal_samples
    >>> from skfda.datasets import make_multimodal_landmarks
    >>> y = make_multimodal_landmarks(n_samples=30, std=.5, random_state=0)
    >>> y = y.flatten()
    >>> fd = make_multimodal_samples(n_samples=30, std=.5, random_state=0)

    We will fit a K-Nearest Neighbors regressor to regress a scalar response.

    >>> from skfda.ml.regression import KNeighborsScalarRegressor
    >>> neigh = KNeighborsScalarRegressor()
    >>> neigh.fit(fd, y)
    KNeighborsScalarRegressor(algorithm='auto', leaf_size=30,...)

    We can predict the modes of new samples

    >>> neigh.predict(fd[:4]).round(2) # Predict first 4 locations
    array([ 0.79,  0.27,  0.71,  0.79])

    See also
    --------
    KNeighborsClassifier
    RadiusNeighborsClassifier
    RadiusNeighborsScalarRegressor
    NearestNeighbors
    NearestCentroids
    Notes
    -----
    See Nearest Neighbors in the sklearn online documentation for a discussion
    of the choice of ``algorithm`` and ``leaf_size``.

    This class wraps the sklearn regressor
    `sklearn.neighbors.KNeighborsRegressor`.

    .. warning::
       Regarding the Nearest Neighbors algorithms, if it is found that two
       neighbors, neighbor `k+1` and `k`, have identical distances
       but different labels, the results will depend on the ordering of the
       training data.

    https://en.wikipedia.org/wiki/K-nearest_neighbor_algorithm

    """

    def __init__(self, n_neighbors=5, weights='uniform', algorithm='auto',
                 leaf_size=30, metric=lp_distance, metric_params=None,
                 n_jobs=1, sklearn_metric=False):
        """Initialize the classifier."""

        super().__init__(n_neighbors=n_neighbors,
                         weights=weights, algorithm=algorithm,
                         leaf_size=leaf_size, metric=metric,
                         metric_params=metric_params, n_jobs=n_jobs,
                         sklearn_metric=sklearn_metric)

    def _init_estimator(self, sk_metric):
        """Initialize the sklearn K neighbors estimator.

        Args:
            sk_metric: (pyfunc or 'precomputed'): Metric compatible with
                sklearn API or matrix (n_samples, n_samples) with precomputed
                distances.

        Returns:
            Sklearn K Neighbors estimator initialized.

        """
        return _KNeighborsRegressor(
            n_neighbors=self.n_neighbors, weights=self.weights,
            algorithm=self.algorithm, leaf_size=self.leaf_size,
            metric=sk_metric, metric_params=self.metric_params,
            n_jobs=self.n_jobs)


class RadiusNeighborsScalarRegressor(NeighborsBase, NeighborsMixin,
                                     RadiusNeighborsMixin, RegressorMixin,
                                     NeighborsScalarRegresorMixin):
    """Scalar regression based on neighbors within a fixed radius.

    The target is predicted by local interpolation of the targets associated of
    the nearest neighbors in the training set.

    Parameters
    ----------
    radius : float, optional (default = 1.0)
        Range of parameter space to use by default for :meth:`radius_neighbors`
        queries.
    weights : str or callable
        weight function used in prediction.  Possible values:

        - 'uniform' : uniform weights.  All points in each neighborhood
          are weighted equally.
        - 'distance' : weight points by the inverse of their distance.
          in this case, closer neighbors of a query point will have a
          greater influence than neighbors which are further away.
        - [callable] : a user-defined function which accepts an
          array of distances, and returns an array of the same shape
          containing the weights.

        Uniform weights are used by default.
    algorithm : {'auto', 'ball_tree', 'brute'}, optional
        Algorithm used to compute the nearest neighbors:

        - 'ball_tree' will use :class:`sklearn.neighbors.BallTree`.
        - 'brute' will use a brute-force search.
        - 'auto' will attempt to decide the most appropriate algorithm
          based on the values passed to :meth:`fit` method.

    leaf_size : int, optional (default = 30)
        Leaf size passed to BallTree. This can affect the
        speed of the construction and query, as well as the memory
        required to store the tree. The optimal value depends on the
        nature of the problem.
    metric : string or callable, (default
        :func:`lp_distance <skfda.metrics.lp_distance>`)
        the distance metric to use for the tree.  The default metric is
        the Lp distance. See the documentation of the metrics module
        for a list of available metrics.
    metric_params : dict, optional (default = None)
        Additional keyword arguments for the metric function.
    n_jobs : int or None, optional (default=None)
        The number of parallel jobs to run for neighbors search.
        ``None`` means 1 unless in a :obj:`joblib.parallel_backend` context.
        ``-1`` means using all processors.
    sklearn_metric : boolean, optional (default = False)
        Indicates if the metric used is a sklearn distance between vectors (see
        :class:`sklearn.neighbors.DistanceMetric`) or a functional metric of
        the module :mod:`skfda.misc.metrics`.
    Examples
    --------
    Firstly, we will create a toy dataset with gaussian-like samples shifted.

    >>> from skfda.datasets import make_multimodal_samples
    >>> from skfda.datasets import make_multimodal_landmarks
    >>> y = make_multimodal_landmarks(n_samples=30, std=.5, random_state=0)
    >>> y = y.flatten()
    >>> fd = make_multimodal_samples(n_samples=30, std=.5, random_state=0)


    We will fit a K-Nearest Neighbors regressor to regress a scalar response.

    >>> from skfda.ml.regression import RadiusNeighborsScalarRegressor
    >>> neigh = RadiusNeighborsScalarRegressor(radius=.2)
    >>> neigh.fit(fd, y)
    RadiusNeighborsScalarRegressor(algorithm='auto', leaf_size=30,...)

    We can predict the modes of new samples.

    >>> neigh.predict(fd[:4]).round(2) # Predict first 4 locations
    array([ 0.84,  0.27,  0.66,  0.79])

    See also
    --------
    KNeighborsClassifier
    RadiusNeighborsClassifier
    KNeighborsScalarRegressor
    NearestNeighbors
    NearestCentroids
    Notes
    -----
    See Nearest Neighbors in the sklearn online documentation for a discussion
    of the choice of ``algorithm`` and ``leaf_size``.

    This class wraps the sklearn classifier
    `sklearn.neighbors.RadiusNeighborsClassifier`.

    https://en.wikipedia.org/wiki/K-nearest_neighbor_algorithm

    """

    def __init__(self, radius=1.0, weights='uniform', algorithm='auto',
                 leaf_size=30, metric=lp_distance, metric_params=None,
                 n_jobs=1, sklearn_metric=False):
        """Initialize the classifier."""

        super().__init__(radius=radius, weights=weights, algorithm=algorithm,
                         leaf_size=leaf_size, metric=metric,
                         metric_params=metric_params, n_jobs=n_jobs,
                         sklearn_metric=sklearn_metric)

    def _init_estimator(self, sk_metric):
        """Initialize the sklearn radius neighbors estimator.

        Args:
            sk_metric: (pyfunc or 'precomputed'): Metric compatible with
                sklearn API or matrix (n_samples, n_samples) with precomputed
                distances.

        Returns:
            Sklearn Radius Neighbors estimator initialized.

        """
        return _RadiusNeighborsRegressor(
            radius=self.radius, weights=self.weights,
            algorithm=self.algorithm, leaf_size=self.leaf_size,
            metric=sk_metric, metric_params=self.metric_params,
            n_jobs=self.n_jobs)


class KNeighborsFunctionalRegressor(NearestNeighborsMixinInit,
                                    NeighborsBase, KNeighborsMixin,
                                    NeighborsFunctionalRegressorMixin):
    """Functional regression based on neighbors within a fixed radius.

    The target is predicted by local interpolation of the targets
    associated of the nearest neighbors in the training set.

    Parameters
    ----------
    n_neighbors : int, optional (default = 5)
        Number of neighbors to use by default for :meth:`kneighbors` queries.
    weights : str or callable
        weight function used in prediction.  Possible values:

        - 'uniform' : uniform weights.  All points in each neighborhood
          are weighted equally.
        - 'distance' : weight points by the inverse of their distance.
          in this case, closer neighbors of a query point will have a
          greater influence than neighbors which are further away.
        - [callable] : a user-defined function which accepts an
          array of distances, and returns an array of the same shape
          containing the weights.

        Uniform weights are used by default.
    regressor : callable, optional ((default =
        :func:`mean <skfda.exploratory.stats.mean>`))
        Function to perform the local regression. By default used the mean. Can
        accept a user-defined function wich accepts a :class:`FDataGrid` with
        the neighbors of a test sample, and if weights != 'uniform' an array
        of weights as second parameter.
    algorithm : {'auto', 'ball_tree', 'brute'}, optional
        Algorithm used to compute the nearest neighbors:

        - 'ball_tree' will use :class:`sklearn.neighbors.BallTree`.
        - 'brute' will use a brute-force search.
        - 'auto' will attempt to decide the most appropriate algorithm
          based on the values passed to :meth:`fit` method.

    leaf_size : int, optional (default = 30)
        Leaf size passed to BallTree. This can affect the
        speed of the construction and query, as well as the memory
        required to store the tree. The optimal value depends on the
        nature of the problem.
    metric : string or callable, (default
        :func:`lp_distance <skfda.metrics.lp_distance>`)
        the distance metric to use for the tree.  The default metric is
        the Lp distance. See the documentation of the metrics module
        for a list of available metrics.
    metric_params : dict, optional (default = None)
        Additional keyword arguments for the metric function.
    n_jobs : int or None, optional (default=None)
        The number of parallel jobs to run for neighbors search.
        ``None`` means 1 unless in a :obj:`joblib.parallel_backend` context.
        ``-1`` means using all processors.
    sklearn_metric : boolean, optional (default = False)
        Indicates if the metric used is a sklearn distance between vectors (see
        :class:`sklearn.neighbors.DistanceMetric`) or a functional metric of
        the module :mod:`skfda.misc.metrics`.
    Examples
    --------
    Firstly, we will create a toy dataset with gaussian-like samples shifted,
    and we will try to predict 5 X +1.

    >>> from skfda.datasets import make_multimodal_samples
    >>> X_train = make_multimodal_samples(n_samples=30, std=.05, random_state=0)
    >>> y_train = 5 * X_train + 1
    >>> X_test = make_multimodal_samples(n_samples=5, std=.05, random_state=0)

    We will fit a K-Nearest Neighbors functional regressor.

    >>> from skfda.ml.regression import KNeighborsFunctionalRegressor
    >>> neigh = KNeighborsFunctionalRegressor()
    >>> neigh.fit(X_train, y_train)
    KNeighborsFunctionalRegressor(algorithm='auto', leaf_size=30,...)

    We can predict the response of new samples.

    >>> neigh.predict(X_test)
    FDataGrid(...)

    See also
    --------
    KNeighborsClassifier
    RadiusNeighborsClassifier
    KNeighborsScalarRegressor
    NearestNeighbors
    NearestCentroids
    Notes
    -----
    See Nearest Neighbors in the sklearn online documentation for a discussion
    of the choice of ``algorithm`` and ``leaf_size``.

    This class wraps the sklearn classifier
    `sklearn.neighbors.RadiusNeighborsClassifier`.

    https://en.wikipedia.org/wiki/K-nearest_neighbor_algorithm

    """

    def __init__(self, n_neighbors=5, weights='uniform', regressor=mean,
                 algorithm='auto', leaf_size=30, metric=lp_distance,
                 metric_params=None, n_jobs=1, sklearn_metric=False):
        """Initialize the classifier."""

        super().__init__(n_neighbors=n_neighbors, radius=1.,
                         weights=weights, algorithm=algorithm,
                         leaf_size=leaf_size, metric=metric,
                         metric_params=metric_params, n_jobs=n_jobs,
                         sklearn_metric=sklearn_metric)
        self.regressor = regressor

    def _query(self, X):
        """Return distances and neighbors of given sample"""
        return self.estimator_.kneighbors(X)


class RadiusNeighborsFunctionalRegressor(NearestNeighborsMixinInit,
                                         NeighborsBase, RadiusNeighborsMixin,
                                         NeighborsFunctionalRegressorMixin):
    """Functional regression based on neighbors within a fixed radius.

    The target is predicted by local interpolation of the targets
    associated of the nearest neighbors in the training set.

    Parameters
    ----------
    radius : float, optional (default = 1.0)
        Range of parameter space to use by default for :meth:`radius_neighbors`
        queries.
    weights : str or callable
        weight function used in prediction.  Possible values:

        - 'uniform' : uniform weights.  All points in each neighborhood
          are weighted equally.
        - 'distance' : weight points by the inverse of their distance.
          in this case, closer neighbors of a query point will have a
          greater influence than neighbors which are further away.
        - [callable] : a user-defined function which accepts an
          array of distances, and returns an array of the same shape
          containing the weights.

        Uniform weights are used by default.
    regressor : callable, optional ((default =
        :func:`mean <skfda.exploratory.stats.mean>`))
        Function to perform the local regression. By default used the mean. Can
        accept a user-defined function wich accepts a :class:`FDataGrid` with
        the neighbors of a test sample, and if weights != 'uniform' an array
        of weights as second parameter.
    algorithm : {'auto', 'ball_tree', 'brute'}, optional
        Algorithm used to compute the nearest neighbors:

        - 'ball_tree' will use :class:`sklearn.neighbors.BallTree`.
        - 'brute' will use a brute-force search.
        - 'auto' will attempt to decide the most appropriate algorithm
          based on the values passed to :meth:`fit` method.

    leaf_size : int, optional (default = 30)
        Leaf size passed to BallTree. This can affect the
        speed of the construction and query, as well as the memory
        required to store the tree. The optimal value depends on the
        nature of the problem.
    metric : string or callable, (default
        :func:`lp_distance <skfda.metrics.lp_distance>`)
        the distance metric to use for the tree.  The default metric is
        the Lp distance. See the documentation of the metrics module
        for a list of available metrics.
    metric_params : dict, optional (default = None)
        Additional keyword arguments for the metric function.
    outlier_response : :class:`FDataGrid`, optional (default = None)
        Default response for test samples without neighbors.
    n_jobs : int or None, optional (default=None)
        The number of parallel jobs to run for neighbors search.
        ``None`` means 1 unless in a :obj:`joblib.parallel_backend` context.
        ``-1`` means using all processors.
    sklearn_metric : boolean, optional (default = False)
        Indicates if the metric used is a sklearn distance between vectors (see
        :class:`sklearn.neighbors.DistanceMetric`) or a functional metric of
        the module :mod:`skfda.misc.metrics`.
    Examples
    --------
    Firstly, we will create a toy dataset with gaussian-like samples shifted,
    and we will try to predict the response 5 X +1.

    >>> from skfda.datasets import make_multimodal_samples
    >>> X_train = make_multimodal_samples(n_samples=30, std=.05, random_state=0)
    >>> y_train = 5 * X_train + 1
    >>> X_test = make_multimodal_samples(n_samples=5, std=.05, random_state=0)

    We will fit a Radius Nearest Neighbors functional regressor.

    >>> from skfda.ml.regression import RadiusNeighborsFunctionalRegressor
    >>> neigh = RadiusNeighborsFunctionalRegressor(radius=.03)
    >>> neigh.fit(X_train, y_train)
    RadiusNeighborsFunctionalRegressor(algorithm='auto', leaf_size=30,...)

    We can predict the response of new samples.

    >>> neigh.predict(X_test)
    FDataGrid(...)

    See also
    --------
    KNeighborsClassifier
    RadiusNeighborsClassifier
    KNeighborsScalarRegressor
    NearestNeighbors
    NearestCentroids
    Notes
    -----
    See Nearest Neighbors in the sklearn online documentation for a discussion
    of the choice of ``algorithm`` and ``leaf_size``.

    This class wraps the sklearn classifier
    `sklearn.neighbors.RadiusNeighborsClassifier`.

    https://en.wikipedia.org/wiki/K-nearest_neighbor_algorithm

    """

    def __init__(self, radius=1., weights='uniform', regressor=mean,
                 algorithm='auto', leaf_size=30, metric=lp_distance,
                 metric_params=None, outlier_response=None, n_jobs=1,
                 sklearn_metric=False):
        """Initialize the classifier."""

        super().__init__(n_neighbors=5, radius=radius,
                         weights=weights, algorithm=algorithm,
                         leaf_size=leaf_size, metric=metric,
                         metric_params=metric_params, n_jobs=n_jobs,
                         sklearn_metric=sklearn_metric)
        self.regressor = regressor
        self.outlier_response = outlier_response

    def _query(self, X):
        """Return distances and neighbors of given sample"""
        return self.estimator_.radius_neighbors(X)
