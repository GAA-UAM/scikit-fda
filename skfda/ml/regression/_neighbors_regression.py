"""Neighbor models for regression."""

from sklearn.neighbors import (
    KNeighborsRegressor as _KNeighborsRegressor,
    RadiusNeighborsRegressor as _RadiusNeighborsRegressor,
)

from .._neighbors_base import (
    KNeighborsMixin,
    NeighborsBase,
    NeighborsRegressorMixin,
    RadiusNeighborsMixin,
)


class KNeighborsRegressor(NeighborsBase, NeighborsRegressorMixin,
                          KNeighborsMixin):
    """Regression based on k-nearest neighbors.

    Regression with scalar, multivariate or functional response.

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

    regressor : callable, optional ((default =
        :func:`mean <skfda.exploratory.stats.mean>`))
        Function to perform the local regression in the functional response
        case. By default used the mean. Can the neighbors of a test sample,
        and if weights != 'uniform' an array of weights as second parameter.
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
        :func:`l2_distance <skfda.misc.metrics.l2_distance>`)
        the distance metric to use for the tree.  The default metric is
        the L2 distance. See the documentation of the metrics module
        for a list of available metrics.
    metric_params : dict, optional (default = None)
        Additional keyword arguments for the metric function.
    n_jobs : int or None, optional (default=None)
        The number of parallel jobs to run for neighbors search.
        ``None`` means 1 unless in a :obj:`joblib.parallel_backend` context.
        ``-1`` means using all processors.
        Doesn't affect :meth:`fit` method.
    multivariate_metric : boolean, optional (default = False)
        Indicates if the metric used is a sklearn distance between vectors (see
        :class:`sklearn.neighbors.DistanceMetric`) or a functional metric of
        the module :mod:`skfda.misc.metrics`.
    Examples
    --------
    Firstly, we will create a toy dataset with gaussian-like samples shifted.

    >>> from skfda.ml.regression import KNeighborsRegressor
    >>> from skfda.datasets import make_multimodal_samples
    >>> from skfda.datasets import make_multimodal_landmarks
    >>> y = make_multimodal_landmarks(n_samples=30, std=.5, random_state=0)
    >>> y_train = y.flatten()
    >>> X_train = make_multimodal_samples(n_samples=30, std=.5, random_state=0)
    >>> X_test = make_multimodal_samples(n_samples=5, std=.05, random_state=0)

    We will fit a K-Nearest Neighbors regressor to regress a scalar response.

    >>> neigh = KNeighborsRegressor()
    >>> neigh.fit(X_train, y_train)
    KNeighborsRegressor(...)

    We can predict the modes of new samples

    >>> neigh.predict(X_test).round(2) # Predict test data
    array([ 0.38, 0.14, 0.27, 0.52, 0.38])


    Now we will create a functional response to train the model

    >>> y_train = 5 * X_train + 1
    >>> y_train
    FDataGrid(...)

    We train the estimator with the functional response

    >>> neigh.fit(X_train, y_train)
    KNeighborsRegressor(...)

    And predict the responses as in the first case.

    >>> neigh.predict(X_test)
    FDataGrid(...)

    See also
    --------
    :class:`~skfda.ml.classification.KNeighborsClassifier`
    :class:`~skfda.ml.classification.RadiusNeighborsClassifier`
    :class:`~skfda.ml.classification.NearestCentroids`
    :class:`~skfda.ml.regression.RadiusNeighborsRegressor`
    :class:`~skfda.ml.clustering.NearestNeighbors`


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

    def __init__(self, n_neighbors=5, weights='uniform', regressor='mean',
                 algorithm='auto', leaf_size=30, metric='l2',
                 metric_params=None, n_jobs=1, multivariate_metric=False):
        """Initialize the regressor."""
        super().__init__(n_neighbors=n_neighbors,
                         weights=weights, algorithm=algorithm,
                         leaf_size=leaf_size, metric=metric,
                         metric_params=metric_params, n_jobs=n_jobs,
                         multivariate_metric=multivariate_metric)
        self.regressor = regressor

    def _init_multivariate_estimator(self, sklearn_metric):
        """Initialize the sklearn K neighbors estimator.

        Args:
            sklearn_metric: (pyfunc or 'precomputed'): Metric compatible with
                sklearn API or matrix (n_samples, n_samples) with precomputed
                distances.

        Returns:
            Sklearn K Neighbors estimator initialized.

        """
        return _KNeighborsRegressor(
            n_neighbors=self.n_neighbors, weights=self.weights,
            algorithm=self.algorithm, leaf_size=self.leaf_size,
            metric=sklearn_metric, metric_params=self.metric_params,
            n_jobs=self.n_jobs)

    def _query(self, X):
        """Return distances and neighbors of given sample."""
        return self.estimator_.kneighbors(X)


class RadiusNeighborsRegressor(NeighborsBase, NeighborsRegressorMixin,
                               RadiusNeighborsMixin):
    """Regression based on neighbors within a fixed radius.

    Regression with scalar, multivariate or functional response.

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
    regressor : callable, optional ((default =
        :func:`mean <skfda.exploratory.stats.mean>`))
        Function to perform the local regression in the functional response
        case. By default used the mean. Can the neighbors of a test sample,
        and if weights != 'uniform' an array of weights as second parameter.
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
        :func:`l2_distance <skfda.metrics.l2_distance>`)
        the distance metric to use for the tree.  The default metric is
        the L2 distance. See the documentation of the metrics module
        for a list of available metrics.
    metric_params : dict, optional (default = None)
        Additional keyword arguments for the metric function.
    outlier_response : :class:`FData`, optional (default = None)
        Default response in the functional response case for test samples
        without neighbors.
    n_jobs : int or None, optional (default=None)
        The number of parallel jobs to run for neighbors search.
        ``None`` means 1 unless in a :obj:`joblib.parallel_backend` context.
        ``-1`` means using all processors.
    multivariate_metric : boolean, optional (default = False)
        Indicates if the metric used is a sklearn distance between vectors (see
        :class:`sklearn.neighbors.DistanceMetric`) or a functional metric of
        the module :mod:`skfda.misc.metrics`.

    Examples
    --------
    Firstly, we will create a toy dataset with gaussian-like samples shifted.

    >>> from skfda.ml.regression import RadiusNeighborsRegressor
    >>> from skfda.datasets import make_multimodal_samples
    >>> from skfda.datasets import make_multimodal_landmarks
    >>> y = make_multimodal_landmarks(n_samples=30, std=.5, random_state=0)
    >>> y_train = y.flatten()
    >>> X_train = make_multimodal_samples(n_samples=30, std=.5, random_state=0)
    >>> X_test = make_multimodal_samples(n_samples=5, std=.05, random_state=0)

    We will fit a Radius-Nearest Neighbors regressor to regress a scalar
    response.

    >>> neigh = RadiusNeighborsRegressor(radius=0.2)
    >>> neigh.fit(X_train, y_train)
    RadiusNeighborsRegressor(...radius=0.2...)

    We can predict the modes of new samples

    >>> neigh.predict(X_test).round(2) # Predict test data
    array([ 0.39, 0.07, 0.26, 0.5 , 0.46])


    Now we will create a functional response to train the model

    >>> y_train = 5 * X_train + 1
    >>> y_train
    FDataGrid(...)

    We train the estimator with the functional response

    >>> neigh.fit(X_train, y_train)
    RadiusNeighborsRegressor(...radius=0.2...)

    And predict the responses as in the first case.

    >>> neigh.predict(X_test)
    FDataGrid(...)

    See also
    --------
    :class:`~skfda.ml.classification.KNeighborsClassifier`
    :class:`~skfda.ml.classification.RadiusNeighborsClassifier`
    :class:`~skfda.ml.classification.NearestCentroids`
    :class:`~skfda.ml.regression.KNeighborsRegressor`
    :class:`~skfda.ml.clustering.NearestNeighbors`


    Notes
    -----
    See Nearest Neighbors in the sklearn online documentation for a discussion
    of the choice of ``algorithm`` and ``leaf_size``.

    This class wraps the sklearn classifier
    `sklearn.neighbors.RadiusNeighborsClassifier`.

    https://en.wikipedia.org/wiki/K-nearest_neighbor_algorithm

    """

    def __init__(self, radius=1.0, weights='uniform', regressor='mean',
                 algorithm='auto', leaf_size=30, metric='l2',
                 metric_params=None, outlier_response=None, n_jobs=1,
                 multivariate_metric=False):
        """Initialize the classifier."""
        super().__init__(radius=radius, weights=weights, algorithm=algorithm,
                         leaf_size=leaf_size, metric=metric,
                         metric_params=metric_params, n_jobs=n_jobs,
                         multivariate_metric=multivariate_metric)
        self.regressor = regressor
        self.outlier_response = outlier_response

    def _init_multivariate_estimator(self, sklearn_metric):
        """Initialize the sklearn radius neighbors estimator.

        Args:
            sklearn_metric: (pyfunc or 'precomputed'): Metric compatible with
                sklearn API or matrix (n_samples, n_samples) with precomputed
                distances.

        Returns:
            Sklearn Radius Neighbors estimator initialized.

        """
        return _RadiusNeighborsRegressor(
            radius=self.radius, weights=self.weights,
            algorithm=self.algorithm, leaf_size=self.leaf_size,
            metric=sklearn_metric, metric_params=self.metric_params,
            n_jobs=self.n_jobs)

    def _query(self, X):
        """Return distances and neighbors of given sample.

        Args:
            X: the sample

        Returns:
            Distances and neighbors of a given sample

        """
        return self.estimator_.radius_neighbors(X)
