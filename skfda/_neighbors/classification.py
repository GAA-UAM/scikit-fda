"""Neighbor models for supervised classification."""


from sklearn.utils.multiclass import check_classification_targets
from sklearn.preprocessing import LabelEncoder
from sklearn.base import ClassifierMixin, BaseEstimator
from sklearn.utils.validation import check_is_fitted as sklearn_check_is_fitted

from ..misc.metrics import lp_distance, pairwise_distance
from ..exploratory.stats import mean as l2_mean
from .base import (NeighborsBase, NeighborsMixin, KNeighborsMixin,
                   NeighborsClassifierMixin, RadiusNeighborsMixin)


class KNeighborsClassifier(NeighborsBase, NeighborsMixin, KNeighborsMixin,
                           ClassifierMixin, NeighborsClassifierMixin):
    """Classifier implementing the k-nearest neighbors vote.

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
    Firstly, we will create a toy dataset with 2 classes

    >>> from skfda.datasets import make_sinusoidal_process
    >>> fd1 = make_sinusoidal_process(phase_std=.25, random_state=0)
    >>> fd2 = make_sinusoidal_process(phase_mean=1.8, error_std=0.,
    ...                               phase_std=.25, random_state=0)
    >>> fd = fd1.concatenate(fd2)
    >>> y = 15*[0] + 15*[1]

    We will fit a K-Nearest Neighbors classifier

    >>> from skfda.ml.classification import KNeighborsClassifier
    >>> neigh = KNeighborsClassifier()
    >>> neigh.fit(fd, y)
    KNeighborsClassifier(algorithm='auto', leaf_size=30,...)

    We can predict the class of new samples

    >>> neigh.predict(fd[::2]) # Predict labels for even samples
    array([0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1])

    And the estimated probabilities.

    >>> neigh.predict_proba(fd[0]) # Probabilities of sample 0
    array([[ 1.,  0.]])

    See also
    --------
    :class:`~skfda.ml.classification.RadiusNeighborsClassifier`
    :class:`~skfda.ml.classification.NearestCentroids`
    :class:`~skfda.ml.regression.KNeighborsRegressor`
    :class:`~skfda.ml.regression.RadiusNeighborsRegressor`
    :class:`~skfda.ml.clustering.NearestNeighbors`

    Notes
    -----
    See Nearest Neighbors in the sklearn online documentation for a discussion
    of the choice of ``algorithm`` and ``leaf_size``.

    This class wraps the sklearn classifier
    `sklearn.neighbors.KNeighborsClassifier`.

    .. warning::
       Regarding the Nearest Neighbors algorithms, if it is found that two
       neighbors, neighbor `k+1` and `k`, have identical distances
       but different labels, the results will depend on the ordering of the
       training data.

    https://en.wikipedia.org/wiki/K-nearest_neighbor_algorithm

    """

    def __init__(self, n_neighbors=5, weights='uniform', algorithm='auto',
                 leaf_size=30, metric='l2', metric_params=None,
                 n_jobs=1, multivariate_metric=False):
        """Initialize the classifier."""

        super().__init__(n_neighbors=n_neighbors,
                         weights=weights, algorithm=algorithm,
                         leaf_size=leaf_size, metric=metric,
                         metric_params=metric_params, n_jobs=n_jobs,
                         multivariate_metric=multivariate_metric)

    def _init_estimator(self, sklearn_metric):
        """Initialize the sklearn K neighbors estimator.

        Args:
            sklearn_metric: (pyfunc or 'precomputed'): Metric compatible with
                sklearn API or matrix (n_samples, n_samples) with precomputed
                distances.

        Returns:
            Sklearn K Neighbors estimator initialized.

        """
        from sklearn.neighbors import (KNeighborsClassifier as
                                       _KNeighborsClassifier)

        return _KNeighborsClassifier(
            n_neighbors=self.n_neighbors, weights=self.weights,
            algorithm=self.algorithm, leaf_size=self.leaf_size,
            metric=sklearn_metric, metric_params=self.metric_params,
            n_jobs=self.n_jobs)

    def predict_proba(self, X):
        """Return probability estimates for the test data X.

        Args:
            X (:class:`FDataGrid` or array-like): FDataGrid with the test
                samples or array (n_query, n_indexed) if metric ==
                'precomputed'.
        Returns
            p : array of shape = [n_samples, n_classes], or a list of n_outputs
                of such arrays if n_outputs > 1.
                The class probabilities of the input samples. Classes are
                ordered by lexicographic order.

        """
        self._check_is_fitted()

        X = self._transform_to_multivariate(X)

        return self.estimator_.predict_proba(X)


class RadiusNeighborsClassifier(NeighborsBase, NeighborsMixin,
                                RadiusNeighborsMixin, ClassifierMixin,
                                NeighborsClassifierMixin):
    """Classifier implementing a vote among neighbors within a given radius.

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
        the L2 distance. See the documentation of the metrics module
        for a list of available metrics.
    outlier_label : int, optional (default = None)
        Label, which is given for outlier samples (samples with no
        neighbors on given radius).
        If set to None, ValueError is raised, when outlier is detected.
    metric_params : dict, optional (default = None)
        Additional keyword arguments for the metric function.
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
    Firstly, we will create a toy dataset with 2 classes.

    >>> from skfda.datasets import make_sinusoidal_process
    >>> fd1 = make_sinusoidal_process(phase_std=.25, random_state=0)
    >>> fd2 = make_sinusoidal_process(phase_mean=1.8, error_std=0.,
    ...                               phase_std=.25, random_state=0)
    >>> fd = fd1.concatenate(fd2)
    >>> y = 15*[0] + 15*[1]

    We will fit a Radius Nearest Neighbors classifier.

    >>> from skfda.ml.classification import RadiusNeighborsClassifier
    >>> neigh = RadiusNeighborsClassifier(radius=.3)
    >>> neigh.fit(fd, y)
    RadiusNeighborsClassifier(algorithm='auto', leaf_size=30,...)

    We can predict the class of new samples.

    >>> neigh.predict(fd[::2]) # Predict labels for even samples
    array([0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1])

    See also
    --------
    :class:`~skfda.ml.classification.KNeighborsClassifier`
    :class:`~skfda.ml.classification.NearestCentroids`
    :class:`~skfda.ml.regression.KNeighborsRegressor`
    :class:`~skfda.ml.regression.RadiusNeighborsRegressor`
    :class:`~skfda.ml.clustering.NearestNeighbors`

    Notes
    -----
    See Nearest Neighbors in the sklearn online documentation for a discussion
    of the choice of ``algorithm`` and ``leaf_size``.

    This class wraps the sklearn classifier
    `sklearn.neighbors.RadiusNeighborsClassifier`.

    https://en.wikipedia.org/wiki/K-nearest_neighbor_algorithm

    """

    def __init__(self, radius=1.0, weights='uniform', algorithm='auto',
                 leaf_size=30, metric='l2', metric_params=None,
                 outlier_label=None, n_jobs=1, multivariate_metric=False):
        """Initialize the classifier."""

        super().__init__(radius=radius, weights=weights, algorithm=algorithm,
                         leaf_size=leaf_size, metric=metric,
                         metric_params=metric_params, n_jobs=n_jobs,
                         multivariate_metric=multivariate_metric)

        self.outlier_label = outlier_label

    def _init_estimator(self, sklearn_metric):
        """Initialize the sklearn radius neighbors estimator.

        Args:
            sklearn_metric: (pyfunc or 'precomputed'): Metric compatible with
                sklearn API or matrix (n_samples, n_samples) with precomputed
                distances.

        Returns:
            Sklearn Radius Neighbors estimator initialized.

        """
        from sklearn.neighbors import (RadiusNeighborsClassifier as
                                       _RadiusNeighborsClassifier)

        return _RadiusNeighborsClassifier(
            radius=self.radius, weights=self.weights,
            algorithm=self.algorithm, leaf_size=self.leaf_size,
            metric=sklearn_metric, metric_params=self.metric_params,
            outlier_label=self.outlier_label, n_jobs=self.n_jobs)


class NearestCentroids(BaseEstimator, ClassifierMixin):
    """Nearest centroid classifier for functional data.

    Each class is represented by its centroid, with test samples classified to
    the class with the nearest centroid.

    Parameters
    ----------
        metric : callable, (default
            :func:`lp_distance <skfda.metrics.lp_distance>`)
            The metric to use when calculating distance between test samples
            and centroids. See the documentation of the metrics module
            for a list of available metrics. Defaults used L2 distance.
        centroid: callable, (default
            :func:`mean <skfda.exploratory.stats.mean>`)
            The centroids for the samples corresponding to each class is the
            point from which the sum of the distances (according to the metric)
            of all samples that belong to that particular class are minimized.
            By default it is used the usual mean, which minimizes the sum of L2
            distances. This parameter allows change the centroid constructor.
            The function must accept a :class:`FData` with the samples of one
            class and return a :class:`FData` object with only one sample
            representing the centroid.
    Attributes
    ----------
    centroids_ : :class:`FDataGrid`
        FDatagrid containing the centroid of each class
    Examples
    --------
    Firstly, we will create a toy dataset with 2 classes

    >>> from skfda.datasets import make_sinusoidal_process
    >>> fd1 = make_sinusoidal_process(phase_std=.25, random_state=0)
    >>> fd2 = make_sinusoidal_process(phase_mean=1.8, error_std=0.,
    ...                               phase_std=.25, random_state=0)
    >>> fd = fd1.concatenate(fd2)
    >>> y = 15*[0] + 15*[1]

    We will fit a Nearest centroids classifier

    >>> from skfda.ml.classification import NearestCentroids
    >>> neigh = NearestCentroids()
    >>> neigh.fit(fd, y)
    NearestCentroids(...)

    We can predict the class of new samples

    >>> neigh.predict(fd[::2]) # Predict labels for even samples
    array([0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1])

    See also
    --------
    :class:`~skfda.ml.classification.KNeighborsClassifier`
    :class:`~skfda.ml.classification.RadiusNeighborsClassifier`
    :class:`~skfda.ml.regression.KNeighborsRegressor`
    :class:`~skfda.ml.regression.RadiusNeighborsRegressor`
    :class:`~skfda.ml.clustering.NearestNeighbors`

    """

    def __init__(self, metric='l2', mean='mean'):
        """Initialize the classifier."""
        self.metric = metric
        self.mean = mean

    def fit(self, X, y):
        """Fit the model using X as training data and y as target values.

        Args:
            X (:class:`FDataGrid`, array_matrix): Training data. FDataGrid
                with the training data or array matrix with shape
                [n_samples, n_samples] if metric='precomputed'.
            y (array-like or sparse matrix): Target values of
                shape = [n_samples] or [n_samples, n_outputs].

        """
        if self.metric == 'precomputed':
            raise ValueError("Precomputed is not supported.")
        elif self.metric == 'l2':
            self._pairwise_distance = pairwise_distance(lp_distance)
        else:
            self._pairwise_distance = pairwise_distance(self.metric)

        mean = l2_mean if self.mean == 'mean' else self.mean

        check_classification_targets(y)

        le = LabelEncoder()
        y_ind = le.fit_transform(y)
        self.classes_ = classes = le.classes_
        n_classes = classes.size
        if n_classes < 2:
            raise ValueError(f'The number of classes has to be greater than'
                             f' one; got {n_classes} class')

        self.centroids_ = mean(X[y_ind == 0])

        for cur_class in range(1, n_classes):
            center_mask = y_ind == cur_class
            centroid = mean(X[center_mask])
            self.centroids_ = self.centroids_.concatenate(centroid)

        return self

    def predict(self, X):
        """Predict the class labels for the provided data.

        Args:
            X (:class:`FDataGrid`): FDataGrid with the test samples.

        Returns:

            (np.array): y : array of shape [n_samples] or
            [n_samples, n_outputs] with class labels for each data sample.

        """
        sklearn_check_is_fitted(self, 'centroids_')

        return self.classes_[self._pairwise_distance(
            X, self.centroids_).argmin(axis=1)]
