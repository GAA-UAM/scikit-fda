"""Neighbor models for supervised classification."""

from sklearn.base import ClassifierMixin
from sklearn.neighbors import (
    KNeighborsClassifier as _KNeighborsClassifier,
    RadiusNeighborsClassifier as _RadiusNeighborsClassifier,
)

from .._neighbors_base import (
    KNeighborsMixin,
    NeighborsBase,
    NeighborsClassifierMixin,
    NeighborsMixin,
    RadiusNeighborsMixin,
)


class KNeighborsClassifier(
    NeighborsBase,
    NeighborsMixin,
    KNeighborsMixin,
    ClassifierMixin,
    NeighborsClassifierMixin,
):
    """Classifier implementing the k-nearest neighbors vote.

    Parameters:
        n_neighbors (int, default = 5):
            Number of neighbors to use by default for :meth:`kneighbors`
            queries.
        weights (str or callable, default = 'uniform'):
            Weight function used in prediction.
            Possible values:
            - 'uniform': uniform weights. All points in each neighborhood
            are weighted equally.
            - 'distance': weight points by the inverse of their distance.
            in this case, closer neighbors of a query point will have a
            greater influence than neighbors which are further away.
            - [callable]: a user-defined function which accepts an
            array of distances, and returns an array of the same shape
            containing the weights.
        algorithm (string, optional):
            Algorithm used to compute the nearest neighbors:
            - 'ball_tree' will use :class:`sklearn.neighbors.BallTree`.
            - 'brute' will use a brute-force search.
            - 'auto' will attempt to decide the most appropriate algorithm
            based on the values passed to :meth:`fit` method.
        leaf_size (int, default = 30):
            Leaf size passed to BallTree or KDTree. This can affect the
            speed of the construction and query, as well as the memory
            required to store the tree. The optimal value depends on the
            nature of the problem.
        metric (string or callable, default
            :func:`l2_distance <skfda.misc.metrics.l2_distance>`):
            the distance metric to use for the tree. The default metric is
            the L2 distance. See the documentation of the metrics module
            for a list of available metrics.
        metric_params (dict, optional):
            Additional keyword arguments for the metric function.
        n_jobs (int or None, optional):
            The number of parallel jobs to run for neighbors search.
            ``None`` means 1 unless in a :obj:`joblib.parallel_backend`
            context.
            ``-1`` means using all processors.
            Doesn't affect :meth:`fit` method.
        multivariate_metric (boolean, default = False):
            Indicates if the metric used is a sklearn distance between vectors
            (see :class:`~sklearn.neighbors.DistanceMetric`) or a functional
            metric of the module `skfda.misc.metrics` if ``False``.

    Examples:
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
        KNeighborsClassifier(...)

        We can predict the class of new samples

        >>> neigh.predict(fd[::2]) # Predict labels for even samples
        array([0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1])

        And the estimated probabilities.

        >>> neigh.predict_proba(fd[0]) # Probabilities of sample 0
        array([[ 1.,  0.]])

    See also:
        :class:`~skfda.ml.classification.RadiusNeighborsClassifier`
        :class:`~skfda.ml.classification.NearestCentroid`
        :class:`~skfda.ml.regression.KNeighborsRegressor`
        :class:`~skfda.ml.regression.RadiusNeighborsRegressor`
        :class:`~skfda.ml.clustering.NearestNeighbors`

    Notes:
        See Nearest Neighbors in the sklearn online documentation for a
        discussion of the choice of ``algorithm`` and ``leaf_size``.

        This class wraps the sklearn classifier
        `sklearn.neighbors.KNeighborsClassifier`.

    Warning:
        Regarding the Nearest Neighbors algorithms, if it is found that two
        neighbors, neighbor `k+1` and `k`, have identical distances
        but different labels, the results will depend on the ordering of the
        training data.

        https://en.wikipedia.org/wiki/K-nearest_neighbor_algorithm
    """

    def __init__(
        self,
        n_neighbors=5,
        weights='uniform',
        algorithm='auto',
        leaf_size=30,
        metric='l2',
        metric_params=None,
        n_jobs=1,
        multivariate_metric=False,
    ):
        super().__init__(
            n_neighbors=n_neighbors,
            weights=weights,
            algorithm=algorithm,
            leaf_size=leaf_size,
            metric=metric,
            metric_params=metric_params,
            n_jobs=n_jobs,
            multivariate_metric=multivariate_metric,
        )

    def predict_proba(self, X):
        """Calculate probability estimates for the test data X.

        Args:
            X (:class:`FDataGrid` or array-like): FDataGrid with the test
                samples or array (n_query, n_indexed) if metric ==
                'precomputed'.

        Returns:
            p (array of shape = (n_samples, n_classes), or a list of n_outputs
                of such arrays if n_outputs > 1):
                The class probabilities of the input samples. Classes are
                ordered by lexicographic order.
        """
        self._check_is_fitted()

        X = self._transform_to_multivariate(X)

        return self.estimator_.predict_proba(X)

    def _init_estimator(self, sklearn_metric):
        """Initialize the sklearn K neighbors estimator.

        Args:
            sklearn_metric (pyfunc or 'precomputed'): Metric compatible with
                sklearn API or matrix (n_samples, n_samples) with precomputed
                distances.

        Returns:
            Sklearn K Neighbors estimator initialized.
        """
        return _KNeighborsClassifier(
            n_neighbors=self.n_neighbors,
            weights=self.weights,
            algorithm=self.algorithm,
            leaf_size=self.leaf_size,
            metric=sklearn_metric,
            metric_params=self.metric_params,
            n_jobs=self.n_jobs,
        )


class RadiusNeighborsClassifier(
    NeighborsBase,
    NeighborsMixin,
    RadiusNeighborsMixin,
    ClassifierMixin,
    NeighborsClassifierMixin,
):
    """Classifier implementing a vote among neighbors within a given radius.

    Parameters:
        radius (float, default = 1.0):
            Range of parameter space to use by default for
            :meth:`radius_neighbors` queries.
        weights (str or callable, default = 'uniform'):
            Weight function used in prediction.
            Possible values:
            - 'uniform': uniform weights. All points in each neighborhood
            are weighted equally.
            - 'distance': weight points by the inverse of their distance.
            in this case, closer neighbors of a query point will have a
            greater influence than neighbors which are further away.
            - [callable]: a user-defined function which accepts an
            array of distances, and returns an array of the same shape
            containing the weights.
        algorithm (string, optional):
            Algorithm used to compute the nearest neighbors:
            - 'ball_tree' will use :class:`sklearn.neighbors.BallTree`.
            - 'brute' will use a brute-force search.
            - 'auto' will attempt to decide the most appropriate algorithm
            based on the values passed to :meth:`fit` method.
        leaf_size (int, default = 30):
            Leaf size passed to BallTree or KDTree. This can affect the
            speed of the construction and query, as well as the memory
            required to store the tree. The optimal value depends on the
            nature of the problem.
        metric (string or callable, default
            :func:`l2_distance <skfda.misc.metrics.l2_distance>`):
            the distance metric to use for the tree. The default metric is
            the L2 distance. See the documentation of the metrics module
            for a list of available metrics.
        outlier_label (int, optional):
            Label, which is given for outlier samples (samples with no
            neighbors on given radius).
            If set to None, ValueError is raised, when outlier is detected.
        metric_params (dict, optional):
            Additional keyword arguments for the metric function.
        n_jobs (int or None, optional):
            The number of parallel jobs to run for neighbors search.
            ``None`` means 1 unless in a :obj:`joblib.parallel_backend`
            context.
            ``-1`` means using all processors.
        multivariate_metric (boolean, default = False):
            Indicates if the metric used is a sklearn distance between vectors
            (see :class:`~sklearn.neighbors.DistanceMetric`) or a functional
            metric of the module `skfda.misc.metrics` if ``False``.

    Examples:
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
        RadiusNeighborsClassifier(...radius=0.3...)

        We can predict the class of new samples.

        >>> neigh.predict(fd[::2]) # Predict labels for even samples
        array([0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1])

    See also:
        :class:`~skfda.ml.classification.KNeighborsClassifier`
        :class:`~skfda.ml.classification.NearestCentroid`
        :class:`~skfda.ml.regression.KNeighborsRegressor`
        :class:`~skfda.ml.regression.RadiusNeighborsRegressor`
        :class:`~skfda.ml.clustering.NearestNeighbors`

    Notes:
        See Nearest Neighbors in the sklearn online documentation for a
        discussion of the choice of ``algorithm`` and ``leaf_size``.

        This class wraps the sklearn classifier
        `sklearn.neighbors.RadiusNeighborsClassifier`.

        https://en.wikipedia.org/wiki/K-nearest_neighbor_algorithm
    """

    def __init__(
        self,
        radius=1.0,
        weights='uniform',
        algorithm='auto',
        leaf_size=30,
        metric='l2',
        metric_params=None,
        outlier_label=None,
        n_jobs=1,
        multivariate_metric=False,
    ):
        super().__init__(
            radius=radius,
            weights=weights,
            algorithm=algorithm,
            leaf_size=leaf_size,
            metric=metric,
            metric_params=metric_params,
            n_jobs=n_jobs,
            multivariate_metric=multivariate_metric,
        )

        self.outlier_label = outlier_label

    def _init_estimator(self, sklearn_metric):
        """Initialize the sklearn radius neighbors estimator.

        Args:
            sklearn_metric (pyfunc or 'precomputed'): Metric compatible with
                sklearn API or matrix (n_samples, n_samples) with precomputed
                distances.

        Returns:
            Sklearn Radius Neighbors estimator initialized.
        """
        return _RadiusNeighborsClassifier(
            radius=self.radius,
            weights=self.weights,
            algorithm=self.algorithm,
            leaf_size=self.leaf_size,
            metric=sklearn_metric,
            metric_params=self.metric_params,
            outlier_label=self.outlier_label,
            n_jobs=self.n_jobs,
        )
