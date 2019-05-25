"""Module with classes to neighbors classification and regression."""

from abc import ABCMeta, abstractmethod, abstractproperty

from sklearn.base import BaseEstimator, ClassifierMixin, RegressorMixin
from sklearn.utils.validation import check_is_fitted as sklearn_check_is_fitted
from sklearn.utils.multiclass import check_classification_targets
from sklearn.preprocessing import LabelEncoder

# Sklearn classes to be wrapped
from sklearn.neighbors import NearestNeighbors as _NearestNeighbors
from sklearn.neighbors import KNeighborsClassifier as _KNeighborsClassifier
from sklearn.neighbors import (RadiusNeighborsClassifier as
                               _RadiusNeighborsClassifier)
from sklearn.neighbors import KNeighborsRegressor as _KNeighborsRegressor
from sklearn.neighbors import (RadiusNeighborsRegressor as
                               _RadiusNeighborsRegressor)

from .. import FDataGrid
from ..misc.metrics import lp_distance, pairwise_distance
from ..exploratory.stats import mean


def _to_multivariate(fdatagrid):
    r"""Returns the data matrix of a fdatagrid in flatten form compatible with
    sklearn.

    Args:
        fdatagrid (:class:`FDataGrid`): Grid to be converted to matrix

    Returns:
        (np.array): Numpy array with size (nsamples, points), where
            points = prod([len(d) for d in fdatagrid.sample_points]

    """
    return fdatagrid.data_matrix.reshape(fdatagrid.nsamples, -1)


def _from_multivariate(data_matrix, sample_points, shape, **kwargs):
    r"""Constructs a FDatagrid from the data matrix flattened.

    Args:
        data_matrix (np.array): Data Matrix flattened as multivariate vector
            compatible with sklearn.
        sample_points (array_like): List with sample points for each dimension.
        shape (tuple): Shape of the data_matrix.
        **kwargs: Named params to be passed to the FDataGrid constructor.

    Returns:
        (:class:`FDataGrid`): FDatagrid with the data.

    """
    return FDataGrid(data_matrix.reshape(shape), sample_points, **kwargs)


def _to_sklearn_metric(metric, sample_points):
    r"""Transform a metric between FDatagrid in a sklearn compatible one.

    Given a metric between FDatagrids returns a compatible metric used to
    wrap the sklearn routines.

    Args:
        metric (pyfunc): Metric of the module `mics.metrics`. Must accept
            two FDataGrids and return a float representing the distance.
        sample_points (array_like): Array of arrays with the sample points of
            the FDataGrids.
        check (boolean, optional): If False it is passed the named parameter
            `check=False` to avoid the repetition of checks in internal
            routines.

    Returns:
        (pyfunc): sklearn vector metric.

    Examples:

        >>> import numpy as np
        >>> from skfda import FDataGrid
        >>> from skfda.misc.metrics import lp_distance
        >>> from skfda.ml._neighbors import _to_sklearn_metric

        Calculate the Lp distance between fd and fd2.
        >>> x = np.linspace(0, 1, 101)
        >>> fd = FDataGrid([np.ones(len(x))], x)
        >>> fd2 =  FDataGrid([np.zeros(len(x))], x)
        >>> lp_distance(fd, fd2).round(2)
        1.0

        Creation of the sklearn-style metric.
        >>> sklearn_lp_distance = _to_sklearn_metric(lp_distance, [x])
        >>> sklearn_lp_distance(np.ones(len(x)), np.zeros(len(x))).round(2)
        1.0

    """
    # Shape -> (Nsamples = 1, domain_dims...., image_dimension (-1))
    shape = [1] + [len(axis) for axis in sample_points] + [-1]

    def sklearn_metric(x, y, check=True, **kwargs):

        return metric(_from_multivariate(x, sample_points, shape),
                      _from_multivariate(y, sample_points, shape),
                      check=check, **kwargs)

    return sklearn_metric


class NeighborsBase(BaseEstimator, metaclass=ABCMeta):
    """Base class for nearest neighbors estimators."""

    @abstractmethod
    def __init__(self, n_neighbors=None, radius=None,
                 weights='uniform', algorithm='auto',
                 leaf_size=30, metric=lp_distance, metric_params=None,
                 n_jobs=None):

        self.n_neighbors = n_neighbors
        self.radius = radius
        self.weights = weights
        self.algorithm = algorithm
        self.leaf_size = leaf_size
        self.metric = metric
        self.metric_params = metric_params
        self.n_jobs = n_jobs

    @abstractmethod
    def _init_estimator(self, sk_metric):
        """Initializes the estimator returned by :meth:`_sklearn_neighbors`."""
        pass

    def fit(self, X, y):
        """Fit the model using X as training data and y as target values.

        Args:
            X (:class:`FDataGrid`, array_matrix): Training data. FDataGrid
                with the training data or array matrix with shape
                [n_samples, n_samples] if metric='precomputed'.
            y (array-like or sparse matrix): Target values of
                shape = [n_samples] or [n_samples, n_outputs].

        Note:
            This method wraps the corresponding sklearn routine in the module
            ``sklearn.neighbors``.

        """
        # If metric is precomputed no different with the Sklearn stimator
        if self.metric == 'precomputed':
            self.estimator_ = self._init_estimator(self.metric)
            self.estimator_.fit(X, y)
        else:
            self._sample_points = X.sample_points
            self._shape = X.data_matrix.shape[1:]

            # Constructs sklearn metric to manage vector instead of FDatagrids
            sk_metric = _to_sklearn_metric(self.metric, self._sample_points)

            self.estimator_ = self._init_estimator(sk_metric)
            self.estimator_.fit(self._transform_to_multivariate(X), y)

        return self

    def _check_is_fitted(self):
        """Check if the estimator is fitted.

        Raises:
            NotFittedError: If the estimator is not fitted.

        """
        sklearn_check_is_fitted(self, ['estimator_'])

    def _transform_to_multivariate(self, X):
        """Transform the input data to array form. If the metric is
        precomputed it is not transformet.

        """
        if X is not None and self.metric != 'precomputed':
            X = _to_multivariate(X)

        return X

    def _transform_from_multivariate(self, X):
        """Transform from array like to FDatagrid."""

        if X.ndim == 1:
            shape = (1, ) + self._shape
        else:
            shape = (len(X), ) + self._shape

        return _from_multivariate(X, self._sample_points, shape)


class KNeighborsMixin:
    """Mixin class for K-Neighbors"""

    def kneighbors(self, X=None, n_neighbors=None, return_distance=True):
        """Finds the K-neighbors of a point.
        Returns indices of and distances to the neighbors of each point.

        Args:
            X (:class:`FDataGrid` or matrix): FDatagrid with the query functions
                or  matrix (n_query, n_indexed) if metric == 'precomputed'. If
                not provided, neighbors of each indexed point are returned. In
                this case, the query point is not considered its own neighbor.
            n_neighbors (int): Number of neighbors to get (default is the value
                passed to the constructor).
        return_distance (boolean, optional): Defaults to True. If False,
            distances will not be returned.

        Returns:
            dist : array
                Array representing the lengths to points, only present if
                return_distance=True
            ind : array
                Indices of the nearest points in the population matrix.

        Notes:
            This method wraps the corresponding sklearn routine in the
            module ``sklearn.neighbors``.

        """
        self._check_is_fitted()
        X = self._transform_to_multivariate(X)

        return self.estimator_.kneighbors(X, n_neighbors, return_distance)

    def kneighbors_graph(self, X=None, n_neighbors=None, mode='connectivity'):
        """Computes the (weighted) graph of k-Neighbors for points in X

        Args:
            X (:class:`FDataGrid` or matrix): FDatagrid with the query functions
                or  matrix (n_query, n_indexed) if metric == 'precomputed'. If
                not provided, neighbors of each indexed point are returned. In
                this case, the query point is not considered its own neighbor.
            n_neighbors (int): Number of neighbors to get (default is the value
                passed to the constructor).
            mode ('connectivity' or 'distance', optional): Type of returned
                matrix: 'connectivity' will return the connectivity matrix with
                ones and zeros, in 'distance' the edges are distance between
                points.

        Returns:
            Sparse matrix in CSR format, shape = [n_samples, n_samples_fit]
            n_samples_fit is the number of samples in the fitted data
            A[i, j] is assigned the weight of edge that connects i to j.

        See also:
            NearestNeighbors.radius_neighbors_graph

        Notes:
            This method wraps the corresponding sklearn routine in the
            module ``sklearn.neighbors``.

        """
        self._check_is_fitted()

        X = self._transform_to_multivariate(X)

        return self.estimator_.kneighbors_graph(X, n_neighbors, mode)


class RadiusNeighborsMixin:
    """Mixin Class for Raius Neighbors"""

    def radius_neighbors(self, X=None, radius=None, return_distance=True):
        """Finds the neighbors within a given radius of a fdatagrid or
        fdatagrids.
        Return the indices and distances of each point from the dataset
        lying in a ball with size ``radius`` around the points of the query
        array. Points lying on the boundary are included in the results.
        The result points are *not* necessarily sorted by distance to their
        query point.

        Args:
            X (:class:`FDataGrid`, optional): fdatagrid with the sample or
                samples whose neighbors will be returned. If not provided,
                neighbors of each indexed point are returned. In this case, the
                query point is not considered its own neighbor.
            radius (float, optional): Limiting distance of neighbors to return.
                (default is the value passed to the constructor).
            return_distance (boolean, optional). Defaults to True. If False,
                distances will not be returned

        Returns
            (array, shape (n_samples): dist : array of arrays representing the
                distances to each point, only present if return_distance=True.
                The distance values are computed according to the ``metric``
                constructor parameter.
            (array, shape (n_samples,): An array of arrays of indices of the
                approximate nearest points from the population matrix that lie
                within a ball of size ``radius`` around the query points.

        See also:
            kneighbors

        Notes:

            Because the number of neighbors of each point is not necessarily
            equal, the results for multiple query points cannot be fit in a
            standard data array.
            For efficiency, `radius_neighbors` returns arrays of objects, where
            each object is a 1D array of indices or distances.

            This method wraps the corresponding sklearn routine in the module
            ``sklearn.neighbors``.

        """
        self._check_is_fitted()

        X = self._transform_to_multivariate(X)

        return self.estimator_.radius_neighbors(X=X, radius=radius,
                                                return_distance=return_distance)

    def radius_neighbors_graph(self, X=None, radius=None, mode='connectivity'):
        """Computes the (weighted) graph of Neighbors for points in X
        Neighborhoods are restricted the points at a distance lower than
        radius.

        Args:
            X  (:class:`FDataGrid`):  The query sample or samples. If not
                provided, neighbors of each indexed point are returned. In this
                case, the query point is not considered its own neighbor.
            radius (float): Radius of neighborhoods. (default is the value
                passed to the constructor).
            mode ('connectivity' or 'distance', optional): Type of returned
                matrix: 'connectivity' will return the connectivity matrix with
                ones and zeros, in 'distance' the edges are distance between
                points.

        Returns:
            sparse matrix in CSR format, shape = [n_samples, n_samples]
            A[i, j] is assigned the weight of edge that connects i to j.

        See also:
            kneighbors_graph

        Notes:
            This method wraps the corresponding sklearn routine in the module
            ``sklearn.neighbors``.
        """
        self._check_is_fitted()

        X = self._transform_to_multivariate(X)

        return self.estimator_.radius_neighbors_graph(X=X, radius=radius,
                                                      mode=mode)


class NeighborsClassifierMixin:
    """Mixin class for classifiers based in nearest neighbors"""

    def predict(self, X):
        """Predict the class labels for the provided data.

        Args:
            X (:class:`FDataGrid` or array-like): FDataGrid with the test
                samples or array (n_query, n_indexed) if metric ==
                'precomputed'.

        Returns:

            (np.array): y : array of shape [n_samples] or
            [n_samples, n_outputs] with class labels for each data sample.

        Notes:
            This method wraps the corresponding sklearn routine in the module
            ``sklearn.neighbors``.

        """
        self._check_is_fitted()

        X = self._transform_to_multivariate(X)

        return self.estimator_.predict(X)

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

class NeighborsScalarRegresorMixin:
    """Mixin class for scalar regressor based in nearest neighbors"""

    def predict(self, X):
        """Predict the target for the provided data
        Parameters
        ----------
        X (:class:`FDataGrid` or array-like): FDataGrid with the test
            samples or array (n_query, n_indexed) if metric ==
            'precomputed'.
        Returns
        -------
        y : array of int, shape = [n_samples] or [n_samples, n_outputs]
            Target values
        Notes
        -----
        This method wraps the corresponding sklearn routine in the module
        ``sklearn.neighbors``.

        """
        self._check_is_fitted()

        X = self._transform_to_multivariate(X)

        return self.estimator_.predict(X)


class NearestNeighbors(NeighborsBase, KNeighborsMixin, RadiusNeighborsMixin):
    """Unsupervised learner for implementing neighbor searches.

    Parameters
    ----------
    n_neighbors : int, optional (default = 5)
        Number of neighbors to use by default for :meth:`kneighbors` queries.
    radius : float, optional (default = 1.0)
        Range of parameter space to use by default for :meth:`radius_neighbors`
        queries.
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
    See also
    --------
    KNeighborsClassifier
    RadiusNeighborsClassifier
    KNeighborsScalarRegressor
    RadiusNeighborsScalarRegressor
    NearestCentroids
    Notes
    -----
    See Nearest Neighbors in the sklearn online documentation for a discussion
    of the choice of ``algorithm`` and ``leaf_size``.

    This class wraps the sklearn classifier
    `sklearn.neighbors.KNeighborsClassifier`.

    https://en.wikipedia.org/wiki/K-nearest_neighbor_algorithm

    """

    def __init__(self, n_neighbors=5, radius=1.0, algorithm='auto',
                 leaf_size=30, metric=lp_distance, metric_params=None,
                 n_jobs=1):
        """Initialize the nearest neighbors searcher."""

        super().__init__(n_neighbors=n_neighbors, radius=radius,
                         algorithm=algorithm, leaf_size=leaf_size,
                         metric=metric, metric_params=metric_params,
                         n_jobs=n_jobs)

    def _init_estimator(self, sk_metric):
        """Initialize the sklearn nearest neighbors estimator.

        Args:
            sk_metric: (pyfunc or 'precomputed'): Metric compatible with
                sklearn API or matrix (n_samples, n_samples) with precomputed
                distances.

        Returns:
            Sklearn K Neighbors estimator initialized.

        """
        return _NearestNeighbors(
            n_neighbors=self.n_neighbors, radius=self.radius,
            algorithm=self.algorithm, leaf_size=self.leaf_size,
            metric=sk_metric, metric_params=self.metric_params,
            n_jobs=self.n_jobs)

    def fit(self, X):
        """Fit the model using X as training data.

        Args:
            X (:class:`FDataGrid`, array_matrix): Training data. FDataGrid
                with the training data or array matrix with shape
                [n_samples, n_samples] if metric='precomputed'.

        Note:
            This method wraps the corresponding sklearn routine in the module
            ``sklearn.neighbors``.

        """
        # If metric is precomputed no different with the Sklearn stimator
        if self.metric == 'precomputed':
            self.estimator_ = self._init_estimator(self.metric)
            self.estimator_.fit(X)
        else:
            self._sample_points = X.sample_points
            self._shape = X.data_matrix.shape[1:]

            # Constructs sklearn metric to manage vector instead of FDatagrids
            sk_metric = _to_sklearn_metric(self.metric, self._sample_points)

            self.estimator_ = self._init_estimator(sk_metric)
            self.estimator_.fit(self._transform_to_multivariate(X))

        return self


class KNeighborsClassifier(NeighborsBase, KNeighborsMixin, ClassifierMixin,
                           NeighborsClassifierMixin):
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
        the Lp distance. See the documentation of the metrics module
        for a list of available metrics.
    metric_params : dict, optional (default = None)
        Additional keyword arguments for the metric function.
    n_jobs : int or None, optional (default=None)
        The number of parallel jobs to run for neighbors search.
        ``None`` means 1 unless in a :obj:`joblib.parallel_backend` context.
        ``-1`` means using all processors.
        Doesn't affect :meth:`fit` method.
    See also
    --------
    RadiusNeighborsClassifier
    KNeighborsScalarRegressor
    RadiusNeighborsScalarRegressor
    NearestNeighbors
    NearestCentroids
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
                 leaf_size=30, metric=lp_distance, metric_params=None,
                 n_jobs=1):
        """Initialize the classifier."""

        super().__init__(n_neighbors=n_neighbors,
                         weights=weights, algorithm=algorithm,
                         leaf_size=leaf_size, metric=metric,
                         metric_params=metric_params, n_jobs=n_jobs)

    def _init_estimator(self, sk_metric):
        """Initialize the sklearn K neighbors estimator.

        Args:
            sk_metric: (pyfunc or 'precomputed'): Metric compatible with
                sklearn API or matrix (n_samples, n_samples) with precomputed
                distances.

        Returns:
            Sklearn K Neighbors estimator initialized.

        """
        return _KNeighborsClassifier(
            n_neighbors=self.n_neighbors, weights=self.weights,
            algorithm=self.algorithm, leaf_size=self.leaf_size,
            metric=sk_metric, metric_params=self.metric_params,
            n_jobs=self.n_jobs)


class RadiusNeighborsClassifier(NeighborsBase, RadiusNeighborsMixin,
                                ClassifierMixin, NeighborsClassifierMixin):
    """Classifier implementing a vote among neighbors within a given radius

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

    See also
    --------
    KNeighborsClassifier
    KNeighborsScalarRegressor
    RadiusNeighborsScalarRegressor
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
                 outlier_label=None, n_jobs=1):
        """Initialize the classifier."""

        super().__init__(radius=radius, weights=weights, algorithm=algorithm,
                         leaf_size=leaf_size, metric=metric,
                         metric_params=metric_params, n_jobs=n_jobs)

        self.outlier_label = outlier_label

    def _init_estimator(self, sk_metric):
        """Initialize the sklearn radius neighbors estimator.

        Args:
            sk_metric: (pyfunc or 'precomputed'): Metric compatible with
                sklearn API or matrix (n_samples, n_samples) with precomputed
                distances.

        Returns:
            Sklearn Radius Neighbors estimator initialized.

        """
        return _RadiusNeighborsClassifier(
            radius=self.radius, weights=self.weights,
            algorithm=self.algorithm, leaf_size=self.leaf_size,
            metric=sk_metric, metric_params=self.metric_params,
            outlier_label=self.outlier_label, n_jobs=self.n_jobs)


class NearestCentroids(BaseEstimator, ClassifierMixin):
    """Nearest centroid classifier for functional data.

    Each class is represented by its centroid, with test samples classified to
    the class with the nearest centroid.

    Parameters
    ----------
        metric : callable, (default
            :func:`lp_distance <skfda.metrics.lp_distance>`)
            The metric to use when calculating distance between test samples and
            centroids. See the documentation of the metrics module
            for a list of available metrics. Defaults used L2 distance.
        mean: callable, (default :func:`mean <skfda.exploratory.stats.mean>`)
            The centroids for the samples corresponding to each class is the
            point from which the sum of the distances (according to the metric)
            of all samples that belong to that particular class are minimized.
            By default it is used the usual mean, which minimizes the sum of L2
            distance. This parameter allows change the centroid constructor. The
            function must accept a :class:`FData` with the samples of one class
            and return a :class:`FData` object with only one sample representing
            the centroid.
    Attributes
    ----------
    centroids_ : :class:`FDataGrid`
        FDatagrid containing the centroid of each class
    See also
    --------
    KNeighborsClassifier
    RadiusNeighborsClassifier
    KNeighborsScalarRegressor
    RadiusNeighborsScalarRegressor
    NearestNeighbors

    """
    def __init__(self, metric=lp_distance, mean=mean):
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

        self._pairwise_distance = pairwise_distance(self.metric)

        check_classification_targets(y)

        le = LabelEncoder()
        y_ind = le.fit_transform(y)
        self.classes_ = classes = le.classes_
        n_classes = classes.size
        if n_classes < 2:
            raise ValueError(f'The number of classes has to be greater than'
                             f' one; got {n_classes} class')

        self.centroids_ = self.mean(X[y_ind == 0])

        # This could be changed to allow all the concatenation at the same time
        # After merge image-operations
        for cur_class in range(1, n_classes):
            center_mask = y_ind == cur_class
            centroid = self.mean(X[center_mask])
            self.centroids_ = self.centroids_.concatenate(centroid)

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

class KNeighborsScalarRegressor(NeighborsBase, KNeighborsMixin, RegressorMixin,
                                NeighborsScalarRegresorMixin):
    """Regression based on k-nearest neighbors with scalar response.

    The target is predicted by local interpolation of the targets
    associated of the nearest neighbors in the training set.

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
                 n_jobs=1):
        """Initialize the classifier."""

        super().__init__(n_neighbors=n_neighbors,
                         weights=weights, algorithm=algorithm,
                         leaf_size=leaf_size, metric=metric,
                         metric_params=metric_params, n_jobs=n_jobs)

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

class RadiusNeighborsScalarRegressor(NeighborsBase, RadiusNeighborsMixin,
                                     RegressorMixin,
                                     NeighborsScalarRegresorMixin):
    """Scalar regression based on neighbors within a fixed radius.

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
                 n_jobs=1):
        """Initialize the classifier."""

        super().__init__(radius=radius, weights=weights, algorithm=algorithm,
                         leaf_size=leaf_size, metric=metric,
                         metric_params=metric_params, n_jobs=n_jobs)


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
