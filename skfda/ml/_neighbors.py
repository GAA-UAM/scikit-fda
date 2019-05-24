"""Module with classes to neighbors classification and regression."""

from abc import ABCMeta, abstractmethod, abstractproperty

from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_is_fitted as sklearn_check_is_fitted

# Sklearn classes to be wrapped
from sklearn.neighbors import NearestNeighbors as _NearestNeighbors
from sklearn.neighbors import KNeighborsClassifier as _KNeighborsClassifier
from sklearn.neighbors import (RadiusNeighborsClassifier as
                               _RadiusNeighborsClassifier)

from .. import FDataGrid
from ..misc.metrics import lp_distance

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


def _to_sklearn_metric(metric, sample_points, check=True):
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
        >>> sklearn_lp_distance = _to_sklearn_metric(lp_distance, x)
        >>> sklearn_lp_distance(np.ones(len(x)), np.zeros(len(x))).round(2)
        1.0

    """
    # Shape -> (Nsamples = 1, domain_dims...., image_dimension (-1))
    shape = [1] + [len(axis) for axis in sample_points] + [-1]

    if check:
        def sklearn_metric(x, y, **kwargs):

            return metric(_from_multivariate(x, sample_points, shape),
                          _from_multivariate(y, sample_points, shape), **kwargs)
    else:
        def sklearn_metric(x, y, **kwargs):

            return metric(_from_multivariate(x, sample_points, shape),
                          _from_multivariate(y, sample_points, shape),
                          check=False, **kwargs)

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
            print(self.estimator_)
            self.estimator_.fit(self._transform_to_multivariate(X), y)

        return self


    def _check_is_fitted(self):
        """Check if the estimator is fitted.

        Raise:
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


class NearestNeighbors(NeighborsBase, KNeighborsMixin, RadiusNeighborsMixin):
    """


    """

    def __init__(self, n_neighbors=5, radius=1.0, weights='uniform',
                 algorithm='auto', leaf_size=30, metric=lp_distance,
                 metric_params=None, outlier_label=None, n_jobs=1):


        super().__init__(n_neighbors=n_neighbors, radius=radius,
                         weights=weights, algorithm=algorithm,
                         leaf_size=leaf_size, metric=metric,
                         metric_params=metric_params, n_jobs=n_jobs)

        self.outlier_label = outlier_label

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
            weights=self.weights, algorithm=self.algorithm,
            leaf_size=self.leaf_size, metric=sk_metric,
            metric_params=self.metric_params, n_jobs=self.n_jobs)


class KNeighborsClassifier(NeighborsBase, KNeighborsMixin, ClassifierMixin):
    r"""


    """
    def __init__(self, n_neighbors=5, weights='uniform', algorithm='auto',
                 leaf_size=30, metric=lp_distance, metric_params=None,
                 n_jobs=1):
        """

        """
        super().__init__(n_neighbors = n_neighbors,
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

    def predict(self, X):
        r"""


        """
        self._check_is_fitted()

        X = self._transform_to_multivariate(X)

        return self.estimator_.predict(X)


    def predict_proba(self, X):
        r"""


        """
        self._check_is_fitted()

        X = self._transform_to_multivariate(X)

        return self.estimator_.predict_proba(X)

class RadiusNeighborsClassifier(NeighborsBase, RadiusNeighborsMixin,
                                ClassifierMixin):
    r"""


    """
    def __init__(self, radius=1.0, weights='uniform', algorithm='auto',
                 leaf_size=30, metric=lp_distance, metric_params=None,
                 outlier_label =None, n_jobs=1):
        """


        """
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

    def predict(self, X):
        r"""


        """

        self._check_is_fitted()

        X = self._transform_to_multivariate(X)

        return self.estimator_.predict(X)


    def predict_proba(self, X):
        r"""


        """
        self._check_is_fitted()

        X = self._transform_to_multivariate(X)

        return self.estimator_.predict_proba(X)
