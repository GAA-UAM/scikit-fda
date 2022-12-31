"""Neighbors outlier detection methods."""
from __future__ import annotations

from typing import Any, TypeVar, Union, overload

from sklearn.base import OutlierMixin
from sklearn.neighbors import LocalOutlierFactor as _LocalOutlierFactor
from typing_extensions import Literal

from ...misc.metrics import PairwiseMetric, l2_distance
from ...ml._neighbors_base import AlgorithmType, KNeighborsMixin
from ...representation import FData
from ...typing._metric import Metric
from ...typing._numpy import NDArrayFloat, NDArrayInt

SelfType = TypeVar("SelfType", bound="LocalOutlierFactor[Any]")
InputBound = Union[NDArrayFloat, FData]
Input = TypeVar("Input", contravariant=True, bound=InputBound)


class LocalOutlierFactor(
    KNeighborsMixin[Input, None],
    OutlierMixin,  # type: ignore[misc]
):
    r"""
    Unsupervised Outlier Detection.

    Unsupervised Outlier Detection using Local Outlier Factor (LOF).

    The anomaly score of each sample is called Local Outlier Factor.
    It measures the local deviation of density of a given sample with
    respect to its neighbors.

    It is local in that the anomaly score depends on how isolated the object
    is with respect to the surrounding neighborhood.

    More precisely, locality is given by k-nearest neighbors, whose distance
    is used to estimate the local density.

    By comparing the local density of a sample to the local densities of
    its neighbors, one can identify samples that have a substantially lower
    density than their neighbors. These are considered outliers.

    Parameters:
        n_neighbors: Number of neighbors to use by default for
            :meth:`kneighbors` queries.
            If n_neighbors is larger than the number of samples provided,
            all samples will be used.
        algorithm: Algorithm used to compute the nearest neighbors:

            - 'ball_tree' will use :class:`BallTree`
            - 'kd_tree' will use :class:`KDTree`
            - 'brute' will use a brute-force search.
            - 'auto' will attempt to decide the most appropriate algorithm
              based on the values passed to :meth:`fit` method.

        leaf_size: Leaf size passed to :class:`BallTree` or :class:`KDTree`.
            This can affect the speed of the construction and query, as well as
            the memory required to store the tree. The optimal value depends on
            the nature of the problem.
        metric: The distance metric to use for the tree.  The default metric is
            the L2 distance. See the documentation of the metrics module
            for a list of available metrics.
        contamination: The amount of contamination of the data set, i.e. the
            proportion of outliers in the data set. When fitting this is used
            to define the threshold on the decision function. If "auto", the
            decision function threshold is determined as in the original paper
            [BKNS2000]_.
        novelty: By default, LocalOutlierFactor is only meant to be used for
            outlier detection (novelty=False). Set novelty to True if you want
            to use LocalOutlierFactor for novelty detection. In this case be
            aware that that you should only use predict, decision_function and
            score_samples on new unseen data and not on the training set.
        n_jobs: The number of parallel jobs to run for neighbors search.
            ``None`` means 1 unless in a :obj:`joblib.parallel_backend`
            context.
            ``-1`` means using all processors. See :term:`Glossary <n_jobs>`
            for more details.
            Affects only :meth:`kneighbors` and :meth:`kneighbors_graph`
            methods.

    Attributes:
        negative_outlier_factor\_ : The opposite LOF of the training samples.
            The higher, the more normal. Inliers tend to have a LOF score close
            to 1 (``negative_outlier_factor_`` close to -1), while outliers
            tend to have a larger LOF score.
            The local outlier factor (LOF) of a sample captures its
            supposed 'degree of abnormality'.
            It is the average of the ratio of the local reachability density of
            a sample and those of its k-nearest neighbors.
        n_neighbors\_ : The actual number of neighbors used for
            :meth:`kneighbors` queries.
        offset\_ : Offset used to obtain binary labels from the raw scores.
            Observations having a negative_outlier_factor smaller than
            `offset_` are detected as abnormal.
            The offset is set to -1.5 (inliers score around -1), except when a
            contamination parameter different than "auto" is provided. In that
            case, the offset is defined in such a way we obtain the expected
            number of outliers in training.

    Examples:
        **Local Outlier Factor (LOF) for outlier detection**.

        >>> from skfda.exploratory.outliers import LocalOutlierFactor

        Creation of simulated dataset with 2 outliers to be used with LOF.

        >>> from skfda.datasets import make_sinusoidal_process
        >>> fd_clean = make_sinusoidal_process(n_samples=25, error_std=0,
        ...                                    phase_std=0.1, random_state=0)
        >>> fd_outliers = make_sinusoidal_process(
        ...     n_samples=2, error_std=0, phase_mean=0.5, random_state=5)
        >>> fd = fd_outliers.concatenate(fd_clean) # Dataset with 2 outliers

        Detection of outliers with LOF.

        >>> lof = LocalOutlierFactor()
        >>> is_outlier = lof.fit_predict(fd)
        >>> is_outlier # -1 for anomalies/outliers and +1 for inliers
        array([-1, -1,  1,  1,  1,  1,  1,  1, ...,  1,  1,  1,  1])

        The negative outlier factor stored.

        >>> lof.negative_outlier_factor_.round(2)
        array([-7.11, -1.54, -1.  , -0.99, ..., -0.97,  -1. ,  -0.99])

        **Novelty detection with LOF**.

        Creation of a dataset without outliers.

        >>> fd_train = make_sinusoidal_process(
        ...     n_samples=25,
        ...     error_std=0,
        ...     phase_std=0.1,
        ...     random_state=9,
        ... )

        Fit of LOF using the dataset without outliers.

        >>> lof = LocalOutlierFactor(novelty=True)
        >>> lof.fit(fd_train)
        LocalOutlierFactor(...novelty=True)

        Detection of annomalies for new samples.

        >>> lof.predict(fd) # Predict with samples not used in fit
        array([-1, -1,  1,  1,  1,  1,  1,  1, ...,  1,  1,  1,  1])


    References:
        .. [BKNS2000] Breunig, M. M., Kriegel, H. P., Ng, R. T., & Sander,
           J. (2000, May). LOF: identifying density-based local outliers.
           In ACM sigmod record.

    Notes:
        This estimator wraps the scikit-learn class
        :class:`~sklearn.neighbors.LocalOutlierFactor` employing functional
        metrics and data instead of the multivariate ones.

    See also:
        :class:`~skfda.ml.classification.KNeighborsClassifier`
        :class:`~skfda.ml.classification.RadiusNeighborsClassifier`
        :class:`~skfda.ml.classification.NearestCentroids`
        :class:`~skfda.ml.regression.KNeighborsRegressor`
        :class:`~skfda.ml.regression.RadiusNeighborsRegressor`
        :class:`~skfda.ml.clustering.NearestNeighbors`
    """

    @overload
    def __init__(
        self: LocalOutlierFactor[NDArrayFloat],
        *,
        n_neighbors: int = 20,
        algorithm: AlgorithmType = 'auto',
        leaf_size: int = 30,
        metric: Literal["precomputed"],
        contamination: float | Literal["auto"] = "auto",
        novelty: bool = False,
        n_jobs: int | None = None,
    ) -> None:
        pass

    @overload
    def __init__(
        self: LocalOutlierFactor[InputBound],
        *,
        n_neighbors: int = 20,
        algorithm: AlgorithmType = 'auto',
        leaf_size: int = 30,
        contamination: float | Literal["auto"] = "auto",
        novelty: bool = False,
        n_jobs: int | None = None,
    ) -> None:
        pass

    @overload
    def __init__(
        self,
        *,
        n_neighbors: int = 20,
        algorithm: AlgorithmType = 'auto',
        leaf_size: int = 30,
        metric: Metric[Input] = l2_distance,
        contamination: float | Literal["auto"] = "auto",
        novelty: bool = False,
        n_jobs: int | None = None,
    ) -> None:
        pass

    def __init__(
        self,
        *,
        n_neighbors: int = 20,
        algorithm: AlgorithmType = 'auto',
        leaf_size: int = 30,
        metric: Literal["precomputed"] | Metric[Input] = l2_distance,
        contamination: float | Literal["auto"] = "auto",
        novelty: bool = False,
        n_jobs: int | None = None,
    ) -> None:
        super().__init__(
            n_neighbors=n_neighbors,
            algorithm=algorithm,
            leaf_size=leaf_size,
            metric=metric,
            n_jobs=n_jobs,
        )
        self.contamination = contamination
        self.novelty = novelty

    def _init_estimator(self) -> _LocalOutlierFactor:
        return _LocalOutlierFactor(
            n_neighbors=self.n_neighbors,
            algorithm=self.algorithm,
            leaf_size=self.leaf_size,
            metric="precomputed",
            contamination=self.contamination,
            novelty=self.novelty,
            n_jobs=self.n_jobs,
        )

    def _store_fit_data(self) -> None:
        """Store the parameters created during the fit."""
        self.negative_outlier_factor_ = (
            self._estimator.negative_outlier_factor_
        )
        self.n_neighbors_ = self._estimator.n_neighbors_
        self.offset_ = self._estimator.offset_

    def fit(  # noqa: D102
        self: SelfType,
        X: Input,
        y: None = None,
    ) -> SelfType:
        self._fit(X, y, fit_with_zeros=False)
        self._store_fit_data()

        return self

    def predict(
        self,
        X: Input,
    ) -> NDArrayInt:
        """
        Predict the labels (1 inlier, -1 outlier) of X according to LOF.

        This method allows to generalize prediction to *new observations* (not
        in the training set). Only available for novelty detection (when
        novelty is set to True).

        If X is None, returns the same as fit_predict(X_train).

        Parameters:
            X: FDataGrid containing the query sample or samples to compute the
                Local Outlier Factor w.r.t. to the training samples or array
                with the distances to the training samples if
                metric='precomputed'.

        Returns:
            Returns -1 for anomalies/outliers and +1 for inliers.

        """
        self._check_is_fitted()
        X_dist = self._X_to_distances(X)

        return self._estimator.predict(X_dist)  # type: ignore[no-any-return]

    def fit_predict(
        self,
        X: Input,
        y: None = None,
    ) -> NDArrayInt:
        """
        Fits the model to the training set X and returns the labels.

        Label is 1 for an inlier and -1 for an outlier according to the LOF
        score and the contamination parameter.

        Parameters:
            X: Training data. Samples, or array with shape
                [n_samples, n_samples] if metric='precomputed'.
            y : Ignored. Not used, present for API consistency by convention.

        Returns:
            Returns -1 for anomalies/outliers and 1 for inliers.

        """
        # In this estimator fit_predict cannot be wrapped as fit().predict()

        self._estimator = self._init_estimator()
        metric = self.metric

        if metric == 'precomputed':
            res = self._estimator.fit_predict(X, y)
        else:
            X_dist = PairwiseMetric(metric)(X)
            res = self._estimator.fit_predict(X_dist, y)

        self._store_fit_data()
        self._fitted_with_distances = True

        return res  # type: ignore[no-any-return]

    def decision_function(
        self,
        X: Input,
    ) -> NDArrayFloat:
        """
        Shifted opposite of the Local Outlier Factor of X.

        Bigger is better, i.e. large values correspond to inliers.
        The shift offset allows a zero threshold for being an outlier.
        Only available for novelty detection (when novelty is set to True).
        The argument X is supposed to contain *new data*: if X contains a
        point from training, it considers the later in its own neighborhood.
        Also, the samples in X are not considered in the neighborhood of any
        point.

        Parameters:
            X: Query sample or samples to compute the
                Local Outlier Factor w.r.t. to the training samples.

        Returns:
            The shifted opposite of the Local Outlier Factor of each input
            samples. The lower, the more abnormal. Negative scores represent
            outliers, positive scores represent inliers.

        """
        self._check_is_fitted()
        X_dist = self._X_to_distances(X)

        return (  # type: ignore[no-any-return]
            self._estimator.decision_function(X_dist)
        )

    def score_samples(
        self,
        X: Input,
    ) -> NDArrayFloat:
        """
        Opposite of the Local Outlier Factor of X.

        It is the opposite as bigger is better, i.e. large values correspond
        to inliers.

        Only available for novelty detection (when novelty is set to True).
        The argument X is supposed to contain *new data*: if X contains a
        point from training, it considers the later in its own neighborhood.
        Also, the samples in X are not considered in the neighborhood of any
        point.

        The score_samples on training data is available by considering the
        the ``negative_outlier_factor_`` attribute.

        Parameters:
            X: Query sample or samples to compute the
                Local Outlier Factor w.r.t. to the training samples.

        Returns:
            The opposite of the Local Outlier Factor of each input samples.
            The lower, the more abnormal.

        """
        self._check_is_fitted()
        X_dist = self._X_to_distances(X)

        return (  # type: ignore[no-any-return]
            self._estimator.score_samples(X_dist)
        )
