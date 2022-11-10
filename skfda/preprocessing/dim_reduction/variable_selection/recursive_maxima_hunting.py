"""Recursive Maxima Hunting implementation."""
from __future__ import annotations

import abc
import copy
import math
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Iterable,
    List,
    Mapping,
    Optional,
    Sequence,
    Tuple,
    Union,
    overload,
)

import numpy as np
import numpy.linalg as linalg
import numpy.ma as ma
import scipy.stats
import sklearn.utils
from typing_extensions import Literal

import dcor

from ...._utils._sklearn_adapter import (
    BaseEstimator,
    InductiveTransformerMixin,
)
from ....representation import FDataGrid, concatenate
from ....typing._numpy import ArrayLike, NDArrayBool, NDArrayFloat, NDArrayInt
from ._base import _compute_dependence, _DependenceMeasure as _DepMeasure

if TYPE_CHECKING:
    from ....misc.covariances import CovarianceLike
    import GPy


def _transform_to_2d(t: ArrayLike) -> NDArrayFloat:
    t = np.asfarray(t)

    dim = len(t.shape)
    assert dim <= 2

    if dim < 2:
        t = np.atleast_2d(t).T

    return t


class _PicklableKernel():
    """Class used to pickle GPy kernels."""

    def __init__(self, kernel: GPy.kern.Kern) -> None:
        super().__setattr__('_PicklableKernel__kernel', kernel)

    def __getattr__(self, name: str) -> Any:
        if name != '__deepcopy__':
            return getattr(self.__kernel, name)

    def __setattr__(self, name: str, value: Any) -> None:
        setattr(self.__kernel, name, value)

    def __getstate__(self) -> Mapping[str, Any]:
        return {
            'class': self.__kernel.__class__,
            'input_dim': self.__kernel.input_dim,
            'values': self.__kernel.param_array,
        }

    def __setstate__(self, state: Mapping[str, Any]) -> None:
        super().__setattr__('_PicklableKernel__kernel', state['class'](
            input_dim=state['input_dim']),
        )
        self.__kernel.param_array[...] = state['values']

    def __call__(self, *args: Any, **kwargs: Any) -> NDArrayFloat:
        return self.__kernel.K(*args, **kwargs)  # type: ignore[no-any-return]


def make_kernel(k: CovarianceLike) -> CovarianceLike:
    try:
        import GPy
    except ImportError:
        return k

    if isinstance(k, GPy.kern.Kern):
        return _PicklableKernel(k)

    return k


def _absolute_argmax(
    function: FDataGrid,
    *,
    mask: NDArrayBool,
) -> Tuple[int, ...]:
    """
    Compute the absolute maximum of a discretized function.

    Some values of the function may be masked in order not to consider them
    as maximum.

    Parameters:
        function: Discretized function.
        mask: Masked values.

    Returns:
        Index of the absolute maximum.

    """
    masked_function = ma.array(  # type: ignore[no-untyped-call]
        function.data_matrix,
        mask=mask,
    )

    t_max = ma.argmax(masked_function)

    return np.unravel_index(t_max, function.data_matrix.shape[1:-1])


class Correction(BaseEstimator):
    """
    Base class for corrections.

    A correction applies a modification after a point is taken, eliminating
    its influence over the rest.

    """

    def begin(self, X: FDataGrid, y: NDArrayFloat) -> None:
        """
        Initialize the correction for a run.

        The initial parameters of Recursive Maxima Hunting can be used there.

        """
        pass

    def conditioned(
        self,
        *,
        X: NDArrayFloat,
        T: NDArrayFloat,
        t_0: float,
    ) -> Correction:
        """
        Return a correction object conditioned to the value of a point.

        This method is necessary because after the RMH correction step, the
        functions follow a different model.

        """
        return self

    @abc.abstractmethod
    def correct(
        self,
        X: FDataGrid,
        selected_index: Tuple[int, ...],
    ) -> FDataGrid:
        """
        Correct the trajectories.

        This method subtracts the influence of the selected point from the
        other points in the function.

        Parameters:
            X: Functions in the current iteration of the algorithm.
            selected_index: Index of the selected point
                in the ``data_matrix``.

        """
        pass

    def __call__(self, *args: Any, **kwargs: Any) -> FDataGrid:
        return self.correct(*args, **kwargs)


class ConditionalMeanCorrection(Correction):
    """
    Base class for applying a correction based on the conditional expectation.

    The functions are assumed to be realizations of a particular stochastic
    process. The information subtracted in each iteration would be the
    mean of the process conditioned to the value observed at the
    selected point.

    """

    @abc.abstractmethod
    def conditional_mean(
        self,
        X: FDataGrid,
        selected_index: Tuple[int, ...],
    ) -> FDataGrid:
        """
        Mean of the process conditioned to the value observed.

        Parameters:
            X: Functions in the current iteration of the algorithm.
            selected_index (int or tuple of int): Index of the selected point
                in the ``data_matrix``.

        """
        pass

    def correct(
        self,
        X: FDataGrid,
        selected_index: Tuple[int, ...],
    ) -> FDataGrid:

        return X - self.conditional_mean(
            X,
            selected_index,
        )


class GaussianCorrection(ConditionalMeanCorrection):
    r"""
    Correction assuming that the underlying process is Gaussian.

    The conditional mean of a Gaussian process :math:`X(t)` is

    .. math::

        \mathbb{E}[X(t) \mid X(t_0) = x_0] = \mathbb{E}[X(t)]
        + \frac{\mathrm{Cov}[X(t), X(t_0)]}{\mathrm{Cov}[X(t_0), X(t_0)]}
        (X(t_0) - \mathbb{E}[X(t_0)])

    The corrections after this is applied are of type
    :class:`GaussianConditionedCorrection`.

    Parameters:
        mean: Mean function of the Gaussian process.
        cov: Covariance function of the Gaussian process.
        fit_hyperparameters: If ``True`` the hyperparameters of the
            covariance function are optimized for the data.

    """

    def __init__(
        self,
        *,
        mean: Union[float, Callable[[NDArrayFloat], NDArrayFloat]] = 0,
        cov: Union[float, CovarianceLike] = 1,
        fit_hyperparameters: bool = False,
    ) -> None:
        super().__init__()

        self.mean = mean
        self.cov = make_kernel(cov)
        self.fit_hyperparameters = fit_hyperparameters

    def begin(self, X: FDataGrid, y: NDArrayFloat) -> None:
        if self.fit_hyperparameters:
            # TODO: Migrate this to scikit-learn
            import GPy

            T = X.grid_points[0]
            X_copy = np.copy(X.data_matrix[..., 0])

            y = np.ravel(y)
            for class_label in np.unique(y):
                trajectories = X_copy[y == class_label, :]

                mean = np.mean(trajectories, axis=0)
                X_copy[y == class_label, :] -= mean

            gpy_kernel = getattr(self.cov, "_PicklableKernel__kernel")

            m = GPy.models.GPRegression(
                T[:, None],
                X_copy.T,
                kernel=gpy_kernel,
            )
            m.constrain_positive('')
            m.optimize()

            self.cov_ = copy.deepcopy(make_kernel(m.kern))

    def _evaluate_mean(self, t: NDArrayFloat) -> NDArrayFloat:

        mean = self.mean

        if isinstance(mean, (int, float)):
            expectation = np.ones_like(t, dtype=float) * mean
        else:
            expectation = mean(t)

        return expectation

    def _evaluate_cov(
        self,
        t_0: NDArrayFloat,
        t_1: NDArrayFloat,
    ) -> NDArrayFloat:
        from ....misc.covariances import _execute_covariance

        cov = getattr(self, "cov_", self.cov)

        return _execute_covariance(cov, t_0, t_1)

    def conditioned(
        self,
        *,
        X: NDArrayFloat,
        T: NDArrayFloat,
        t_0: float,
    ) -> Correction:
        # If the point makes the matrix singular, don't change the correction

        cov = getattr(self, "cov_", self.cov)

        try:

            correction = GaussianConditionedCorrection(
                mean=self.mean,
                cov=cov,
                conditioning_points=np.asarray(t_0),
            )

            correction._covariance_matrix_inv()

            return correction

        except linalg.LinAlgError:

            return self

    def conditional_mean(
        self,
        X: FDataGrid,
        selected_index: Tuple[int, ...],
    ) -> FDataGrid:

        T = X.grid_points[0]

        t_0 = T[selected_index]

        x_index = (slice(None),) + tuple(selected_index) + (np.newaxis,)
        x_0 = X.data_matrix[x_index]

        T = _transform_to_2d(T)

        var = self._evaluate_cov(t_0, t_0)

        expectation = self._evaluate_mean(T)
        assert expectation.shape == T.shape

        t_0_expectation = expectation[selected_index]

        b_T = self._evaluate_cov(T, t_0)
        assert b_T.shape == T.shape

        cond_expectation = (
            expectation
            + b_T / var
            * (x_0.T - t_0_expectation)
        ) if var else expectation + np.zeros_like(x_0.T)

        return X.copy(
            data_matrix=cond_expectation.T,
            sample_names=None,
        )


class GaussianConditionedCorrection(GaussianCorrection):
    """
    Correction for a conditioned Gaussian process.

    Correction assuming that the underlying process is Gaussian, with several
    values conditioned to 0.

    The conditional mean is inherited from :class:`GaussianCorrection`, with
    the conditioned mean and covariance.

    The corrections after this is applied are of type
    :class:`GaussianConditionedCorrection`, adding additional points.

    Parameters:
        conditioning_points: Points where the process is conditioned to
            have the value 0.
        mean: Mean function of the (unconditioned) Gaussian process.
        cov: Covariance function of the (unconditioned) Gaussian process.

    """

    def __init__(
        self,
        conditioning_points: NDArrayFloat,
        *,
        mean: Union[float, Callable[[NDArrayFloat], NDArrayFloat]] = 0,
        cov: CovarianceLike = 1,
    ) -> None:

        super().__init__(
            mean=mean,
            cov=cov,
        )

        self.conditioning_points = conditioning_points

    def _covariance_matrix_inv(self) -> NDArrayFloat:

        cond_points = self._conditioning_points()

        cov_matrix_inv = getattr(self, "_cov_matrix_inv", None)
        if cov_matrix_inv is None:

            cov_matrix = super()._evaluate_cov(
                cond_points, cond_points,
            )

            self._cov_matrix_inv = np.linalg.inv(cov_matrix)
            cov_matrix_inv = self._cov_matrix_inv

        return cov_matrix_inv

    def _conditioning_points(self) -> NDArrayFloat:
        return _transform_to_2d(self.conditioning_points)

    def conditioned(
        self,
        *,
        X: NDArrayFloat,
        T: NDArrayFloat,
        t_0: float,
    ) -> Correction:

        # If the point makes the matrix singular, don't change the correction
        try:

            correction = GaussianConditionedCorrection(
                mean=self.mean,
                cov=self.cov,
                conditioning_points=np.concatenate(
                    (self._conditioning_points(), np.array([[t_0]])),
                ),
            )

            correction._covariance_matrix_inv()

            return correction

        except linalg.LinAlgError:

            return self

    def _evaluate_mean(self, t: NDArrayFloat) -> NDArrayFloat:

        cond_points = self._conditioning_points()

        A_inv = self._covariance_matrix_inv()

        b_T = super()._evaluate_cov(t, cond_points)

        c = -super()._evaluate_mean(cond_points)
        assert c.shape == np.shape(cond_points)

        original_expect = super()._evaluate_mean(t)
        assert original_expect.shape == t.shape

        modified_expect = b_T.dot(A_inv).dot(c)
        assert modified_expect.shape == t.shape

        expectation = original_expect + modified_expect
        assert expectation.shape == t.shape

        return expectation  # type: ignore[no-any-return]

    def _evaluate_cov(
        self,
        t_0: NDArrayFloat,
        t_1: NDArrayFloat,
    ) -> NDArrayFloat:

        cond_points = self._conditioning_points()

        A_inv = self._covariance_matrix_inv()

        b_t_0_T = super()._evaluate_cov(t_0, cond_points)

        b_t_1 = super()._evaluate_cov(cond_points, t_1)

        return (
            super()._evaluate_cov(t_0, t_1)
            - b_t_0_T @ A_inv @ b_t_1
        )


class GaussianSampleCorrection(ConditionalMeanCorrection):
    """
    Gaussian correction with sample covariance.

    Correction assuming that the process is Gaussian and using as the kernel
    the sample covariance.

    """

    def begin(self, X: FDataGrid, y: NDArrayFloat) -> None:

        X_copy = np.copy(X.data_matrix[..., 0])

        y = np.ravel(y)
        for class_label in np.unique(y):
            trajectories = X_copy[y == class_label, :]

            mean = np.mean(trajectories, axis=0)
            X_copy[y == class_label, :] -= mean

        self.cov_matrix_ = np.cov(X_copy, rowvar=False)
        self.t_ = np.ravel(X.grid_points)
        self.gaussian_correction_ = GaussianCorrection(
            cov=self.cov_fun,
        )

    def cov_fun(
        self,
        t_0: ArrayLike,
        t_1: ArrayLike,
    ) -> NDArrayFloat:
        i = np.searchsorted(self.t_, t_0)
        j = np.searchsorted(self.t_, t_1)

        i_r = np.ravel(i)
        j_r = np.ravel(j)

        return self.cov_matrix_[  # type: ignore[no-any-return]
            np.ix_(i_r, j_r)
        ]

    def conditioned(
        self,
        *,
        X: NDArrayFloat,
        T: NDArrayFloat,
        t_0: float,
    ) -> Correction:
        self.gaussian_correction_ = self.gaussian_correction_.conditioned(
            X=X,
            T=T,
            t_0=t_0,
        )
        return self

    def conditional_mean(
        self,
        X: FDataGrid,
        selected_index: Tuple[int, ...],
    ) -> FDataGrid:

        return self.gaussian_correction_.conditional_mean(
            X,
            selected_index,
        )


class UniformCorrection(Correction):
    """
    Correction for uniform process.

    Correction assuming that the underlying process is an Ornstein-Uhlenbeck
    process with infinite lengthscale.

    The initial conditional mean subtracts the observed value from every
    point, and the following correction is a :class:`GaussianCorrection`
    with a :class:`~skfda.misc.covariances.Brownian` covariance function with
    the selected point as its origin.

    """

    def conditioned(
        self,
        X: NDArrayFloat,
        T: NDArrayFloat,
        t_0: float,
    ) -> Correction:
        from ....misc.covariances import Brownian

        return GaussianCorrection(cov=Brownian(origin=t_0))

    def correct(
        self,
        X: FDataGrid,
        selected_index: Tuple[int, ...],
    ) -> FDataGrid:
        x_index = (slice(None),) + selected_index
        x_0 = X.data_matrix[x_index]

        return X - x_0


class StoppingCondition(BaseEstimator):
    """
    Stopping condition for RMH.

    This is a callable that should return ``True`` if the algorithm must stop
    and the current point should not be selected.

    """

    @abc.abstractmethod
    def __call__(
        self,
        *,
        selected_index: Tuple[int, ...],
        dependences: FDataGrid,
        selected_variable: NDArrayFloat,
        X: FDataGrid,
        y: NDArrayFloat,
    ) -> bool:
        """Whether the algorithm should stop."""


class ScoreThresholdStop(StoppingCondition):
    """
    Stop when the score is under a threshold.

    This stopping condition requires that the score has a known bound, for
    example that it takes values in the interval :math:`[0, 1]`.

    This is one of the simplest stopping criterions, but it requires that
    the user chose a threshold parameter, which controls the number of
    points chosen and can vary per problem.

    Parameters:
        threshold: Value compared with the score. If the score
            of the selected point is not higher than that,
            the point will not be selected (unless it is
            the first iteration) and RMH will end.

    """

    def __init__(self, threshold: float = 0.2) -> None:

        super().__init__()
        self.threshold = threshold

    def __call__(
        self,
        *,
        selected_index: Tuple[int, ...],
        dependences: FDataGrid,
        **kwargs: Any,
    ) -> bool:

        score = float(dependences.data_matrix[0, selected_index, 0])

        return score < self.threshold


class AsymptoticIndependenceTestStop(StoppingCondition):
    r"""
    Stop when the selected point is independent from the target.

    It uses an asymptotic test based on the chi-squared distribution described
    in :footcite:`szekely+rizzo_2010_brownian`. The test rejects independence
    if

    .. math::

        \frac{n \mathcal{V}_n}{T_2} \geq \mathcal{X}_{1-\alpha}^2,

    where :math:`n` is the number of samples, :math:`\mathcal{V}_n` is the
    sample distance correlation between the selected point and the target,
    :math:`\mathcal{X}_{1-\alpha}^2` is the :math:`1-\alpha` quantile of a
    chi-squared variable with 1 degree of freedom. :math:`T_2` is the product
    of the means of the distance matrices of the selected point and the
    target, a term which is involved in the standard computation of the sample
    distance covariance.

    Parameters:
        significance: Significance used in the independence test. By
            default is 0.01 (1%).

    References:
        .. footbibliography::

    """

    def __init__(self, significance: float = 0.01) -> None:
        super().__init__()
        self.significance = significance

    def chi_bound(
        self,
        x: NDArrayFloat,
        y: NDArrayFloat,
        significance: float,
    ) -> float:

        x_dist = dcor.distances.pairwise_distances(x)
        y_dist = dcor.distances.pairwise_distances(y)

        t2 = np.mean(x_dist) * np.mean(y_dist)

        chi_quant = scipy.stats.chi2.ppf(1 - significance, df=1)

        return float(chi_quant * t2 / x_dist.shape[0])

    def __call__(
        self,
        *,
        selected_variable: NDArrayFloat,
        y: NDArrayFloat,
        **kwargs: Any,
    ) -> bool:

        bound = self.chi_bound(selected_variable, y, self.significance)

        return bool(
            dcor.u_distance_covariance_sqr(selected_variable, y) < bound,
        )


class RedundancyCondition(BaseEstimator):
    """
    Redundancy condition for RMH.

    This is a callable that should return ``True`` if the two points are
    redundant and false otherwise.

    """

    @abc.abstractmethod
    def __call__(
        self,
        *,
        max_point: NDArrayFloat,
        test_point: NDArrayFloat,
        **kwargs: Any,
    ) -> bool:
        pass


class DependenceThresholdRedundancy(RedundancyCondition):
    """
    The points are redundant if their dependency is above a given threshold.

    This stopping condition requires that the dependency has a known bound, for
    example that it takes values in the interval :math:`[0, 1]`.

    Parameters:
        threshold: Value compared with the score. If the score
            of the selected point is not higher than that,
            the point will not be selected (unless it is
            the first iteration) and RMH will end.
        dependence_measure: Dependence measure to use. By default,
            it uses the bias corrected squared distance correlation.

    """

    def __init__(
        self,
        threshold: float = 0.9,
        *,
        dependence_measure: _DepMeasure[
            NDArrayFloat,
            NDArrayFloat,
        ] = dcor.u_distance_correlation_sqr,
    ) -> None:
        super().__init__()
        self.threshold = threshold
        self.dependence_measure = dependence_measure

    def __call__(
        self,
        *,
        max_point: NDArrayFloat,
        test_point: NDArrayFloat,
        **kwargs: Any,
    ) -> bool:
        return bool(
            self.dependence_measure(max_point, test_point) > self.threshold,
        )


def _get_influence_mask(
    X: NDArrayFloat,
    t_max_index: Tuple[int, ...],
    redundancy_condition: RedundancyCondition,
    old_mask: NDArrayBool,
) -> NDArrayBool:
    """Get the mask of points that have a large dependence with another."""
    sl = slice(None)

    def get_index(
        index: Tuple[int, ...],
    ) -> Tuple[Union[slice, int, None], ...]:
        return (sl,) + tuple(index) + (np.newaxis,)

    def is_redundant(index: Tuple[int, ...]) -> bool:

        max_point = np.squeeze(X[get_index(t_max_index)], axis=1)
        test_point = np.squeeze(X[get_index(index)], axis=1)

        return redundancy_condition(
            max_point=max_point,
            test_point=test_point,
        )

    def adjacent_indexes(index: Tuple[int, ...]) -> Iterable[Tuple[int, ...]]:
        for i, coord in enumerate(index):
            # Out of bounds right check
            if coord < (X.shape[i + 1] - 1):
                new_index = list(index)
                new_index[i] += 1
                yield tuple(new_index)
            # Out of bounds left check
            if coord > 0:
                new_index = list(index)
                new_index[i] -= 1
                yield tuple(new_index)

    def update_mask(
        new_mask: NDArrayBool,
        index: Tuple[int, ...],
    ) -> None:
        indexes = [index]

        while indexes:
            index = indexes.pop()
            # Check if it wasn't masked before
            if (
                not old_mask[index] and not new_mask[index]
                and is_redundant(index)
            ):
                new_mask[index] = True
                for i in adjacent_indexes(index):
                    indexes.append(i)

    new_mask = np.zeros_like(old_mask)

    update_mask(new_mask, t_max_index)

    # The selected point is masked even if min_redundancy is high
    new_mask[t_max_index] = True

    return new_mask


class RecursiveMaximaHunting(
    BaseEstimator,
    InductiveTransformerMixin[
        FDataGrid,
        NDArrayFloat,
        Union[NDArrayInt, NDArrayFloat],
    ],
):
    """
    Recursive Maxima Hunting variable selection.

    This is a filter variable selection method for problems with a target
    variable. It evaluates a dependence measure between each point of the
    function and the target variable, selects the point that maximizes this
    dependence, subtracts the information of the selected point from
    the original functions and repeat the process.

    This method is inspired by :class:`MaximaHunting`, and shares
    similarities with it. However, as the information of the selected point
    is subtracted from each function in each step of the algorithm, this
    algorithm can uncover points that are not relevant by themselves but are
    relevant once other points are selected. Those points would not be
    selected by :class:`MaximaHunting` alone.

    This method was originally described in a special case in article
    :footcite:`torrecilla+suarez_2016_hunting`.
    Additional information about the usage of this method can be found in
    :doc:`/modules/preprocessing/dim_reduction/recursive_maxima_hunting`.

    Parameters:
        dependence_measure: Dependence measure to use. By default,
            it uses the bias corrected squared distance correlation.
        max_features: Maximum number of features to select. By default
            there is no limit.
        correction: Correction used to subtract the information
            of each selected point in each iteration. By default it is
            a :class:`.UniformCorrection` object.
        redundancy_condition: Condition to consider a point
            redundant with the selected maxima and discard it from future
            consideration as a maximum. By default it is a
            :class:`DependenceThresholdRedundancy` object.
        stopping_condition: Condition to stop the algorithm. By
            default it is a :class:`.AsymptoticIndependenceTestStop`
            object.

    Examples:
        >>> from skfda.preprocessing.dim_reduction import variable_selection
        >>> from skfda.datasets import make_gaussian_process
        >>> import skfda
        >>> import numpy as np

        We create trajectories from two classes, one with zero mean and the
        other with a peak-like mean. Both have Brownian covariance.

        >>> n_samples = 1000
        >>> n_features = 100
        >>>
        >>> def mean_1(t):
        ...     return (
        ...         np.abs(t - 0.25)
        ...         - 2 * np.abs(t - 0.5)
        ...         + np.abs(t - 0.75)
        ...     )
        >>>
        >>> X_0 = make_gaussian_process(
        ...     n_samples=n_samples // 2,
        ...     n_features=n_features,
        ...     random_state=0,
        ... )
        >>> X_1 = make_gaussian_process(
        ...     n_samples=n_samples // 2,
        ...     n_features=n_features,
        ...     mean=mean_1,
        ...     random_state=1,
        ... )
        >>> X = skfda.concatenate((X_0, X_1))
        >>>
        >>> y = np.zeros(n_samples)
        >>> y [n_samples // 2:] = 1

        Select the relevant points to distinguish the two classes

        >>> rmh = variable_selection.RecursiveMaximaHunting()
        >>> _ = rmh.fit(X, y)
        >>> point_mask = rmh.get_support()
        >>> points = X.grid_points[0][point_mask]
        >>> np.allclose(points, [0.25, 0.5, 0.75], rtol=1e-1)
        True

        Apply the learned dimensionality reduction

        >>> X_dimred = rmh.transform(X)
        >>> len(X.grid_points[0])
        100
        >>> X_dimred.shape
        (1000, 3)

    References:
        .. footbibliography::

    """

    def __init__(
        self,
        *,
        dependence_measure: _DepMeasure[
            NDArrayFloat,
            NDArrayFloat,
        ] = dcor.u_distance_correlation_sqr,
        max_features: Optional[int] = None,
        correction: Optional[Correction] = None,
        redundancy_condition: Optional[RedundancyCondition] = None,
        stopping_condition: Optional[StoppingCondition] = None,
        _get_intermediate_results: bool = False,
    ) -> None:
        self.dependence_measure = dependence_measure
        self.max_features = max_features
        self.correction = correction
        self.redundancy_condition = redundancy_condition
        self.stopping_condition = stopping_condition
        self._get_intermediate_results = _get_intermediate_results

    def fit(  # type: ignore[override] # noqa: D102
        self,
        X: FDataGrid,
        y: Union[NDArrayInt, NDArrayFloat],
    ) -> RecursiveMaximaHunting:
        """Recursive maxima hunting algorithm."""
        self.features_shape_ = X.data_matrix.shape[1:]

        y = np.asfarray(y)

        correction = (
            self.correction
            if self.correction
            else UniformCorrection()
        )

        redundancy_condition = (
            self.redundancy_condition
            if self.redundancy_condition
            else DependenceThresholdRedundancy()
        )

        stopping_condition = (
            self.stopping_condition
            if self.stopping_condition
            else AsymptoticIndependenceTestStop()
        )

        max_features = (
            self.max_features
            if self.max_features
            else math.inf
        )

        mask = np.zeros([len(t) for t in X.grid_points], dtype=bool)
        indexes: List[Tuple[int, ...]] = []
        corrected_functions = []
        relevances = []
        first_pass = True

        correction.begin(X, y)

        while True:
            dependences = _compute_dependence(
                X,
                y,
                dependence_measure=self.dependence_measure,
            )
            corrected_functions.append(X)
            relevances.append(dependences)

            t_max_index = _absolute_argmax(
                dependences,
                mask=mask,
            )

            repeated_point = mask[t_max_index]

            stopping_condition_reached = stopping_condition(
                selected_index=t_max_index,
                dependences=dependences,
                selected_variable=X.data_matrix[
                    (slice(None),) + tuple(t_max_index)
                ],
                X=X,
                y=y,
            )

            if (
                (
                    len(indexes) >= max_features
                    or repeated_point
                    or stopping_condition_reached
                )
                and not first_pass
            ):
                self.indexes_ = tuple(np.transpose(indexes).tolist())
                self._relevances = concatenate(relevances)
                self._corrected_functions = corrected_functions
                return self

            indexes.append(t_max_index)

            influence_mask = _get_influence_mask(
                X=X.data_matrix,
                t_max_index=t_max_index,
                redundancy_condition=redundancy_condition,
                old_mask=mask,
            )

            mask |= influence_mask

            # Correct the influence of t_max
            X = correction(
                X=X,
                selected_index=t_max_index,
            )

            correction = correction.conditioned(
                X=X.data_matrix,
                T=X.grid_points[0],
                t_0=X.grid_points[0][t_max_index],
            )

            first_pass = False

    def transform(self, X: FDataGrid) -> NDArrayFloat:

        X_matrix = X.data_matrix

        sklearn.utils.validation.check_is_fitted(self)

        if X_matrix.shape[1:] != self.features_shape_:
            raise ValueError(
                "The trajectories have a different number of "
                "points than the ones fitted",
            )

        output = X_matrix[(slice(None),) + self.indexes_]

        return output.reshape(  # type: ignore[no-any-return]
            X.n_samples,
            -1,
        )

    @overload
    def get_support(
        self,
        indices: Literal[True],
    ) -> Sequence[Tuple[int, ...]]:
        pass

    @overload
    def get_support(
        self,
        indices: Literal[False] = False,
    ) -> NDArrayBool:
        pass

    def get_support(
        self,
        indices: bool = False,
    ) -> Union[Sequence[Tuple[int, ...]], NDArrayBool]:

        if indices:
            return self.indexes_

        mask = np.zeros(self.features_shape_[0], dtype=bool)
        mask[self.indexes_] = True
        return mask
