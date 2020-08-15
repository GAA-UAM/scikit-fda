import abc
import copy
import numbers
import random

import scipy.stats
import sklearn.base
import sklearn.utils

import dcor
import numpy as np
import numpy.linalg as linalg
import numpy.ma as ma

from ....representation import FDataGrid
from .maxima_hunting import _compute_dependence


def _transform_to_2d(t):
    t = np.asarray(t)

    dim = len(t.shape)
    assert dim <= 2

    if dim < 2:
        t = np.atleast_2d(t).T

    return t


def _execute_kernel(kernel, t_0, t_1):
    t_0 = _transform_to_2d(t_0)
    t_1 = _transform_to_2d(t_1)

    if isinstance(kernel, numbers.Number):
        return kernel
    else:
        if callable(kernel):
            result = kernel(t_0, t_1)
        else:
            # GPy kernel
            result = kernel.K(t_0, t_1)

        assert result.shape[0] == len(t_0)
        assert result.shape[1] == len(t_1)
        return result


def _absolute_argmax(function, *, mask):
    '''
    Computes the absolute maximum of a discretized function.

    Some values of the function may be masked in order not to consider them
    as maximum.

    Parameters:
        function (numpy array): Discretized function.
        mask (numpy boolean array): Masked values.

    Returns:
        int: Index of the absolute maximum.

    '''
    masked_function = ma.array(function, mask=mask)

    t_max = ma.argmax(masked_function)

    t_max = np.unravel_index(t_max, function.shape)

    return t_max


class Correction(abc.ABC):
    '''
    Base class for applying a correction after a point is taken, eliminating
    its influence over the rest
    '''

    def begin(self, X: FDataGrid, Y):
        '''
        Initialization
        '''
        pass

    def conditioned(self, **kwargs):
        '''
        Returns a correction object that is conditioned to the value of a point
        '''
        return self

    @abc.abstractmethod
    def correct(self, X, selected_index):
        '''
        Correct the trajectories.

        Arguments:
            X: Matrix with one trajectory per row
            T: Times of each measure
            selected_index: index of the selected value
        '''
        pass

    def __call__(self, *args, **kwargs):
        self.correct(*args, **kwargs)


class ConditionalExpectationCorrection(Correction):

    @abc.abstractmethod
    def conditional_expectation(self, T, t_0, x_0, selected_index):
        pass

    def correct(self, X, selected_index):
        T = X.sample_points[0]

        t_0 = T[selected_index]

        x_index = (slice(None),) + tuple(selected_index) + (np.newaxis,)
        x_0 = X.data_matrix[x_index]

        T = _transform_to_2d(T)

        X.data_matrix[...] -= self.conditional_expectation(T, t_0, x_0,
                                                           selected_index).T

        X.data_matrix[:, selected_index] = 0


class SampleGPCorrection(ConditionalExpectationCorrection):
    '''
    Correction assuming that the process is Gaussian and using as the kernel
    the sample covariance.
    '''

    def __init__(self, markov=False):
        self.gaussian_correction = None
        self.covariance_matrix = None
        self.T = None
        self.time = 0
        self.markov = markov
        self.cond_points = []

    def begin(self, X: FDataGrid, Y):
        T = X.sample_points

        X_copy = np.copy(X.data_matrix[..., 0])

        Y = np.ravel(Y)
        for class_label in np.unique(Y):
            trajectories = X_copy[Y == class_label, :]

            mean = np.mean(trajectories, axis=0)
            X_copy[Y == class_label, :] -= mean

        self.covariance_matrix = np.cov(X_copy, rowvar=False)
        self.T = np.ravel(T)
        self.gaussian_correction = GaussianCorrection(kernel=self.__kernel)

    def __kernel(self, t_0, t_1):
        i = np.searchsorted(self.T, t_0)
        j = np.searchsorted(self.T, t_1)

        i = np.ravel(i)
        j = np.ravel(j)

        return self.covariance_matrix[np.ix_(i, j)]

    def conditioned(self, t_0, **kwargs):
        self.cond_points.append(t_0)
        self.cond_points.sort()
        self.gaussian_correction = self.gaussian_correction.conditioned(
            t_0=t_0, **kwargs)
        return self

    def conditional_expectation(self, T, t_0, x_0, selected_index):

        gp_condexp = self.gaussian_correction.conditional_expectation(
            T, t_0, x_0, selected_index)

        if self.markov:
            left_index = np.searchsorted(self.cond_points, t_0)

            left_value = (self.cond_points[left_index - 1]
                          if left_index != 0 else None)
            right_value = (self.cond_points[left_index]
                           if left_index != len(self.cond_points) else None)

            if left_value is not None:
                gp_condexp[:, T.ravel() < left_value, :] = 0

            if right_value is not None:
                gp_condexp[:, T.ravel() > right_value, :] = 0

        return gp_condexp


class PicklableKernel():

    def __init__(self, kernel):
        super().__setattr__('_PicklableKernel__kernel', kernel)

    def __getattr__(self, name):
        if name != '__deepcopy__':
            return getattr(self.__kernel, name)

    def __setattr__(self, name, value):
        setattr(self.__kernel, name, value)

    def __getstate__(self):
        return {'class': self.__kernel.__class__,
                'input_dim': self.__kernel.input_dim,
                'values': self.__kernel.param_array}

    def __setstate__(self, state):
        super().__setattr__('_PicklableKernel__kernel', state['class'](
            input_dim=state['input_dim']))
        self.__kernel.param_array[...] = state['values']

    def __call__(self, *args, **kwargs):
        return self.__kernel.K(*args, **kwargs)


def make_kernel(k):
    try:
        import GPy
    except ImportError:
        return k

    if isinstance(k, GPy.kern.Kern):
        return PicklableKernel(k)
    else:
        return k


class UniformCorrection(Correction):
    '''
    Correction assuming that the underlying process is an Ornstein-Uhlenbeck
    process with infinite lengthscale.
    '''

    def __init__(self):
        pass

    def conditioned(self, X, t_0, **kwargs):
        from ....misc.covariances import Brownian

        return GaussianCorrection(kernel=Brownian(origin=t_0))

    def correct(self, X, selected_index):
        x_index = (slice(None),) + tuple(selected_index) + (np.newaxis,)

        # Have to copy it because otherwise is a view and shouldn't be
        # subtracted from the original matrix
        x_0 = np.copy(X.data_matrix[x_index])

        X.data_matrix[...] -= x_0


class GaussianCorrection(ConditionalExpectationCorrection):
    '''
    Correction assuming that the underlying process is Gaussian.
    '''

    def __init__(self, expectation=0, kernel=1, optimize_kernel=False):
        super(GaussianCorrection, self).__init__()

        self.__expectation = expectation
        self.__kernel = make_kernel(kernel)
        self.optimize_kernel = optimize_kernel
        self.kernel_params_optimized_names = None
        self.kernel_params_optimized_values = None

    def begin(self, X, Y):
        if self.optimize_kernel:
            import GPy

            T = X.sample_points[0]
            X_copy = np.copy(X.data_matrix[..., 0])

            Y = np.ravel(Y)
            for class_label in np.unique(Y):
                trajectories = X_copy[Y == class_label, :]

                mean = np.mean(trajectories, axis=0)
                X_copy[Y == class_label, :] -= mean

            m = GPy.models.GPRegression(
                T[:, None], X_copy.T,
                kernel=self.__kernel._PicklableKernel__kernel)
            m.constrain_positive('')
            m.optimize()

            self.kernel_params_optimized_names = m.parameter_names(),
            self.kernel_params_optimized_values = m.param_array

            self.__kernel = copy.deepcopy(make_kernel(m.kern))

    def conditioned(self, X, t_0, **kwargs):
        # If the point makes the matrix singular, don't change the correction
        try:
            return GaussianConditionedCorrection(
                expectation=self.expectation,
                kernel=self.kernel,
                point_list=t_0)
        except linalg.LinAlgError:
            return self

    def expectation(self, t):

        if isinstance(self.__expectation, numbers.Number):
            expectation = np.ones_like(t, dtype=float) * self.__expectation
        else:
            expectation = self.__expectation(t)

        return expectation

    def kernel(self, t_0, t_1):
        return _execute_kernel(self.__kernel, t_0, t_1)

    covariance = kernel

    def variance(self, t):
        return self.covariance(t, t)

    def conditional_expectation(self, T, t_0, x_0, selected_index):

        var = self.variance(t_0)

        expectation = self.expectation(T)
        assert expectation.shape == T.shape

        t_0_expectation = expectation[selected_index]

        b_T = self.covariance(T, t_0)
        assert b_T.shape == T.shape

        cond_expectation = (expectation +
                            b_T / var *
                            (x_0.T - t_0_expectation)
                            ) if var else expectation + np.zeros_like(x_0.T)

        return cond_expectation


class GaussianConditionedCorrection(GaussianCorrection):
    '''
    Correction assuming that the underlying process is a Gaussian conditioned
    to several points with value 0.
    '''

    def __init__(self, point_list, expectation=0,
                 kernel=1, **kwargs):
        super(GaussianConditionedCorrection, self).__init__(
            expectation=self.__expectation,
            kernel=self.__kernel,
            **kwargs)

        self.point_list = _transform_to_2d(point_list)
        self.__gaussian_expectation = expectation
        self.__gaussian_kernel = make_kernel(kernel)
        self.__covariance_matrix = self.gaussian_kernel(
            self.point_list, self.point_list
        )
        self.__covariance_matrix_inv = np.linalg.inv(self.__covariance_matrix)

    def conditioned(self, X, t_0, **kwargs):

        # If the point makes the matrix singular, don't change the correction
        try:
            return GaussianConditionedCorrection(
                expectation=self.__gaussian_expectation,
                kernel=self.__gaussian_kernel,
                point_list=np.concatenate((self.point_list, [[t_0]]))
            )
        except linalg.LinAlgError:
            return self

    def gaussian_expectation(self, t):
        if isinstance(self.__gaussian_expectation, numbers.Number):
            expectation = (np.ones_like(t, dtype=float) *
                           self.__gaussian_expectation)
        else:
            expectation = self.__gaussian_expectation(t)

        return expectation

    def gaussian_kernel(self, t_0, t_1):
        return _execute_kernel(self.__gaussian_kernel, t_0, t_1)

    def __expectation(self, t):

        A_inv = self.__covariance_matrix_inv

        b_T = self.gaussian_kernel(t, self.point_list)
        # assert b_T.shape[0] == np.shape(t)[-1]
        # assert b_T.shape[1] == np.shape(point_list)[-1]

        c = -self.gaussian_expectation(self.point_list)
        assert c.shape == np.shape(self.point_list)

        original_expect = self.gaussian_expectation(t)
        assert original_expect.shape == t.shape

        modified_expect = b_T.dot(A_inv).dot(c)
        assert modified_expect.shape == t.shape

        expectation = original_expect + modified_expect
        assert expectation.shape == t.shape

        return expectation

    def __kernel(self, t_0, t_1):

        A_inv = self.__covariance_matrix_inv

        b_t_0_T = self.gaussian_kernel(t_0, self.point_list)
        # assert b_t_0_T.shape[0] == np.shape(np.atleast_2d(t_0))[0]
        # assert b_t_0_T.shape[1] == np.shape(point_list)[-1]

        b_t_1 = self.gaussian_kernel(self.point_list, t_1)
        # assert b_t_1.shape[0] == np.shape(point_list)[-1]
        # assert b_t_1.shape[1] == np.shape(np.atleast_2d(t_1))[0]

        return (self.gaussian_kernel(t_0, t_1) -
                b_t_0_T @ A_inv @ b_t_1)


class RMHResult(object):

    def __init__(self, index, score):
        self.index = index
        self.score = score
        self.matrix_after_correction = None
        self.original_dependence = None
        self.influence_mask = None
        self.current_mask = None

    def __repr__(self):
        return (self.__class__.__name__ +
                "(index={index}, score={score})"
                .format(index=self.index, score=self.score))


def get_influence_mask(X, t_max_index, min_redundancy, dependence_measure,
                       old_mask):
    '''
    Get the mask of the points that have much dependence with the
    selected point.
    '''

    sl = slice(None)

    def get_index(index):
        return (sl,) + tuple(index) + (np.newaxis,)

    def is_redundant(index):

        max_point = np.squeeze(X[get_index(t_max_index)], axis=1)
        test_point = np.squeeze(X[get_index(index)], axis=1)

        return (dependence_measure(max_point, test_point) >
                min_redundancy)

    def adjacent_indexes(index):
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

    def update_mask(new_mask, index):
        indexes = [index]

        while indexes:
            index = indexes.pop()
            # Check if it wasn't masked before
            if (
                not old_mask[index] and not new_mask[index] and
                is_redundant(index)
            ):
                new_mask[index] = True
                for i in adjacent_indexes(index):
                    indexes.append(i)

    new_mask = np.zeros_like(old_mask)

    update_mask(new_mask, t_max_index)

    # The selected point is masked even if min_redundancy is high
    new_mask[t_max_index] = True

    return new_mask


class StoppingCondition(abc.ABC):
    '''Stopping condition for RMH.'''

    def begin(self, **kwargs):
        pass

    @abc.abstractmethod
    def __call__(self, **kwargs):
        pass


class ScoreThresholdStop(StoppingCondition):
    '''Stop when the score is under a threshold.'''

    def __init__(self, threshold=None):
        super().__init__()
        self.threshold = threshold
        self.threshold_specified = threshold is not None

    def begin(self, min_relevance, **kwargs):
        if not self.threshold_specified:
            self.threshold = min_relevance

    def __call__(self, *, score, **kwargs):
        return score < self.threshold


def chi_bound(x, y, significance):

    x_dist = dcor.distances.pairwise_distances(x)
    y_dist = dcor.distances.pairwise_distances(y)

    t2 = np.mean(x_dist) * np.mean(y_dist)

    chi_quant = scipy.stats.chi2.ppf(1 - significance, df=1)

    return chi_quant * t2 / x_dist.shape[0]


def normal_bound(x, y, significance):

    x_dist = dcor.distances.pairwise_distances(x)
    y_dist = dcor.distances.pairwise_distances(y)

    t2 = np.mean(x_dist) * np.mean(y_dist)

    norm_quant = scipy.stats.norm.ppf(1 - significance / 2, df=1)

    return norm_quant ** 2 * t2 / x_dist.shape[0]


class Chi2BoundStop(StoppingCondition):
    '''Stop when the score is under a threshold.'''

    def __init__(self, significance=0.01):
        super().__init__()
        self.significance = significance

    def __call__(self, *, selected_variable, X, Y,
                 **kwargs):
        bound = chi_bound(selected_variable, Y, self.significance)
        # print(f'bound = {bound}')
        return dcor.u_distance_covariance_sqr(selected_variable, Y) < bound


class NormalBoundStop(StoppingCondition):
    '''Stop when the score is under a threshold.'''

    def __init__(self, significance=0.01):
        super().__init__()
        self.significance = significance

    def __call__(self, *, selected_variable, X, Y,
                 **kwargs):
        bound = normal_bound(selected_variable, Y, self.significance)
        # print(f'bound = {bound}')
        return dcor.u_distance_covariance_sqr(selected_variable, Y) < bound


class DcovTestStop(StoppingCondition):
    '''Stop when the score is under a threshold.'''

    def __init__(self, significance=0.01, num_resamples=200,
                 random_state=None):
        super().__init__()

        if random_state == -1:
            random_state = None

        self.significance = significance
        self.num_resamples = num_resamples
        self.random_state = random_state

    def __call__(self, *, selected_variable, X, Y,
                 **kwargs):
        return dcor.independence.distance_covariance_test(
            selected_variable, Y,
            num_resamples=self.num_resamples,
            random_state=self.random_state).p_value >= self.significance


class NComponentsStop(StoppingCondition):
    '''Stop when the first n components are selected.'''

    def __init__(self, n_components=1):
        super().__init__()
        self.n_components = n_components

    def begin(self, min_relevance, **kwargs):
        self.selected_components = 0

    def __call__(self, *, score, **kwargs):
        stop = self.selected_components >= self.n_components
        self.selected_components += 1
        return stop


def redundancy_distance_covariance(x, y):
    dcov = dcor.u_distance_covariance_sqr(x, y)
    dvar = dcor.u_distance_covariance_sqr(x, x)

    return dcov / dvar


def rec_maxima_hunting_gen_no_copy(
        X: FDataGrid, Y, min_redundancy=0.9, min_relevance=0.2,
        dependence_measure=dcor.u_distance_correlation_sqr,
        redundancy_dependence_measure=None,
        correction=None,
        mask=None,
        get_intermediate_results=False,
        stopping_condition=None):
    '''
    Find the most relevant features of a function using recursive maxima
    hunting. It changes the original matrix.

    Arguments:
        X: Matrix with one trajectory per row
        Y: Vector for the response variable
        min_redundancy: Minimum dependence between two features to be
        considered redundant.
        min_relevance: Minimum score to consider a point relevant
        dependence_measure: Measure of the dependence between variables
        correction: Class that defines the correction to apply to eliminate the
        influence of the selected feature.
    '''

    # X = np.asfarray(X)
    Y = np.asfarray(Y)

    if correction is None:
        correction = UniformCorrection()

    if redundancy_dependence_measure is None:
        redundancy_dependence_measure = dependence_measure

    if mask is None:
        mask = np.zeros([len(t) for t in X.sample_points], dtype=bool)

    if stopping_condition is None:
        stopping_condition = Chi2BoundStop()

    first_pass = True

    correction.begin(X, Y)

    try:
        stopping_condition.begin(X=X.data_matrix, Y=Y, T=X.sample_points[0],
                                 min_relevance=min_relevance,
                                 dependence_measure=dependence_measure)
    except AttributeError:
        pass

    while True:
        dependencies = _compute_dependence(
            X=X.data_matrix, Y=Y,
            dependence_measure=dependence_measure)

        t_max_index = _absolute_argmax(dependencies,
                                       mask=mask)
        score = dependencies[t_max_index]

        repeated_point = mask[t_max_index]

        stopping_condition_reached = stopping_condition(
            selected_index=t_max_index,
            selected_variable=X.data_matrix[(slice(None),) +
                                            tuple(t_max_index)],
            score=score,
            X=X.data_matrix, Y=Y)

        if ((repeated_point or stopping_condition_reached) and
                not first_pass):
            return

        influence_mask = get_influence_mask(
            X=X.data_matrix, t_max_index=t_max_index,
            min_redundancy=min_redundancy,
            dependence_measure=redundancy_dependence_measure,
            old_mask=mask)

        mask |= influence_mask

        # Correct the influence of t_max
        correction(X=X,
                   selected_index=t_max_index)
        result = RMHResult(index=t_max_index, score=score)

        # Additional info, useful for debugging
        if get_intermediate_results:
            result.matrix_after_correction = np.copy(X.data_matrix)
            result.original_dependence = dependencies
            result.influence_mask = influence_mask
            result.current_mask = mask

        new_X = yield result  # Accept modifications to the matrix
        if new_X is not None:
            X.data_matrix = new_X

        correction = correction.conditioned(
            X=X.data_matrix,
            T=X.sample_points[0],
            t_0=X.sample_points[0][t_max_index])

        first_pass = False


def rec_maxima_hunting_gen(X, *args, **kwargs):
    yield from rec_maxima_hunting_gen_no_copy(copy.copy(X),
                                              *args, **kwargs)


def rec_maxima_hunting(*args, **kwargs):
    return list(rec_maxima_hunting_gen(*args, **kwargs))


class RecursiveMaximaHunting(
        sklearn.base.BaseEstimator, sklearn.base.TransformerMixin):
    r'''
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

    This method was originally described in a special case in article [1]_.

    Parameters:

        dependence_measure (callable): Dependence measure to use. By default,
            it uses the bias corrected squared distance correlation.
        local_maxima_selector (callable): Function to detect local maxima. The
            default is :func:`select_local_maxima` with ``order`` parameter
            equal to one. The original article used a similar function testing
            different values of ``order``.

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

        >>> rmh = variable_selection.RecursiveMaximaHunting()
        >>> _ = rmh.fit(X, y)
        >>> point_mask = rmh.get_support()
        >>> points = X.sample_points[0][point_mask]
        >>> np.allclose(points, [0.25, 0.5, 0.75], rtol=1e-1)
        True

        Apply the learned dimensionality reduction

        >>> X_dimred = rmh.transform(X)
        >>> len(X.sample_points[0])
        100
        >>> X_dimred.shape
        (10000, 3)

    References:

        .. [1] J. L. Torrecilla and A. Suárez, “Feature selection in
               functional data classification with recursive maxima hunting,”
               in Advances in Neural Information Processing Systems 29,
               Curran Associates, Inc., 2016, pp. 4835–4843.

    '''

    def __init__(self,
                 min_redundancy=0.9,
                 min_relevance=0.2,
                 dependence_measure=dcor.u_distance_correlation_sqr,
                 redundancy_dependence_measure=None,
                 n_components=None,
                 correction=None,
                 stopping_condition=None,
                 num_extra_features=0):
        self.min_redundancy = min_redundancy
        self.min_relevance = min_relevance
        self.dependence_measure = dependence_measure
        self.redundancy_dependence_measure = redundancy_dependence_measure
        self.n_components = n_components
        self.correction = correction
        self.stopping_condition = stopping_condition
        self.num_extra_features = num_extra_features

    def fit(self, X, y):

        self.features_shape_ = X.data_matrix.shape[1:]

        red_dep_measure = self.redundancy_dependence_measure

        indexes = []
        for i, result in enumerate(
            rec_maxima_hunting_gen(
                X=X.copy(),
                Y=y,
                min_redundancy=self.min_redundancy,
                min_relevance=self.min_relevance,
                dependence_measure=self.dependence_measure,
                redundancy_dependence_measure=red_dep_measure,
                correction=self.correction,
                stopping_condition=self.stopping_condition,
                get_intermediate_results=(self.num_extra_features != 0))):

            if self.n_components is None or i < self.n_components:
                indexes.append(result.index)

                if self.num_extra_features:
                    mask = result.influence_mask
                    new_indexes = [a[0] for a in np.ndenumerate(mask) if a[1]]
                    new_indexes.remove(result.index)
                    new_indexes = random.sample(new_indexes, min(
                        len(new_indexes), self.num_extra_features))

                    indexes = indexes + new_indexes

            else:
                break

        self.indexes_ = tuple(np.transpose(indexes).tolist())

        return self

    def transform(self, X):

        X_matrix = X.data_matrix

        sklearn.utils.validation.check_is_fitted(self)

        if X_matrix.shape[1:] != self.features_shape_:
            raise ValueError("The trajectories have a different number of "
                             "points than the ones fitted")

        output = X_matrix[(slice(None),) + self.indexes_]

        return output.reshape(X.n_samples, -1)

    def get_support(self, indices: bool=False):
        indexes_unraveled = self.indexes_
        if indices:
            return indexes_unraveled
        else:
            mask = np.zeros(self.features_shape_[0], dtype=bool)
            mask[self.indexes_] = True
            return mask

    def fit_all(self, param_grid, X_train, y_train):

        # We can fit at the same time all n_components, but nothing else

        if len(param_grid) == 0:
            return NotImplemented

        n_components_max = 1

        for param in param_grid:
            if len(param) != 1:
                return NotImplemented

            n_components = param.get("n_components", None)

            if n_components is None:
                return NotImplemented

            n_components_max = max(n_components_max, n_components)

        print(f'Fitting RMH with n_components={n_components_max}')

        cloned = sklearn.base.clone(self)
        cloned.set_params(n_components=n_components_max)
        cloned.fit(X_train, y_train)

        fitted_estimators = [None] * len(param_grid)

        for i, param in enumerate(param_grid):
            n_components = param["n_components"]
            fitted_estimators[i] = copy.copy(cloned)
            fitted_estimators[i].set_params(n_components=n_components)
            fitted_estimators[i].indexes_ = cloned.indexes_[:n_components]

        return fitted_estimators
