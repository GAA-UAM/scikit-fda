from __future__ import annotations

from typing import Tuple, Union, overload

import numpy as np
from sklearn.utils import check_random_state
from typing_extensions import Literal

from ... import concatenate
from ..._utils import RandomStateLike
from ...datasets import make_gaussian_process
from ...misc.metrics import lp_distance
from ...representation import FData, FDataGrid
from ...representation._typing import ArrayLike


def v_sample_stat(fd: FData, weights: ArrayLike, p: int = 2) -> float:
    r"""
    Compute sample statistic.

    Calculates a statistic that measures the variability between groups of
    samples in a :class:`skfda.representation.FData` object.

    The statistic defined as below is calculated between all the samples in a
    :class:`skfda.representation.FData` object with a given set of
    weights.

    Let :math:`\{f_i\}_{i=1}^k` be a set of samples in a FData object.
    Let :math:`\{w_j\}_{j=1}^k` be a set of weights, where :math:`w_i` is
    related to the sample :math:`f_i` for :math:`i=1,\dots,k`.
    The statistic is defined as:

    .. math::
        V_n = \sum_{i<j}^kw_i\|f_i-f_j\|^2

    This statistic is defined in Cuevas :footcite:`cuevas++_2004_anova`.

    Args:
         fd: Object containing all the samples for which we want
            to calculate the statistic.
         weights: Weights related to each sample. Each
            weight is expected to appear in the same position as its
            corresponding sample in the FData object.
         p: p of the lp norm. Must be greater or equal
            than 1. If p='inf' or p=np.inf it is used the L infinity metric.
            Defaults to 2.

    Returns:
        The value of the statistic.

    Raises:
        ValueError

    Examples:
        >>> from skfda.inference.anova import v_sample_stat
        >>> from skfda.representation.grid import FDataGrid
        >>> import numpy as np

        We create different trajectories to be applied in the statistic and a
        set of weights.

        >>> t = np.linspace(0, 1, 50)
        >>> x1 = t * (1 - t) ** 5
        >>> x2 = t ** 2 * (1 - t) ** 4
        >>> x3 = t ** 3 * (1 - t) ** 3
        >>> fd = FDataGrid([x1, x2, x3], grid_points=t)
        >>> weights = [10, 20, 30]

        Finally the value of the statistic is calculated:

        >>> v_sample_stat(fd, weights)
        0.01649448843348894

    References:
        .. footbibliography::

    """
    weights = np.asarray(weights)
    if not isinstance(fd, FData):
        raise ValueError("Argument type must inherit FData.")
    if len(weights) != fd.n_samples:
        raise ValueError("Number of weights must match number of samples.")

    t_ind = np.tril_indices(fd.n_samples, -1)
    coef = weights[t_ind[1]]
    return float(np.sum(
        coef * lp_distance(
            fd[t_ind[0]],
            fd[t_ind[1]],
            p=p,
        ) ** p,
    ))


def v_asymptotic_stat(fd: FData, weights: ArrayLike, p: int = 2) -> float:
    r"""
    Compute asymptitic statistic.

    Calculates a statistic that measures the variability between groups of
    samples in a :class:`skfda.representation.FData` object.

    The statistic defined as below is calculated between all the samples in a
    :class:`skfda.representation.FData` object with a given set of
    weights.

    Let :math:`\{f_i\}_{i=1}^k` be a set of samples in a FData object.
    Let :math:`\{w_j\}_{j=1}^k` be a set of weights, where :math:`w_i` is
    related to the sample :math:`f_i` for :math:`i=1,\dots,k`.
    The statistic is defined as:

    .. math::
        \sum_{i<j}^k\|f_i-f_j\sqrt{\cfrac{w_i}{w_j}}\|^2

    This statistic is defined in Cuevas :footcite:`cuevas++_2004_anova`.

    Args:
         fd: Object containing all the samples for which we want
            to calculate the statistic.
         weights: Weights related to each sample. Each
            weight is expected to appear in the same position as its
            corresponding sample in the FData object.
         p: p of the lp norm. Must be greater or equal
            than 1. If p='inf' or p=np.inf it is used the L infinity metric.
            Defaults to 2.

    Returns:
        The value of the statistic.

    Raises:
        ValueError

    Examples:
        >>> from skfda.inference.anova import v_asymptotic_stat
        >>> from skfda.representation.grid import FDataGrid
        >>> import numpy as np

        We create different trajectories to be applied in the statistic and a
        set of weights.

        >>> t = np.linspace(0, 1, 50)
        >>> x1 = t * (1 - t) ** 5
        >>> x2 = t ** 2 * (1 - t) ** 4
        >>> x3 = t ** 3 * (1 - t) ** 3
        >>> fd = FDataGrid([x1, x2, x3], grid_points=t)
        >>> weights = [10, 20, 30]

        Finally the value of the statistic is calculated:

        >>> v_asymptotic_stat(fd, weights)
        0.0018159320335885969

    References:
        .. footbibliography::

    """
    weights = np.asarray(weights)
    if not isinstance(fd, FData):
        raise ValueError("Argument type must inherit FData.")
    if len(weights) != fd.n_samples:
        raise ValueError("Number of weights must match number of samples.")
    if np.count_nonzero(weights) != len(weights):
        raise ValueError("All weights must be non-zero.")

    t_ind = np.tril_indices(fd.n_samples, -1)
    coef = np.sqrt(weights[t_ind[1]] / weights[t_ind[0]])
    left_fd = fd[t_ind[1]]
    right_fd = fd[t_ind[0]] * coef
    return float(np.sum(lp_distance(left_fd, right_fd, p=p) ** p))


def _anova_bootstrap(
    fd_grouped: Tuple[FData, ...],
    n_reps: int,
    random_state: RandomStateLike = None,
    p: int = 2,
    equal_var: bool = True,
) -> np.ndarray:

    n_groups = len(fd_grouped)
    if n_groups < 2:
        raise ValueError("At least two groups must be passed in fd_grouped.")

    for fd in fd_grouped[1:]:
        if not np.array_equal(fd.domain_range, fd_grouped[0].domain_range):
            raise ValueError(
                "Domain range must match for every FData in fd_grouped.",
            )

    start, stop = fd_grouped[0].domain_range[0]

    # List with sizes of each group
    sizes = [fd.n_samples for fd in fd_grouped]

    # Instance a random state object in case random_state is an int
    random_state = check_random_state(random_state)

    if equal_var:
        k_est = concatenate(fd_grouped).cov().data_matrix[0, ..., 0]
        k_est = [k_est] * len(fd_grouped)
    else:
        # Estimating covariances for each group
        k_est = [fd.cov().data_matrix[0, ..., 0] for fd in fd_grouped]

    # Number of sample points for gaussian processes have to match
    # the features of the covariances.
    n_features = k_est[0].shape[0]

    # Simulating n_reps observations for each of the n_groups gaussian
    # processes
    sim = [
        make_gaussian_process(
            n_reps,
            n_features=n_features,
            start=start,
            stop=stop,
            cov=k_est[i],
            random_state=random_state,
        )
        for i in range(n_groups)
    ]

    v_samples = np.empty(n_reps)
    for i in range(n_reps):
        fd = FDataGrid([s.data_matrix[i, ..., 0] for s in sim])
        v_samples[i] = v_asymptotic_stat(fd, sizes, p=p)
    return v_samples


@overload
def oneway_anova(
    *args: FData,
    n_reps: int = 2000,
    return_dist: Literal[False] = False,
    random_state: RandomStateLike = None,
    p: int = 2,
    equal_var: bool = True,
) -> Tuple[float, float]:
    pass


@overload
def oneway_anova(
    *args: FData,
    n_reps: int = 2000,
    return_dist: Literal[True],
    random_state: RandomStateLike = None,
    p: int = 2,
    equal_var: bool = True,
) -> Tuple[float, float, np.ndarray]:
    pass


def oneway_anova(
    *args: FData,
    n_reps: int = 2000,
    return_dist: bool = False,
    random_state: RandomStateLike = None,
    p: int = 2,
    equal_var: bool = True,
) -> Union[Tuple[float, float], Tuple[float, float, np.ndarray]]:
    r"""
    Perform one-way functional ANOVA.

    This function implements an asymptotic method to test the following
    null hypothesis:

    Let :math:`\{X_i\}_{i=1}^k` be a set of :math:`k` independent samples
    each one with :math:`n_i` trajectories, and let :math:`E(X_i) = m_i(
    t)`. The null hypothesis is defined as:

    .. math::
        H_0: m_1(t) = \dots = m_k(t)

    This function calculates the value of the statistic
    :func:`~skfda.inference.anova.v_sample_stat` :math:`V_n` with the means
    of the given samples. Under the null hypothesis this statistic is
    asymptotically equivalent to
    :func:`~skfda.inference.anova.v_asymptotic_stat`, where each sample
    is replaced by a gaussian process, with mean zero and the same
    covariance function as the original.

    The simulation of the distribution of the asymptotic statistic :math:`V` is
    implemented using a bootstrap procedure. One observation of the
    :math:`k` different gaussian processes defined above is simulated,
    and the value of :func:`~skfda.inference.anova.v_asymptotic_stat` is
    calculated. This procedure is repeated `n_reps` times, creating a
    sampling distribution of the statistic.

    This procedure is from Cuevas :footcite:`cuevas++_2004_anova`.

    Args:
        args: The sample measurements for each each group.
        n_reps: Number of simulations for the bootstrap
            procedure. Defaults to 2000 (This value may change in future
            versions).
        return_dist: Flag to indicate if the function should
            return a numpy.array with the sampling distribution simulated.
        random_state: Random state.
        p: p of the lp norm. Must be greater or equal
            than 1. If p='inf' or p=np.inf it is used the L infinity metric.
            Defaults to 2.
        equal_var: If True (default), perform a One-way
            ANOVA assuming the same covariance operator for all the groups,
            else considers an independent covariance operator for each group.

    Returns:
        Tuple containing the value of the sample statistic, p-value (and
        sampling distribution of the simulated asymptotic statistic if
        `return_dist` is `True`).

    Examples:
        >>> from skfda.inference.anova import oneway_anova
        >>> from skfda.datasets import fetch_gait
        >>> from numpy.random import RandomState
        >>> from numpy import printoptions

        >>> fd = fetch_gait()["data"].coordinates[1]
        >>> fd1, fd2, fd3 = fd[:13], fd[13:26], fd[26:]
        >>> oneway_anova(fd1, fd2, fd3, random_state=RandomState(42))
        (179.52499999999998, 0.5945)
        >>> _, _, dist = oneway_anova(fd1, fd2, fd3, n_reps=3,
        ...     random_state=RandomState(42),
        ...     return_dist=True)
        >>> with printoptions(precision=4):
        ...     print(dist)
        [ 184.0698 212.7395  195.3663]

    References:
        .. footbibliography::

    """
    if len(args) < 2:
        raise ValueError("At least two groups must be passed as parameter.")
    if not all(isinstance(fd, FData) for fd in args):
        raise ValueError("Argument type must inherit FData.")
    if n_reps < 1:
        raise ValueError("Number of simulations must be positive.")

    fd_groups = args
    if not all(isinstance(fd, type(fd_groups[0])) for fd in fd_groups[1:]):
        raise TypeError('Found mixed FData types in arguments.')

    for fd in fd_groups[1:]:
        if not np.array_equal(fd.domain_range, fd_groups[0].domain_range):
            raise ValueError("Domain range must match for every FData passed.")

    if isinstance(fd_groups[0], FDataGrid):
        # Creating list with all the sample points
        list_sample = [fd.grid_points[0].tolist() for fd in fd_groups]
        # Checking that the all the entries in the list are the same
        if list_sample.count(list_sample[0]) != len(list_sample):
            raise ValueError(
                "All FDataGrid passed must have the same grid points.",
            )
    else:  # If type is FDataBasis, check same basis
        list_basis = [fd.basis for fd in fd_groups]
        if list_basis.count(list_basis[0]) != len(list_basis):
            raise NotImplementedError(
                "Not implemented for FDataBasis with different basis.",
            )

    # FData where each sample is the mean of each group
    fd_means = concatenate([fd.mean() for fd in fd_groups])

    # Base statistic
    vn = v_sample_stat(fd_means, [fd.n_samples for fd in fd_groups], p=p)

    # Computing sampling distribution
    simulation = _anova_bootstrap(
        fd_groups,
        n_reps,
        random_state=random_state,
        p=p,
        equal_var=equal_var,
    )

    p_value = np.sum(simulation > vn) / len(simulation)

    if return_dist:
        return vn, p_value, simulation

    return vn, p_value
