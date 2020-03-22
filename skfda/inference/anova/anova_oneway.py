import numpy as np
from sklearn.utils import check_random_state

from skfda.misc.metrics import norm_lp
from skfda.representation import FData, FDataGrid, FDataBasis
from skfda.datasets import make_gaussian_process


def v_sample_stat(fd, weights, p=2):
    r"""
    Calculates a statistic that measures the variability between groups of
    samples in a :class:`skfda.representation.grid.FDataGrid` object.

    The statistic defined as below is calculated between all the samples in a
    :class:`skfda.representation.grid.FDataGrid` object with a given set of
    weights, and the desired :math:`L_p` norm.

    Let :math:`\{f_i\}_{i=1}^k` be a set of samples in a FDataGrid object.
    Let :math:`\{w_j\}_{j=1}^k` be a set of weights, where :math:`w_i` is
    related to the sample :math:`f_i` for :math:`i=1,\dots,k`.
    The statistic is defined as:

    .. math::
        V_n = \sum_{i<j}^kw_i\|f_i-f_j\|^p

    This statistic is defined in Cuevas[1].

    Args:
         fd (FDataGrid): Object containing all the samples for which we want
            to calculate the statistic.
         weights (list of int): Weights related to each sample. Each
            weight is expected to appear in the same position as its
            corresponding sample in the FDataGrid object.
         p (int, optional): p of the lp norm. Must be greater or equal
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
        >>> fd = FDataGrid([x1, x2, x3], sample_points=t)
        >>> weights = [10, 20, 30]

        Finally the value of the statistic is calculated:

        >>> v_sample_stat(fd, weights)
        0.01649448843348894

    References:
        [1] Antonio Cuevas, Manuel Febrero-Bande, and Ricardo Fraiman. "An
        anova test for functional data". *Computational Statistics  Data
        Analysis*, 47:111-112, 02 2004
    """

    if not isinstance(fd, FData):
        raise ValueError("Argument type must inherit FData.")
    if len(weights) != fd.n_samples:
        raise ValueError("Number of weights must match number of samples.")
    if isinstance(fd, FDataBasis):
        raise NotImplementedError("Not implemented for FDataBasis objects.")

    n = fd.n_samples
    v_n = 0
    for j in range(n):
        for i in range(j):
            v_n += weights[i] * norm_lp(fd[i] - fd[j], p=p) ** p
    return v_n


def v_asymptotic_stat(fd, weights, p=2):
    r"""
    Calculates a statistic that measures the variability between groups of
    samples in a :class:`skfda.representation.grid.FDataGrid` object.

    The statistic defined as below is calculated between all the samples in a
    :class:`skfda.representation.grid.FDataGrid` object with a given set of
    weights, and the desired :math:`L_p` norm.

    Let :math:`\{f_i\}_{i=1}^k` be a set of samples in a FDataGrid object.
    Let :math:`\{w_j\}_{j=1}^k` be a set of weights, where :math:`w_i` is
    related to the sample :math:`f_i` for :math:`i=1,\dots,k`.
    The statistic is defined as:

    .. math::
        \sum_{i<j}^k\|f_i-f_j\sqrt{\cfrac{w_i}{w_j}}\|^p

    This statistic is defined in Cuevas[1].

    Args:
         fd (FDataGrid): Object containing all the samples for which we want
            to calculate the statistic.
         weights (list of int): Weights related to each sample. Each
            weight is expected to appear in the same position as its
            corresponding sample in the FDataGrid object.
         p (int, optional): p of the lp norm. Must be greater or equal
            than 1. If p='inf' or p=np.inf it is used the L infinity metric.
            Defaults to 2.

    Returns:
        The value of the statistic.

    Raises:
        ValueError.

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
        >>> fd = FDataGrid([x1, x2, x3], sample_points=t)
        >>> weights = [10, 20, 30]

        Finally the value of the statistic is calculated:

        >>> v_asymptotic_stat(fd, weights)
        0.0018159320335885969

    References:
        [1] Antonio Cuevas, Manuel Febrero-Bande, and Ricardo Fraiman. "An
        anova test for functional data". *Computational Statistics  Data
        Analysis*, 47:111-112, 02 2004
    """
    if not isinstance(fd, FData):
        raise ValueError("Argument type must inherit FData.")
    if len(weights) != fd.n_samples:
        raise ValueError("Number of weights must match number of samples.")
    if isinstance(fd, FDataBasis):
        raise NotImplementedError("Not implemented for FDataBasis objects.")

    n = fd.n_samples
    v = 0
    for j in range(n):
        for i in range(j):
            v += norm_lp(
                fd[i] - fd[j] * np.sqrt(weights[i] / weights[j]), p=p) ** p
    return v


def _anova_bootstrap(fd_grouped, n_sim, p=2, random_state=None):
    assert len(fd_grouped) > 0

    n_groups = len(fd_grouped)
    sample_points = fd_grouped[0].sample_points
    m = len(sample_points[0])  # Number of points in the grid
    start, stop = fd_grouped[0].domain_range[0]

    sizes = [fd.n_samples for fd in fd_grouped]  # List with sizes of each group

    # Estimating covariances for each group
    k_est = [fd.cov().data_matrix[0, ..., 0] for fd in fd_grouped]

    # Instance a random state object in case random_state is an int
    random_state = check_random_state(random_state)

    # Simulating n_sim observations for each of the n_groups gaussian processes
    sim = [make_gaussian_process(n_sim, n_features=m, start=start, stop=stop,
                                 cov=k_est[i], random_state=random_state)
           for i in range(n_groups)]
    v_samples = np.empty(n_sim)
    for i in range(n_sim):
        fd = FDataGrid([s.data_matrix[i, ..., 0] for s in sim])
        v_samples[i] = v_asymptotic_stat(fd, sizes, p=p)
    return v_samples


def oneway_anova(*args, n_sim=2000, p=2, return_dist=False, random_state=None):
    r"""
    Performs one-way functional ANOVA.

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
    calculated. This procedure is repeated `n_sim` times, creating a
    sampling distribution of the statistic.

    This procedure is from Cuevas[1].

    Args:
        fd1,fd2,.... (FDataGrid): The sample measurements for each each group.

        n_sim (int, optional): Number of simulations for the bootstrap
            procedure. Defaults to 2000 (This value may change in future
            versions).

        p (int, optional): p of the lp norm. Must be greater or equal
            than 1. If p='inf' or p=np.inf it is used the L infinity metric.
            Defaults to 2.

        return_dist (bool, optional): Flag to indicate if the function should
        return a
            numpy.array with the sampling distribution simulated.

        random_state (optional): Random state.

    Returns:
        Value of the sample statistic, p-value and sampling distribution of
        the simulated asymptotic statistic.

    Return type:
        (float, float, numpy.array)

    Raises:
        ValueError: In case of bad arguments.

    Examples:
        >>> from skfda.inference.anova import oneway_anova
        >>> from skfda.datasets import fetch_gait
        >>> from numpy.random import RandomState

        >>> fd = fetch_gait()["data"].coordinates[1]
        >>> fd1, fd2, fd3 = fd[:13], fd[13:26], fd[26:]
        >>> oneway_anova(fd1, fd2, fd3, random_state=RandomState(42))
        (179.52499999999998, 0.602)
        >>> oneway_anova(fd1, fd2, fd3, p=1, random_state=RandomState(42))
        (67.27499999999999, 0.0)
        >>> _, _, dist = oneway_anova(fd1, fd2, fd3, n_sim=3,
        ...     random_state=RandomState(42),
        ...     return_dist=True)
        >>> print(dist)
        [163.35765183 208.59495097 229.76780354]



    References:
        [1] Antonio Cuevas, Manuel Febrero-Bande, and Ricardo Fraiman. "An
        anova test for functional data". *Computational Statistics  Data
        Analysis*, 47:111-112, 02 2004
    """

    if len(args) < 2:
        raise ValueError("At least two samples must be passed as parameter.")
    if not all(isinstance(fd, FData) for fd in args):
        raise ValueError("Argument type must inherit FData.")
    if n_sim < 1:
        raise ValueError("Number of simulations must be positive.")
    if any(isinstance(fd, FDataBasis) for fd in args):
        raise NotImplementedError("Not implemented for FDataBasis objects.")

    fd_groups = args
    fd_means = fd_groups[0].mean()
    for fd in fd_groups[1:]:
        fd_means = fd_means.concatenate(fd.mean())

    vn = v_sample_stat(fd_means, [fd.n_samples for fd in fd_groups], p=p)

    simulation = _anova_bootstrap(fd_groups, n_sim, p=p,
                                  random_state=random_state)
    p_value = np.sum(simulation > vn) / len(simulation)

    if return_dist:
        return vn, p_value, simulation

    return vn, p_value
