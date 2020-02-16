import numpy as np
from skfda.misc.metrics import norm_lp
from skfda.representation import FDataGrid
from skfda.datasets import make_gaussian_process


def v_sample_stat(fd, weights, p=2):
    """
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
        TODO

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
    k = fd.n_samples
    v_n = 0
    for i in range(k):
        for j in range(i + 1, k):
            v_n += weights[i] * norm_lp(fd[i] - fd[j], p=p) ** p
    return v_n


def v_asymptotic_stat(fd, weights, p=2):
    """
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
        TODO

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

    k = fd.n_samples
    v = 0
    for i in range(k):
        for j in range(i + 1, k):
            v += norm_lp(
                fd[i] - fd[j] * np.sqrt(weights[i] / weights[j]), p=p) ** 2
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

    # Simulating n_sim observations for each of the n_groups gaussian processes
    sim = [make_gaussian_process(n_sim, n_features=m, start=start, stop=stop,
                                 cov=k_est[i], random_state=random_state)
           for i in range(n_groups)]
    v_samples = np.array([])
    for i in range(n_sim):
        fd = FDataGrid([s.data_matrix[i, ..., 0] for s in sim])
        v_samples = np.append(v_samples, v_asymptotic_stat(fd, sizes, p=p))
    return v_samples


def oneway_anova(*args, n_sim=2000, p=2, return_dist=False, random_state=None):
    """
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
            procedure. Defaults to 2000.

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
        TODO

    References:
        [1] Antonio Cuevas, Manuel Febrero-Bande, and Ricardo Fraiman. "An
        anova test for functional data". *Computational Statistics  Data
        Analysis*, 47:111-112, 02 2004
    """

    assert len(args) > 0

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


'''
def v_usc(values):
    """

    :rtype: object
    """
    k = len(values)
    v = 0
    for i in range(k):
        for j in range(i + 1, k):
            v += norm_lp(values[i] - values[j])
    return v


def anova_bootstrap_usc(fd_grouped, n_sim):
    assert len(fd_grouped) > 0

    m = fd_grouped[0].ncol
    samples = fd_grouped[0].sample_points
    start, stop = fd_grouped[0].domain_range[0]
    sizes = [fd.n_samples for fd in fd_grouped]

    # Estimating covariances for each group
    k_est = [fd.cov().data_matrix[0, ..., 0] for fd in fd_grouped]

    l_vector = []
    for l in range(n_sim):
        sim = FDataGrid(np.empty((0, m)), sample_points=samples)
        for i, fd in enumerate(fd_grouped):
            process = make_gaussian_process(1, n_features=m, start=start,
                                            stop=stop, cov=k_est[i])
            sim = sim.concatenate(process)
        l_vector.append(v_usc(sim))

    return l_vector


def anova_oneway_usc(*args, n_sim=2000):
    # TODO Check grids

    assert len(args) > 0

    fd_groups = args
    fd_means = fd_groups[0].mean()
    for fd in fd_groups[1:]:
        fd_means = fd_means.concatenate(fd.mean())

    vn = v_usc(fd_means)

    simulation = anova_bootstrap_usc(fd_groups, n_sim=n_sim)
    p_value = len(np.where(simulation >= vn)[0]) / len(simulation)

    return p_value, vn, simulation
'''
