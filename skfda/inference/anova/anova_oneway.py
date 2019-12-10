import numpy as np
from skfda.misc.metrics import norm_lp, lp_distance
from skfda.representation import FDataGrid
from skfda.datasets import make_gaussian_process


def vn_statistic(fd_means, sizes):
    k = fd_means.data_matrix.shape[0]
    v_n = 0
    for i in range(k):
        for j in range(i + 1, k):
            # v1 = np.squeeze(fd_means[i].data_matrix[0])
            # v2 = np.squeeze(fd_means[j].data_matrix[0])
            # v_n += sizes[i] * np.linalg.norm(v1 - v2) ** 2
            v_n += sizes[i] * norm_lp(fd_means[i] - fd_means[j]) ** 2
    return v_n


def anova_bootstrap(fd_grouped, n_sim):
    assert len(fd_grouped) > 0

    m = fd_grouped[0].ncol
    samples = fd_grouped[0].sample_points
    k = len(fd_grouped)
    start, stop = fd_grouped[0].domain_range[0]
    sizes = [fd.n_samples for fd in fd_grouped]

    # Estimating covariances for each group
    k_est = [np.squeeze(fd.cov().data_matrix[0]) for fd in fd_grouped]

    l_vector = []
    for l in range(n_sim):
        sim = FDataGrid(np.empty((0, m)), sample_points=samples)
        for i, fd in enumerate(fd_grouped):
            process = make_gaussian_process(fd.n_samples, n_features=m, start=start, stop=stop, cov=k_est[i])
            sim = sim.concatenate(process.mean())
        # l_vector.append(v_hat_statistic(sim, sizes))
        l_vector.append(v_usc(sim))
    return l_vector


def v_hat_statistic(values, sizes):
    k = len(values)
    v_hat = 0
    for i in range(k):
        for j in range(i + 1, k):
            # v1 = np.squeeze(values[i].data_matrix[0])
            # v2 = np.squeeze(values[j].data_matrix[0])
            # v_hat += np.linalg.norm(v1 - v2 * np.sqrt(sizes[i] / sizes[j]))**2
            v_hat += norm_lp(values[i] - values[j] * np.sqrt(sizes[i] / sizes[j])) ** 2
    return v_hat


def func_oneway(*args, n_sim=2000):
    # TODO Check grids
    assert len(args) > 0

    fd_groups = args
    fd_means = fd_groups[0].mean()
    for fd in fd_groups[1:]:
        fd_means = fd_means.concatenate(fd.mean())

    # vn = vn_statistic(fd_means, [fd.n_samples for fd in fd_groups])
    vn = v_usc(fd_means)
    simulation = anova_bootstrap(fd_groups, n_sim)
    p_value = len(np.where(simulation >= vn)[0]) / len(simulation)

    return p_value, vn, simulation


def v_usc(values):
    k = len(values)
    v = 0
    for i in range(k):
        for j in range(i + 1, k):
            v += norm_lp(values[i] - values[j])
    return v
