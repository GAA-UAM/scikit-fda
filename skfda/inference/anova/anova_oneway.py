import numpy as np
from skfda.misc.metrics import lp_distance
from skfda.representation import FDataGrid
from skfda.datasets import make_gaussian_process


def vn_statistic(fd_means, sizes):
    # Calculating weighted sum of L2 distances between means
    distances_m = np.tril(lp_distance(fd_means, fd_means))  # lp_distance not working as expected
    # Calculating square of the distances and summing by groups
    distances_group = np.sum(np.multiply(distances_m, distances_m), axis=1)
    # Weighted sum
    return sum(distances_group * sizes)


def anova_bootstrap(fd_grouped, n_sim):
    if len(fd_grouped) < 1:
        return

    m = fd_grouped[0].ncol
    k = len(fd_grouped)
    start, stop = fd_grouped[0].domain_range[0]

    # Estimating covariances
    k_est = [np.squeeze(fd.cov().data_matrix[0]) for fd in fd_grouped]

    # Simulation
    simulation = np.empty((0, k, m))
    for l in range(n_sim):
        sim_l = np.empty((0, m))
        for i, fd in enumerate(fd_grouped):
            process = make_gaussian_process(n_samples=1, n_features=m, start=start,
                                            stop=stop, cov=k_est[i])
            sim_l = np.append(sim_l, [np.squeeze(process.data_matrix)], axis=0)
        simulation = np.append(simulation, [sim_l], axis=0)
    return simulation


def vn_temp(fd_means, sizes):

    means = []
    for f in fd_means.data_matrix:
        means.append(FDataGrid(np.squeeze(f), sample_points=np.squeeze(fd_means.sample_points[0])))

    v = 0

    for i in range(len(means)):
        for j in range(i + 1, len(means)):
            v += sizes[i] * lp_distance(means[i], means[j]) ** 2

    return v


def v_gorros(simulaciones, sizes):
    distr = []
    for s in simulaciones:
        v = 0
        for i in range(len(s)):
            for j in range(i + 1, len(s)):
                v += np.linalg.norm(s[i] - s[j] * np.sqrt(sizes[i] / sizes[j])) ** 2
        distr.append(v)
    return np.array(distr)


def func_oneway(fdata, groups, n_sim):
    # Obtaining the different group labels
    group_set = np.unique(groups)

    fd_groups = []
    means = None
    for group in group_set:
        # Creating an independent FDataGrid for each group
        indices = np.where(groups == group)[0]
        fd = FDataGrid(np.squeeze(np.take(fdata.data_matrix, indices, axis=0)),
                       sample_points=fdata.sample_points)
        fd_groups.append(fd)
        # Creating FDataGrid with the means of each group
        if not means:
            means = fd.mean()
        else:
            means = means.concatenate(fd.mean())

    # vn = vn_statistic(means, [fd.n_samples for fd in fd_groups])
    vn = vn_temp(means, [fd.n_samples for fd in fd_groups])

    simulation = anova_bootstrap(fd_groups, n_sim)
    v = v_gorros(simulation, [10, 10, 10])
    p_value = len(np.where(v >= vn)[0]) / len(v)

    return p_value, vn, v
