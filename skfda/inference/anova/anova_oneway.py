import numpy as np
from skfda.misc.metrics import lp_distance
from skfda.representation import FDataGrid
import matplotlib.pyplot as plt


def vn_statistic(fd_means, sizes):
    # Calculating weighted sum of L2 distances between means
    distances_m = np.tril(lp_distance(fd_means, fd_means)) # lp_distance not working as expected
    # Calculating square of the distances and summing by groups
    distances_group = np.sum(np.multiply(distances_m, distances_m), axis=1)
    # Weighted sum
    return sum(distances_group * sizes)


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

# func_oneway(None, np.array(['a', 'b', 'a', 'a']), 1000)

