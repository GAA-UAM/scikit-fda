from skfda import FDataGrid
from skfda.datasets import make_gaussian_process
import matplotlib.pyplot as plt
import numpy as np
import sklearn
from skfda.misc.metrics import lp_distance
from statsmodels.distributions.empirical_distribution import ECDF
from skfda.inference.anova.anova_oneway import  func_oneway

def generate_samples_independent(mean, sigma, n_samples):
    return [mean + np.random.normal(0, sigma, len(mean)) for _ in range(n_samples)]


# Cuevas simulation study
grid = np.linspace(0, 1, 25)
n_levels = 3

# Case M2
mean1 = np.vectorize(lambda t: t*(1-t)**5)(grid)
mean2 = np.vectorize(lambda t: t**2*(1-t)**4)(grid)
mean3 = np.vectorize(lambda t: t**3*(1-t)**3)(grid)

fd_means = FDataGrid([mean1, mean2, mean3])

samples1 = generate_samples_independent(mean1, 0.2/25, 10)
samples2 = generate_samples_independent(mean2, 0.2/25, 10)
samples3 = generate_samples_independent(mean3, 0.2/25, 10)

# Storing in FDataGrid
fd_1 = FDataGrid(samples1, sample_points=grid, dataset_label="Process 1")
fd_2 = FDataGrid(samples2, sample_points=grid, dataset_label="Process 2")
fd_3 = FDataGrid(samples3, sample_points=grid, dataset_label="Process 3")
fd_total = fd_1.concatenate(fd_2.concatenate(fd_3))

# print(fd_total.data_matrix[0])
# print(np.squeeze(np.take(fd_total.data_matrix, np.array([0, 3]), axis=0)))

# Anova


def f_oneway(*args):

    if len(args) < 1:
        return
    # fd_total = args[0].concatenate(*args[1:])
    N = 2000
    alpha = 0.05

    simulations = [np.squeeze(make_gaussian_process(N, len(p.sample_points[0]), cov=np.squeeze(p.cov().data_matrix[0])).data_matrix) for p in args]

    ecdf = np.array([])
    for l in range(N):
        for i in range(len(simulations)):
            for j in range(i + 1, len(simulations)):
                ecdf = np.append(ecdf, np.linalg.norm(simulations[i][l] - (np.sqrt(1)) * simulations[j][l]))

    v_alpha = np.quantile(ecdf, 1 - alpha)
    F = ECDF(ecdf)
    print(v_alpha)


func_oneway(fd_total, np.array(['a' for _ in range(10)] + [ 'b' for _ in range(10)] + ['c' for _ in range(10)]), 100)


















