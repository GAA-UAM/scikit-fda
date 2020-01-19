from skfda import FDataGrid
import numpy as np
from skfda.inference.anova.anova_oneway import func_oneway, func_oneway_usc
from skfda.datasets import make_gaussian_process
from matplotlib import pyplot as plt


def generate_samples_independent(mean, sigma, n_samples):
    return [mean + np.random.normal(0, sigma, len(mean)) for _ in range(n_samples)]


scale = 25

start = 0
stop = 1

n_levels = 3
n_samples = 100

t = np.linspace(start, stop, scale)

sigmas = np.array([0, 0.2, 1, 1.8, 2.6, 3.4, 4.2, 5])
sigmas_star = sigmas * scale

# Case M1
mean1 = t * (1 - t)
mean2 = t * (1 - t)
mean3 = t * (1 - t)

fd_means = FDataGrid([mean1, mean2, mean3])
fd_means.plot()
plt.show()

p = []
reps = 500

for i in range(reps):
    if i % 100 == 1 and i != 1:
        print(np.mean(p))
        p = []

    print('Simulation {}...'.format(i + 1))
    samples1 = generate_samples_independent(mean1, sigmas_star[1], n_samples)
    samples2 = generate_samples_independent(mean2, sigmas_star[1], n_samples)
    samples3 = generate_samples_independent(mean3, sigmas_star[1], n_samples)

    # Storing in FDataGrid
    fd_1 = FDataGrid(samples1, sample_points=t, dataset_label="Process 1")
    fd_2 = FDataGrid(samples2, sample_points=t, dataset_label="Process 2")
    fd_3 = FDataGrid(samples3, sample_points=t, dataset_label="Process 3")
    fd_total = fd_1.concatenate(fd_2.concatenate(fd_3))

    p.append(func_oneway(fd_1, fd_2, fd_3, n_sim=2000)[0])

print(np.mean(p))
