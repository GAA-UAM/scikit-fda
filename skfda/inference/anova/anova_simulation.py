from skfda import FDataGrid
import numpy as np
from skfda.inference.anova.anova_oneway import func_oneway


def generate_samples_independent(mean, sigma, n_samples):
    return [mean + np.random.normal(0, sigma, len(mean)) for _ in
            range(n_samples)]


scale = 25

start = 0
stop = 1

n_levels = 3
n_samples = 10

t = np.linspace(start, stop, scale)

sigmas = np.array([0, 0.2, 1, 1.8, 2.6, 3.4, 4.2, 5])
sigmas_star = sigmas / scale

# Case M1
# mean1 = t * (1 - t)
# mean2 = t * (1 - t)
# mean3 = t * (1 - t)

mean1 = t * (1 - t) ** 5
mean2 = t ** 2 * (1 - t) ** 4
mean3 = t ** 3 * (1 - t) ** 3

fd_means = FDataGrid([mean1, mean2, mean3])

p = []
reps = 20

for i in range(reps):
    print('Simulation {}...'.format(i + 1))
    samples1 = generate_samples_independent(mean1, sigmas_star[2], n_samples)
    samples2 = generate_samples_independent(mean2, sigmas_star[2], n_samples)
    samples3 = generate_samples_independent(mean3, sigmas_star[2], n_samples)

    # Storing in FDataGrid
    fd_1 = FDataGrid(samples1, sample_points=t, dataset_label="Process 1")
    fd_2 = FDataGrid(samples2, sample_points=t, dataset_label="Process 2")
    fd_3 = FDataGrid(samples3, sample_points=t, dataset_label="Process 3")
    fd_total = fd_1.concatenate(fd_2.concatenate(fd_3))

    anova = func_oneway(fd_1, fd_2, fd_3)
    print(anova)
    p.append(anova[0])

print(np.mean(p))
