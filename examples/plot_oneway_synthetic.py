"""
One-way functional ANOVA with synthetic data
============================================

This example shows how to perform a functional one-way ANOVA test with
synthetic data.
"""

# Author: David García Fernández
# License: MIT

import skfda
from skfda.inference.anova import oneway_anova
from skfda.representation import FDataGrid

################################################################################
# *One-way ANOVA* (analysis of variance) is a test that can be used to
# compare the means of different samples of data.
# Let :math:`X_{ij}(t), j=1, \dots, n_i` be trajectories corresponding to
# :math:`k` independent samples :math:`(i=1,\dots,k)` and let :math:`E(X_i(t)) =
# m_i(t)`. Thus, the null hypothesis in the statistical test is:
#
# .. math::
#   H_0: m_1(t) = \dots = m_k(t)
#
# In this example we will explain the nature of ANOVA method and its behavior
# under certain conditions simulating data. Specifically, we will generate
# three different trajectories, for each one we will simulate a stochastic
# process by adding to them brownian processes. The main objective of the
# test is to illustrate the differences in the results of the ANOVA method
# when the covariance function of the brownian processes changes.

import numpy as np

import skfda
from skfda.representation import FDataGrid
from skfda.inference.anova import oneway_anova
from skfda.datasets import make_gaussian_process

################################################################################
# First, the means for the future processes are drawn.

n_samples = 100
n_features = 50
n_groups = 3

t = np.linspace(-np.pi, np.pi, n_features)

m1 = np.sin(t)
m2 = 1.1 * np.sin(t)
m3 = 1.2 * np.sin(t)

_ = FDataGrid([m1, m2, m3],
              dataset_label="Means to be used in the simulation").plot()


###############################################################################
# Now, a function to simulate processes as described above is implemented,
# to make code clearer.

def make_process_b_noise(mean, cov, random_state):
    return FDataGrid([mean for _ in range(n_samples)]) \
           + make_gaussian_process(n_samples, n_features=mean.shape[0],
                                   cov=cov, random_state=random_state)


################################################################################
# A total of `n_samples` trajectories will be created for each mean, so a array
# of labels is created to identify them when plotting.

groups = np.full(n_samples * n_groups, 'Sample 1')
groups[100:200] = 'Sample 2'
groups[200:] = 'Sample 3'

###############################################################################
# First simulation uses a low :math:`\sigma = 0.1` value. In this case the
# differences between the means of each group should be clear, and the
# p-value for the test should be near to zero.

sigma = 0.1
cov = np.identity(n_features) * sigma

fd1 = make_process_b_noise(m1, cov, random_state=1)
fd2 = make_process_b_noise(m2, cov, random_state=2)
fd3 = make_process_b_noise(m3, cov, random_state=3)

stat, p_val = oneway_anova(fd1, fd2, fd3, random_state=1)
print("Statistic: ", stat)
print("p-value: ", p_val)

################################################################################
# In the plot below we can see the simulated trajectories for each mean,
# and the averages for each group.

fd = fd1.concatenate(fd2.concatenate(fd3.concatenate()))
fd.dataset_label = f"Sample with $\sigma$ = {sigma}, p-value = {p_val}"
fd.plot(group=groups, legend=True)
fd1.mean().concatenate(fd2.mean().concatenate(fd3.mean()).concatenate()).plot()

################################################################################
# In the following, the same process will be followed incrementing sigma
# value, this way the differences between the averages of each group will be
# lower and the p-values will increase (the null hypothesis will be harder to
# refuse).

################################################################################
# Plot for :math:`\sigma = 1`:

sigma = 1
cov = np.identity(n_features) * sigma

fd1 = make_process_b_noise(m1, cov, random_state=1)
fd2 = make_process_b_noise(m2, cov, random_state=2)
fd3 = make_process_b_noise(m3, cov, random_state=3)

_, p_val = oneway_anova(fd1, fd2, fd3, random_state=1)

fd = fd1.concatenate(fd2.concatenate(fd3.concatenate()))
fd.dataset_label = f"Sample with $\sigma$ = {sigma}, p-value = {p_val}"
fd.plot(group=groups, legend=True)
fd1.mean().concatenate(fd2.mean().concatenate(fd3.mean()).concatenate()).plot()

################################################################################
# Plot for :math:`\sigma = 10`:

sigma = 10
cov = np.identity(n_features) * sigma

fd1 = make_process_b_noise(m1, cov, random_state=1)
fd2 = make_process_b_noise(m2, cov, random_state=2)
fd3 = make_process_b_noise(m3, cov, random_state=3)

_, p_val = oneway_anova(fd1, fd2, fd3, random_state=1)

fd = fd1.concatenate(fd2.concatenate(fd3.concatenate()))
fd.dataset_label = f"Sample with $\sigma$ = {sigma}, p-value = {p_val}"
fd.plot(group=groups, legend=True)
fd1.mean().concatenate(fd2.mean().concatenate(fd3.mean()).concatenate()).plot()

################################################################################
# **References:**
#
#  [1] Antonio Cuevas, Manuel Febrero-Bande, and Ricardo Fraiman. "An anova test
#  for functional data". *Computational Statistics  Data Analysis*,
#  47:111-112, 02 2004
