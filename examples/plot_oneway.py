"""
One-way functional ANOVA
========================

This example shows how to perform a functional one-way ANOVA test.
"""

# Author: David García Fernández
# License: MIT

# sphinx_gallery_thumbnail_number = 2

import numpy as np
import skfda
from skfda.inference.anova import func_oneway

################################################################################
# *One-way ANOVA* (analysis of variance) is a test that can be used to
# compare the means of different samples of data.
# Let :math:`X_{ij}(t), j=1, \dots, n_i` be trajectories corresponding to
# :math:`k` independent samples :math:`(i=1,\dots,k)` and let :math:`E(X_i(t)) =
# m_i(t)`. Thus, the null hypotesis in the statistical test is:
#
# .. math::
#   H_0: m_0(t) = m_1(t) = \dots = m_k(t)
#
# In this particular example we are going to use the Spanish Weather dataset,
# with information about the average temperature for the period 1980-2009 in
# meteorological stations of different provinces of Spain.

dataset = skfda.datasets.fetch_aemet()

y = dataset['meta']
fd = dataset['data'][0]
meta_names = dataset['meta_names']

province = y[:, np.asarray(meta_names) == 'province'].ravel()

fig = fd.plot(group=province)

################################################################################
# In the figure above we can see different trajectories that represent the
# average temperature in an specific meteorological station. The measurements
# of stations in the same province are represented in the same color.
#
# For this example we will study only five provinces located in the
# mediterranean coast, and we will try to test the average temperatures
# equality using the *ANOVA* test.


sel_prov = ['BARCELONA', 'TARRAGONA', 'VALENCIA', 'ALICANTE', 'MURCIA']

# Creating a filter with only the selected provinces in sel_prov
filt = np.logical_or.reduce([np.asarray(province) == p for p in sel_prov])

province = province[filt]
fd = fd[filt]

fig = fd.plot(group=province, legend=True)

###############################################################################
# Now it is necessary to prepare the data. Each independent sample of data
# has to be stored in different :class:`~skfda.representation.grid.FDataGrid`
# objects. So, we need to group the measurements of the same provinces.


fd_groups = [fd.copy(data_matrix=fd.data_matrix[province == label],
                     dataset_label=fd.dataset_label + ' in ' + label)
             for label in sel_prov]

###############################################################################
# At this point is time to perform the *ANOVA* test. This functionality is
# implemented in the function :func:`~skfda.inference.anova.func_oneway`. As
# it consists in an asymptotic method it is possible to set the number of
# simulations necessary to approximate the result of the statistic. It is
# possible to set the :math:`p` of the :math:`L_p` norm used in the
# calculations (defaults 2).
#
p_val, v_n, dist = func_oneway(*fd_groups, n_sim=1500)

################################################################################
# The function returns first the *p-value* for the test, second the value of
# the statistic :func:`~skfda.inference.anova.v_sample_stat` used to measure
# the variability between groups. The third return value corresponds to the
# sampling distribution of the statistic which is compared with the previous
# one to get the *p-value*. For further information visit
# :func:`~skfda.inference.anova.func_oneway` and [1].

print('p-value: ', p_val)
print('Statistic: ', v_n)
print('Distribution: ', dist)

################################################################################
# **References:**
#
#  [1] Antonio Cuevas, Manuel Febrero-Bande, and Ricardo Fraiman. An anova test
#  for functional data. *Computational Statistics  Data Analysis*,
#  47:111-112, 02 2004
