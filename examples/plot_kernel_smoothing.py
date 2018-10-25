"""
Kernel Smoothing
================

This example uses different kernel smoothing methods over the phoneme data
set and shows how cross validations scores vary over a range of different
parameters used in the smoothing methods. It also show examples of
undersmoothing and oversmoothing.
"""

import fda
import fda.kernel_smoothers as ks
import fda.validation as val
import matplotlib.pylab as plt
import numpy as np

###############################################################################
#
# For this example, we will use the
# :func:`phoneme <fda.datasets.fetch_phoneme>` dataset. This dataset
# contains the log-periodograms of several phoneme pronunciations. The phoneme
# curves are very irregular and noisy, so we usually will want to smooth them
# as a preprocessing step.
#
# As an example, we will smooth the first 300 curves only. In the following
# plot, the first five curves are shown.

dataset = fda.datasets.fetch_phoneme()
fd = dataset['data'][:300]

fd[0:5].plot()

###############################################################################
# Here we show the general cross validation scores for different values of the
# parameters given to the different smoothing methods.

param_values = np.linspace(start=2, stop=25, num=24)

# Local linear regression kernel smoothing.
llr = val.minimise(fd, param_values,
                   smoothing_method=ks.local_linear_regression)
# Nadaraya-Watson kernel smoothing.
nw = fda.validation.minimise(fd, param_values,
                             smoothing_method=ks.nw)
# K-nearest neighbours kernel smoothing.
knn = fda.validation.minimise(fd, param_values,
                              smoothing_method=ks.knn)

plt.plot(param_values, knn['scores'])
plt.plot(param_values, llr['scores'])
plt.plot(param_values, nw['scores'])

ax = plt.gca()
ax.set_xlabel('Smoothing method parameter')
ax.set_ylabel('GCV score')
ax.set_title('Scores through GCV for different smoothing methods')
ax.legend(['k-nearest neighbours', 'local linear regression',
           'Nadaraya-Watson'],
          title='Smoothing method')

###############################################################################
# We can plot the smoothed curves corresponding to the 11th element of the data
# set (this is a random choice) for the three different smoothing methods.

fd[10].plot()
knn['fdatagrid'][10].plot()
llr['fdatagrid'][10].plot()
nw['fdatagrid'][10].plot()
ax = plt.gca()
ax.legend(['original data', 'k-nearest neighbours',
           'local linear regression',
           'Nadaraya-Watson'],
          title='Smoothing method')

###############################################################################
# We can compare the curve before and after the smoothing.

# Not smoothed
fd[10].plot()

# Smoothed
plt.figure()
fd[10].scatter(s=0.5)
nw['fdatagrid'][10].plot(c='g')

###############################################################################
# Now, we can see the effects of a proper smoothing. We can plot the same 5
# samples from the beginning using the Nadaraya-Watson kernel smoother with
# the best choice of parameter.

plt.figure(4)
nw['fdatagrid'][0:5].plot()

###############################################################################
# We can also appreciate the effects of undersmoothing and oversmoothing in
# the following plots.

fd_us = fda.FDataGrid(
    ks.nw(fd.sample_points, h=2).dot(fd.data_matrix[10, ..., 0]),
    fd.sample_points, fd.sample_range, fd.dataset_label,
    fd.axes_labels)
fd_os = fda.FDataGrid(
    ks.nw(fd.sample_points, h=15).dot(fd.data_matrix[10, ..., 0]),
    fd.sample_points, fd.sample_range, fd.dataset_label,
    fd.axes_labels)

# Under-smoothed
fd[10].scatter(s=0.5)
fd_us.plot(c='sandybrown')

# Over-smoothed
plt.figure()
fd[10].scatter(s=0.5)
fd_os.plot(c='r')
