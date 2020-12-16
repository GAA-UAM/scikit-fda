"""
Kernel Smoothing
================

This example uses different kernel smoothing methods over the phoneme data
set and shows how cross validations scores vary over a range of different
parameters used in the smoothing methods. It also show examples of
undersmoothing and oversmoothing.
"""

# Author: Miguel Carbajo Berrocal
# License: MIT

import matplotlib.pylab as plt
import numpy as np

import skfda
import skfda.preprocessing.smoothing.kernel_smoothers as ks
import skfda.preprocessing.smoothing.validation as val

##############################################################################
#
# For this example, we will use the
# :func:`phoneme <skfda.datasets.fetch_phoneme>` dataset. This dataset
# contains the log-periodograms of several phoneme pronunciations. The phoneme
# curves are very irregular and noisy, so we usually will want to smooth them
# as a preprocessing step.
#
# As an example, we will smooth the first 300 curves only. In the following
# plot, the first five curves are shown.
dataset = skfda.datasets.fetch_phoneme()
fd = dataset['data'][:300]

fd[0:5].plot()

##############################################################################
# Here we show the general cross validation scores for different values of the
# parameters given to the different smoothing methods.

param_values_knn = np.arange(1, 24, 2)
param_values_others = param_values_knn / 32

# Local linear regression kernel smoothing.
llr = val.SmoothingParameterSearch(
    ks.LocalLinearRegressionSmoother(), param_values_others)
llr.fit(fd)
llr_fd = llr.transform(fd)

# Nadaraya-Watson kernel smoothing.
nw = val.SmoothingParameterSearch(
    ks.NadarayaWatsonSmoother(), param_values_others)
nw.fit(fd)
nw_fd = nw.transform(fd)

# K-nearest neighbours kernel smoothing.
knn = val.SmoothingParameterSearch(
    ks.KNeighborsSmoother(), param_values_knn)
knn.fit(fd)
knn_fd = knn.transform(fd)

fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)
ax.plot(param_values_knn, knn.cv_results_['mean_test_score'],
        label='k-nearest neighbors')
ax.plot(param_values_knn, llr.cv_results_['mean_test_score'],
        label='local linear regression')
ax.plot(param_values_knn, nw.cv_results_['mean_test_score'],
        label='Nadaraya-Watson')
ax.legend()
fig

##############################################################################
# We can plot the smoothed curves corresponding to the 11th element of the
# data set (this is a random choice) for the three different smoothing
# methods.

fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)
ax.set_xlabel('Smoothing method parameter')
ax.set_ylabel('GCV score')
ax.set_title('Scores through GCV for different smoothing methods')

fd[10].plot(fig=fig)
knn_fd[10].plot(fig=fig)
llr_fd[10].plot(fig=fig)
nw_fd[10].plot(fig=fig)
ax.legend(['original data', 'k-nearest neighbors',
           'local linear regression',
           'Nadaraya-Watson'],
          title='Smoothing method')
fig

##############################################################################
# We can compare the curve before and after the smoothing.

##############################################################################
# Not smoothed

fd[10].plot()

##############################################################################
# Smoothed

fig = fd[10].scatter(s=0.5)
nw_fd[10].plot(fig=fig, color='green')
fig

##############################################################################
# Now, we can see the effects of a proper smoothing. We can plot the same 5
# samples from the beginning using the Nadaraya-Watson kernel smoother with
# the best choice of parameter.

nw_fd[0:5].plot()

##############################################################################
# We can also appreciate the effects of undersmoothing and oversmoothing in
# the following plots.

fd_us = ks.NadarayaWatsonSmoother(
    smoothing_parameter=2 / 32).fit_transform(fd[10])
fd_os = ks.NadarayaWatsonSmoother(
    smoothing_parameter=15 / 32).fit_transform(fd[10])

##############################################################################
# Under-smoothed

fig = fd[10].scatter(s=0.5)
fd_us.plot(fig=fig)

##############################################################################
# Over-smoothed

fig = fd[10].scatter(s=0.5)
fd_os.plot(fig=fig)
