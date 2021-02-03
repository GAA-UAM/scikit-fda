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
# parameters given to the different smoothing methods. Currently we have
# three kernel smoothing methods implemented: Nadaraya Watson, Local Linear
# Regression and K Nearest Neighbors (k-NN)

##############################################################################
# The smoothing parameter for k-NN is the number of neighbors. We will choose
# this parameter between 1 and 23 in this example.

n_neighbors = np.arange(1, 24)

##############################################################################
# The smoothing parameter for Nadaraya Watson and Local Linear Regression is
# a bandwidth parameter, with the same units as the domain of the function.
# As we want to compare the results of these smoothers with k-NN, with uses
# as the smoothing parameter the number of neighbors, we want to use a
# comparable range of values. In this case, we know that our grid points are
# equispaced, so a given bandwidth ``B`` will include
# ``B * N / D`` grid points, where ``N`` is the total number of grid points
# and ``D`` the size of the whole domain range. Thus, if we pick
# ``B = n_neighbors * D / N``, ``B`` will include ``n_neighbors`` grid points
# and we could compare the results of the different smoothers.

scale_factor = (
    (fd.domain_range[0][1] - fd.domain_range[0][0])
    / len(fd.grid_points[0])
)

bandwidth = n_neighbors * scale_factor

# K-nearest neighbours kernel smoothing.
knn = val.SmoothingParameterSearch(
    ks.KNeighborsSmoother(),
    n_neighbors,
)
knn.fit(fd)
knn_fd = knn.transform(fd)

# Local linear regression kernel smoothing.
llr = val.SmoothingParameterSearch(
    ks.LocalLinearRegressionSmoother(),
    bandwidth,
)
llr.fit(fd)
llr_fd = llr.transform(fd)

# Nadaraya-Watson kernel smoothing.
nw = val.SmoothingParameterSearch(
    ks.NadarayaWatsonSmoother(),
    bandwidth,
)
nw.fit(fd)
nw_fd = nw.transform(fd)

##############################################################################
# The plot of the mean test scores for all smoothers is shown below.
# As the X axis we will use the neighbors for all the smoothers in order
# to compare k-NN with the others, but remember that the bandwidth is
# this quantity scaled by ``scale_factor``!

fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)
ax.plot(
    n_neighbors,
    knn.cv_results_['mean_test_score'],
    label='k-nearest neighbors',
)
ax.plot(
    n_neighbors,
    llr.cv_results_['mean_test_score'],
    label='local linear regression',
)
ax.plot(
    n_neighbors,
    nw.cv_results_['mean_test_score'],
    label='Nadaraya-Watson',
)
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
ax.legend(
    [
        'original data',
        'k-nearest neighbors',
        'local linear regression',
        'Nadaraya-Watson',
    ],
    title='Smoothing method',
)
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
    smoothing_parameter=2 * scale_factor,
).fit_transform(fd[10])
fd_os = ks.NadarayaWatsonSmoother(
    smoothing_parameter=15 * scale_factor,
).fit_transform(fd[10])

##############################################################################
# Under-smoothed

fig = fd[10].scatter(s=0.5)
fd_us.plot(fig=fig)

##############################################################################
# Over-smoothed

fig = fd[10].scatter(s=0.5)
fd_os.plot(fig=fig)
