#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
from scipy import stats

import matplotlib.pyplot as plt

from fda.basis import FDataBasis, BSpline
from fda.grid import FDataGrid

# Data parameters
nsamples = 9  # Number of samples
nfine = 100  # Number of points per sample
sd = .1
shift_sd = .1  # Standard deviation of phase variation
amp_sd = 0 #.1  # Standard deviation of amplitude variation
error_sd = .05  # Standard deviation of gaussian noise
xlim = (-1, 1) # Domain range

# Basis parameters
nbasis = 7  # Number of fourier basis elements
nknots = 20

# Registration parameters
maxiter = 20
tol = 1e-5

# Plot options
width = .8  # Width of the samples curves
samples_color = 'teal'
mean_color = 'black'
curve_color = 'maroon'
ylim = None
iterations = 5  # Number of iterations in the step-by-step figure

def noise_gaussian(t, nsamples, sd, shift_sd, amp_sd, error_sd):
    """Noisy Gaussian curve function

    Args:
        t (ndarray): Array of times
        nsamples (float): Number of samples
        period (float): Period of the gaussian function sin(2*pi*t/period)
        shift_sd: Standard deviation of the shift variation normaly distributed
        amp_sd: Standard deviation of the amplitude variation
        error_sd: Standard deviation of the error of the samples

    Returns:
        ndarray with the samples evaluated. Each row is a sample and each
        column is a discrete time.
    """

    shift_variation = np.outer(np.random.normal(0, shift_sd,  nsamples),
                               np.ones(len(t)))

    error = np.random.normal(0, error_sd, (nsamples, len(t)))

    amp = np.diag(np.random.normal(1, amp_sd, nsamples))

    tsamples = t - shift_variation

    return amp @ stats.norm.pdf(tsamples, 0, sd) + error



if __name__ == '__main__':

    # Matplotlib stylesheet
    plt.style.use('seaborn')

    # Fixing random state for reproducibility
    #np.random.seed(987654)

    # Matrix with times where each sample will be evaluated
    t = np.linspace(xlim[0], xlim[1], nfine)

    # Noisy gaussian data, with amplitude variation and gaussian error
    data = noise_gaussian(t, nsamples, sd, shift_sd, amp_sd, error_sd)

    # Real gaussian function
    gaussian = stats.norm.pdf(t, 0, sd)

    # Plot the samples
    plt.figure()
    plt.title('Raw data')
    #plt.ylim(ylim)
    plt.xlim(xlim)
    l1 = plt.plot(t, data.T, label='samples', c=samples_color, linewidth=width)
    l2 = plt.plot(t, gaussian, label='gaussian', c=curve_color, linestyle='dashed')
    l3 = plt.plot(t, np.mean(data.T, axis=1), label='mean', c=mean_color)
    plt.legend(handles=[l1[0], l3[0], l2[0]], loc=1)


    knots = np.cos (np.pi * np.arange(nknots + 1) / nknots)
    knots = np.linspace(-1,1,nknots)
    # Curves smoothed with the matrix penalty method
    fd = FDataBasis.from_data(data, t, BSpline(xlim, knots=knots))
    unregmean = fd.mean()  # Mean of unregistered curves

    # Plots the smoothed curves
    plt.figure()
    plt.title('Unregistered curves')
    #plt.ylim(ylim)
    plt.xlim(xlim)
    l1 = fd.plot(label='samples', c=samples_color, linewidth=width)
    l2 = plt.plot(t, gaussian, label='gaussian', c=curve_color, linestyle='dashed')
    l3 = unregmean.plot(label='mean', c=mean_color)
    plt.legend(handles=[l1[0], l3[0], l2[0]], loc=1)

    # Shift registered curves
    regbasis = fd.shift_registration(maxiter=maxiter,tol=tol)
    regmean = regbasis.mean()  # Registered mean

    # Plots the registered curves
    plt.figure()
    plt.title('Registered curves')
    #plt.ylim(ylim)
    plt.xlim(xlim)
    l1 = regbasis.plot(label='samples', c=samples_color,
                       linewidth=width)
    l2 = plt.plot(t, gaussian, label='gaussian', c=curve_color, linestyle='dashed')
    l3 = regmean.plot(label='mean', c=mean_color)
    plt.legend(handles=[l1[0], l3[0], l2[0]], loc=1)

    # Plots the process step by step
    f, axarr = plt.subplots(iterations+1, 1, sharex=True, sharey=True)
    axarr[0].title.set_text('Step by step registration')
    plt.xlim(xlim)

    fd.plot(ax=axarr[0], c=samples_color, linewidth=width)
    axarr[0].set_ylabel('Unregistered')

    for i in range(1, iterations+1):
        # tol=0 to realize all the iterations
        regfd = fd.shift_registration(maxiter=i, tol=0.)
        regfd.plot(ax=axarr[i], c=samples_color, linewidth=width)
        axarr[i].set_ylabel('%d iteration%s' % (i, '' if i == 1 else 's'))

    plt.show()
