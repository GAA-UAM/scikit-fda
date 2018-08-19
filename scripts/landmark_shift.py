#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
from scipy import stats

import matplotlib.pyplot as plt

from fda.basis import BSpline
from fda.grid import FDataGrid

# Data parameters
nsamples = 12  # Number of samples
nfine = 100  # Number of points per sample
sd = .1 # sd of gaussian curves
shift_sd = .2  # Standard deviation of phase variation
amp_sd = .1  # Standard deviation of amplitude variation
error_sd = .05  # Standard deviation of gaussian noise
xlim = (-1, 1) # Domain range

# Location of the landmark
# Possible values are 'minimize', 'mean', 'median', 'middle' or a number
location = 'middle'
periodic = 'default'

# Basis parameters
nknots = 20

# Plot options
width = .8  # Width of the samples curves
samples_color = 'teal'
mean_color = 'black'
c_color = 'maroon'

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
        column is a discrete time and the maximun of the curves
    """

    shift = np.random.normal(0, shift_sd,  nsamples)

    shift_variation = np.outer(shift, np.ones(len(t)))

    error = np.random.normal(0, error_sd, (nsamples, len(t)))

    amp = np.diag(np.random.normal(1, amp_sd, nsamples))

    tsamples = t - shift_variation

    return amp @ stats.norm.pdf(tsamples, 0, sd) + error, shift



if __name__ == '__main__':

    # Matplotlib stylesheet
    plt.style.use('seaborn')

    # Fixing random state for reproducibility
    np.random.seed(987654)

    # Matrix with times where each sample will be evaluated
    t = np.linspace(xlim[0], xlim[1], nfine)

    # Noisy gaussian data, with amplitude variation and gaussian error
    data, shift = noise_gaussian(t, nsamples, sd, shift_sd, amp_sd, error_sd)
    fdgrid = FDataGrid(data, t, xlim, 'Raw data')

    # Real gaussian function
    gaussian = stats.norm.pdf(t, 0, sd)

    # Plot the samples
    plt.figure()
    plt.xlim(xlim)
    l1 = fdgrid.plot(label='samples', c=samples_color, linewidth=width)
    l2 = plt.plot(t, gaussian, label='gaussian', c=c_color, linestyle='dashed')
    l3 = fdgrid.mean().plot(label='mean', c=mean_color)
    plt.legend(handles=[l1[0], l3[0], l2[0]], loc=1)


    # Convert to BSplines
    fd = fdgrid.to_basis(BSpline(xlim, nbasis=nknots+2))
    unregmean = fd.mean()  # Mean of unregistered curves

    # Plots the smoothed curves
    plt.figure()
    plt.title('Unregistered curves')
    plt.xlim(xlim)
    l1 = fd.plot(label='samples', c=samples_color, linewidth=width)
    l2 = plt.plot(t, gaussian, label='gaussian', c=c_color, linestyle='dashed')
    l3 = unregmean.plot(label='mean', c=mean_color)
    plt.legend(handles=[l1[0], l3[0], l2[0]], loc=1)

    # Shift registered curves
    regbasis = fd.landmark_shift(shift, location, periodic=periodic)
    regmean = regbasis.mean()  # Registered mean

    # Plots the registered curves
    plt.figure()
    plt.title('Registered curves')
    plt.xlim(xlim)
    l1 = regbasis.plot(label='samples', c=samples_color, linewidth=width)
    l2 = plt.plot(t, gaussian, label='gaussian', c=c_color, linestyle='dashed')
    l3 = regmean.plot(label='mean', c=mean_color)
    plt.legend(handles=[l1[0], l3[0], l2[0]], loc=1)


    plt.show()
