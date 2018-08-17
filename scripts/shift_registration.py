#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt

from fda.basis import FDataBasis, Fourier
from fda.grid import FDataGrid

# Data parameters
nsamples = 15  # Number of samples
nfine = 100  # Number of points per sample
phase_sd = .6  # Standard deviation of phase variation
amp_sd = .05  # Standard deviation of amplitude variation
error_sd = .2  # Standard deviation of gaussian noise
period = 1  # Period of the sine used to generate the data
nbasis = 11  # Number of fourier basis elements

# Plot options
width = .8  # Width of the samples curves
samples_color = 'teal'
mean_color = 'black'
curve_color = 'maroon'
ylim = (-1.75, 1.75)
xlim = (0, 1)
iterations = 5  # Number of iterations in the step-by-step figure


def noise_sin(t, nsamples, period=1, phase_sd=0, amp_sd=0, error_sd=0):
    """Sine noisy function

    Generates several samples of the function

    ..math::
        x_i(t) = a_i \sin(\frac{2 \pi t}{period} + \delta_i) + \epsilon(t)

    evaluated at the samples points. :math: `a_i, \delta_{i}` and :math:
    `\epsilon(t_{ij})` are normaly distributed.

    Args:
        t (ndarray): Array of times
        nsamples (float): Number of samples
        period (float): Period of the sine function
        phase_sd (float): Standard deviation of the phase variation
        amp_sd (float): Standard deviation of the amplitude variation
        error_sd (float): Standard deviation of the error of the samples

    Returns:
        ndarray: Matrix with the samples evaluated. Each row is a sample and
        each column is a discrete time.
    """

    phase_variation = np.outer(np.random.normal(0, phase_sd,  nsamples),
                               np.ones(len(t)))

    error = np.random.normal(0, error_sd, (nsamples, len(t)))

    amp = np.diag(np.random.normal(1, amp_sd, nsamples))

    return amp @ np.sin(2*np.pi*t/period + phase_variation) + error


if __name__ == '__main__':

    # Seaborn style
    plt.style.use('seaborn')

    # Fixing random state for reproducibility
    np.random.seed(98765)

    # Matrix with times where each sample will be evaluated
    t = np.linspace(xlim[0], xlim[1], nfine)

    # Noisy sine data, with amplitude variation and gaussian error
    data = noise_sin(t, nsamples, period, phase_sd, amp_sd, error_sd)
    fdgrid = FDataGrid(data, t, xlim, 'Raw Data')
    # Real sine function
    sine = np.sin(2*np.pi*t/period)

    # Plot the samples
    plt.figure()
    plt.ylim(ylim)
    plt.xlim(xlim)
    l1 = fdgrid.plot(label='samples', c=samples_color, linewidth=width)
    l2 = plt.plot(t, sine, label='sine', c=curve_color, linestyle='dashed')
    l3 = fdgrid.mean().plot(label='mean', c=mean_color)
    plt.legend(handles=[l1[0], l3[0], l2[0]], loc=1)

    # Curves smoothed with the matrix penalty method
    # Curves smoothed with the matrix penalty method
    fd = fdgrid.to_basis(Fourier(xlim, nbasis, period))
    unregmean = fd.mean()  # Mean of unregistered curves

    # Plots the smoothed curves
    plt.figure()
    plt.title('Unregistered curves')
    plt.ylim(ylim)
    plt.xlim(xlim)
    l1 = fd.plot(label='samples', c=samples_color, linewidth=width)
    l2 = plt.plot(t, sine, label='sine', c=curve_color, linestyle='dashed')
    l3 = unregmean.plot(label='mean', c=mean_color)
    plt.legend(handles=[l1[0], l3[0], l2[0]], loc=1)

    # Shift registered curves
    regbasis = fd.shift_registration()
    regmean = regbasis.mean()  # Registered mean

    # Plots the registered curves
    plt.figure()
    plt.title('Registered curves')
    plt.ylim(ylim)
    plt.xlim(xlim)
    l1 = regbasis.plot(label='samples', c=samples_color,
                       linewidth=width)
    l2 = plt.plot(t, sine, label='sine', c=curve_color, linestyle='dashed')
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
