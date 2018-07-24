#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt

from fda.basis import FDataBasis, Fourier


nsamples = 15 # Number of samples
nfine = 100 # Number of points per sample
nbasis = 11 # Number of fourier basis elements
phase_sd = .6 # Standard deviation of phase variation
error_sd = .2 # Standard deviation of gaussian noise
period = 1 # Period of the sine used to generate the data
iterations = 3 # Number of iterations in the step-by-step figure


def noise_sin(t, period=1., phase_sd=1., error_sd=1.):

    phase_variation = np.outer(np.ones(t.shape[0]),
                         np.random.normal(0, phase_sd, t.shape[1]))

    error = np.random.normal(0, error_sd, t.shape)

    return np.sin(2*np.pi*t/period + phase_variation) + error


if __name__ == '__main__':

    # Fixing random state for reproducibility
    np.random.seed(19587801)

    # Matrix with times where each sample will be evaluated
    t = np.linspace(0,1,nfine)
    tsamples = np.outer(t,np.ones(nsamples))

    # Noisy sine data, with amplitude variation and gaussian error
    data = noise_sin(tsamples,period=period, phase_sd=phase_sd, error_sd=error_sd)

    # Plot the samples
    plt.figure(1)
    plt.title('Raw data')
    plt.plot(t, data,c='b', linewidth=0.8)
    plt.plot(t,np.sin(2*np.pi*t/period),c='r',linestyle='dashed') # Original curve

    # Curves smoothed with the matrix penalty method
    basis = FDataBasis.from_data(data.T, t, Fourier(nbasis=nbasis))
    plt.figure(2)
    plt.title('Unregistered curves')
    basis.plot(c='b', linewidth=0.8)
    #plt.plot(t,np.sin(2*np.pi*t/period),c='r',linestyle='dashed')

    # Mean of unregistered curves
    unregmean = basis.mean()
    plt.figure(3)
    plt.title('Unregistered mean')
    unregmean.plot(c='b')
    plt.plot(t,np.sin(2*np.pi*t/period),c='r',linestyle='dashed')

    # Shift registered curves
    registered_basis = basis.shift_registration()
    plt.figure(4)
    plt.title('Shift registered curves')
    registered_basis.plot(c='b', linewidth=0.8)
    #plt.plot(t,np.sin(2*np.pi*t/period),c='r',linestyle='dashed')

    # Registered mean
    regmean = registered_basis.mean()
    plt.figure(5)
    plt.title('Registered mean')
    regmean.plot(c='b')
    plt.plot(t,np.sin(2*np.pi*t/period),c='r',linestyle='dashed')


    f, axarr = plt.subplots(iterations+1, 2, sharex='col', sharey='row')

    basis.plot(ax=axarr[0][0], c='b', linewidth=0.8)
    axarr[0][1].plot(t,np.sin(2*np.pi*t/period),c='r',linestyle='dashed')
    basis.mean().plot(ax=axarr[0][1], c='b')


    axarr[0][0].set_title('Registered curves')
    axarr[0][1].set_title('Mean curves')
    axarr[0][0].set_ylabel('Unregistered')

    for i in range(1,iterations+1):
        partial_registered_basis = basis.shift_registration(maxiter=i)
        partial_registered_basis.plot(ax=axarr[i][0], c='b', linewidth=0.8)
        axarr[i][1].plot(t,np.sin(2*np.pi*t/period),c='r',linestyle='dashed')
        partial_registered_basis.mean().plot(ax=axarr[i][1], c='b')

        axarr[i][0].set_ylabel('%d iterations' % i)

    plt.show()
