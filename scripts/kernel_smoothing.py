""" Script to reproduce the example in the Kernel Smoothing section of the
end of degree project.

Uses different kernel smoothing methods over the phoneme data set and shows
how cross validations scores vary over a range of different parameters used in
the smoothing methods.

"""

import os
import numpy as np
import matplotlib.pylab as plt

import fda
import fda.validation as val
import fda.kernel_smoothers as ks


if __name__ == '__main__':

    # Loads the phoneme data set from a csv file and build a FDataGrid object.
    _dir = os.path.dirname(__file__)

    data = np.genfromtxt(os.path.join(_dir, '../data/phoneme_data.csv'),
                         delimiter=',',
                         skip_header=1)
    fd = fda.FDataGrid(data, list(range(data.shape[1])),
                       names=['Phoneme learn', 'frequencies',
                              'log-periodograms'])

    # Plots the first five samples of the data set.
    plt.figure(1)
    fd[0:5].plot()

    # Calculates general cross validation scores for different values of the
    #  parameters given to the different smoothing methods.

    # Local linear regression kernel smoothing.
    llr = val.minimise(fd, list(np.linspace(2, 25, 24)),
                       smoothing_method=ks.local_linear_regression)
    # Nadaraya-Watson kernel smoothing.
    nw = fda.validation.minimise(fd, list(np.linspace(2, 25, 24)),
                                 smoothing_method=ks.nw)
    # K-nearest neighbours kernel smoothing.
    knn = fda.validation.minimise(fd, list(np.linspace(2, 25, 24)),
                                  smoothing_method=ks.knn)

    # Plots the different scores for the different choices of parameters for
    #  the 3 methods.
    plt.figure(2)
    plt.plot(np.linspace(2, 25, 24), knn['scores'])
    plt.plot(np.linspace(2, 25, 24), llr['scores'])
    plt.plot(np.linspace(2, 25, 24), nw['scores'])

    ax = plt.gca()
    ax.set_xlabel('Smoothing method parameter')
    ax.set_ylabel('GCV score')
    ax.set_title('Scores through GCV for different smoothing methods')
    ax.legend(['k-nearest neighbours', 'local linear regression',
               'Nadaraya-Watson'],
              title='Smoothing method')

    # Plots the smoothed curves corresponding to the 11th element of the data
    #  set (this is a random choice) for the three different smoothing methods.
    plt.figure(3)
    fd[10].plot()
    knn['fdatagrid'][10].plot()
    llr['fdatagrid'][10].plot()
    nw['fdatagrid'][10].plot()
    ax = plt.gca()
    ax.legend(['original data', 'k-nearest neighbours',
               'local linear regression',
               'Nadaraya-Watson'],
              title='Smoothing method')

    # Plots the same 5 samples from the beginning using the Nadaraya-Watson
    # kernel smoother with the best choice of parameter.
    plt.figure(4)
    nw['fdatagrid'][0:5].plot()

    # Undersmoothing and oversmoothing
    fd_us = fda.FDataGrid(ks.nw(fd.sample_points, h=2) @ fd.data_matrix[10],
                          fd.sample_points, fd.sample_range, fd.names)
    fd_os = fda.FDataGrid(ks.nw(fd.sample_points, h=15) @ fd.data_matrix[10],
                          fd.sample_points, fd.sample_range, fd.names)

    # Not smoothed
    plt.figure(5)
    fd[10].plot()

    # Smoothed
    plt.figure(6)
    fd[10].scatter(s=0.5)
    nw['fdatagrid'][10].plot(c='g')

    # Under-smoothed
    plt.figure(7)
    fd[10].scatter(s=0.5)
    fd_us.plot(c='sandybrown')

    # Over-smoothed
    plt.figure(8)
    fd[10].scatter(s=0.5)
    fd_os.plot(c='r')
    plt.show()
