"""
Exploring data
==============

Script to reproduce the example in the Exploring Data section of the
end of degree project.

Explores the Tecator data set by plotting the functional data and calculating
means and derivatives.
"""
import os

from fda import FDataGrid
import fda
import matplotlib.pylab as plt
import numpy as np

###############################################################################
# Tecator data is structured in 3 differents csv files. One containing
# the data at the sample points. Other containing the sample points. And
# a last one containing information about the percentage of fat,
# water and protein in each sample.

_dir = os.getcwd()

# Loads all 3 csv files.
data = np.genfromtxt(os.path.join(_dir, '../data/tecator_data.csv'),
                     delimiter=',',
                     skip_header=1)

sample_points = np.genfromtxt(
    os.path.join(_dir, '../data/tecator_sample_points.csv'),
    delimiter=',',
    skip_header=1)

y = np.genfromtxt(os.path.join(_dir, '../data/tecator_y.csv'),
                  delimiter=',',
                  skip_header=1)

# Builds a FDataGrid object using the loaded information.
fd = FDataGrid(data, sample_points,
               dataset_label='Spectrometric curves',
               axes_labels=['Wavelength (mm)', 'Absorbances'])

###############################################################################
# Plots in red samples containing less than 20% of fat and in blue the
# rest.

plt.figure()
fd[y[:, 0] < 20].plot(c='r', linewidth=0.5)
fd[np.logical_not(y[:, 0] < 20)].plot(c='b', linewidth=0.5, alpha=0.7)

###############################################################################
# Plots the mean of each group.

plt.figure()
fda.mean(fd[y[:, 0] < 20]).plot(c='r', linewidth=0.5)
fda.mean(fd[np.logical_not(y[:, 0] < 20)]).plot(c='b', linewidth=0.5,
                                                alpha=0.7)
fd.dataset_label = fd.dataset_label + ' - means'

###############################################################################
# Plots the derivative of all samples and uses the same color code as
# above.

plt.figure()
fdd = fd.derivative(1)
fdd[y[:, 0] < 20].plot(c='r', linewidth=0.5)
fdd[np.logical_not(y[:, 0] < 20)].plot(c='b', linewidth=0.5, alpha=0.7)
plt.show()
