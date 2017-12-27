import os
import numpy as np
import matplotlib.pylab as plt

import fda
from fda import FDataGrid


if __name__ == '__main__':

    # Tecator data is structured in 3 differents csv files. One containing
    # the data at the sample points. Other containing the sample points. And
    #  a last one containing information about the percentage of fat,
    # water and protein in each sample.
    _dir = os.path.dirname(__file__)

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
                   names=['Spectrometric curves', 'Wavelength (mm)',
                          'Absorbances'])

    fd = fd[:5]
    # Plots the first 5 samples in a scatter plot.
    plt.figure()
    fd.scatter(s=0.5)
    plt.show()

    # Plots the first 5 samples in a line plot.
    plt.figure()
    fd.plot()
    plt.show()