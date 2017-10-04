"""This module defines a class for representing functional data as a series of mesures taken in a list of timestamps.

"""

import numpy
import matplotlib.pyplot


__author__ = "Miguel Carbajo Berrocal"
__license__ = "GPL3"
__version__ = ""
__maintainer__ = ""
__email__ = ""
__status__ = "Development"


class FDataGrid:
    """Class for representing functional data as a set of curves discretised in a list of points.

    Attributes:
        data_matrix (numpy.ndarray): an array containing the values of the function at the points of discretisation.
        argvals (list or list of lists): a list containing the points of discretisation where values have been recorded
            or a lists of list with each of the list containing the points of dicretisation for each axis.
        argvals_range (tuple or list): contains the edges of the interval in which the functional data is considered
            to exist.
        names (list): list containing the names of the data set, x label, y label, z label and so on.

    """
    def __init__(self, data_matrix, argvals=None, argvals_range=None, names=None):
        if isinstance(data_matrix, numpy.ndarray):
            self.data_matrix = data_matrix
        else:
            self.data_matrix = numpy.array(data_matrix)
        # TODO check dimensionality

        if argvals is None:
            if self.data_matrix.ndim > 2:
                self.argvals = [numpy.linspace(0, 1, self.data_matrix.shape[i]) for i in range(1,
                                                                                               self.data_matrix.ndim)]
            else:
                self.argvals = numpy.linspace(0, 1, self.data_matrix.shape[1])

        else:
            # Check that the dimension of the data matches the argvals list
            self.argvals = argvals
            if self.data_matrix.ndim == 1 \
                    or (self.data_matrix.ndim == 2 and len(self.argvals) != self.data_matrix.shape[1]) \
                    or (self.data_matrix.ndim > 2 and self.data_matrix.ndim != len(self.argvals) + 1):
                raise ValueError("Incorrect dimension in data_matrix and argvals arguments.")

        if argvals_range is None:
            if self.data_matrix.ndim == 2:
                self.argvals_range = (self.argvals[0], self.argvals[-1])
            else:
                self.argvals_range = [(self.argvals[i][0], self.argvals[i][-1]) for i in range(len(self.argvals))]
            # Default value for argvals_range is a list of tuples with the first and last element of each list of
                # the argvals
        else:
            self.argvals_range = argvals_range
            if len(self.argvals_range) != 2:
                raise ValueError("Incorrect value of argvals_range. It should have two elements.")
            if self.argvals_range[0] > self.argvals[0] or self.argvals_range[-1] < self.argvals[-1]:
                raise ValueError("Timestamps must be within the time range.")

        # TODO handle names
        self.names = ['Data set', 'xlabel', 'ylabel']

        return

    def round(self, decimals=0):
        return FDataGrid(self.data_matrix.round(decimals), self.argvals, self.argvals_range, self.names)

    def ndim(self):
        return self.data_matrix.ndim

    def nrow(self):
        return self.data_matrix.shape[0]

    def ncol(self):
        return self.data_matrix.shape[1]

    def shape(self):
        return self.data_matrix.shape

    def derivate(self, nderiv=1):
        # TODO make the loop more efficient
        if self.ndim() > 2:
            raise NotImplementedError("Not implemented for 2 or more dimensional data")
        if numpy.isnan(self.data_matrix).any():
            raise ValueError("The object cannot contain nan elements.")
        data_matrix = self.data_matrix
        argvals = self.argvals
        for i in range(nderiv):
            mdata = []
            for j in range(self.nrow()):
                arr = numpy.diff(data_matrix[j])/(argvals[1:] - argvals[:-1])
                arr = numpy.append(arr,arr[-1])
                arr[1:] += arr[:-1]
                arr[1:-1] /= 2
                mdata.append(arr)
            data_matrix = numpy.array(mdata)

        return FDataGrid(data_matrix, argvals, self.argvals_range)

    def __add__(self, other):
        if not isinstance(other, FDataGrid):
            raise TypeError("Object type is not FDataGrid.")
        if self.data_matrix.shape[1] != other.data_matrix.shape[1]:
            raise ValueError("Error in columns dimensions")
        if self.argvals != other.argvals:
            raise ValueError("Error in argvals")
        return FDataGrid(self.data_matrix + other.data_matrix, self.argvals, self.argvals_range, self.names)

    def __sub__(self, other):
        if not isinstance(other, FDataGrid):
            raise TypeError("Object type is not FDataGrid.")
        if self.data_matrix.shape[1] != other.data_matrix.shape[1]:
            raise ValueError("Error in columns dimensions")
        if self.argvals != other.argvals:
            raise ValueError("Error in argvals")
        return FDataGrid(self.data_matrix - other.data_matrix, self.argvals, self.argvals_range, self.names)

    def __mul__(self, other):
        if not isinstance(other, FDataGrid):
            raise TypeError("Object type is not FDataGrid.")
        if self.data_matrix.shape[1] != other.data_matrix.shape[1]:
            raise ValueError("Error in columns dimensions")
        if self.argvals != other.argvals:
            raise ValueError("Error in argvals")
        return FDataGrid(self.data_matrix * other.data_matrix, self.argvals, self.argvals_range, self.names)

    def __div__(self, other):
        if not isinstance(other, FDataGrid):
            raise TypeError("Object type is not FDataGrid.")
        if self.data_matrix.shape[1] != other.data_matrix.shape[1]:
            raise ValueError("Error in columns dimensions")
        if self.argvals != other.argvals:
            raise ValueError("Error in argvals")
        return FDataGrid(self.data_matrix / other.data_matrix, self.argvals, self.argvals_range, self.names)

    def plot(self, *args, **kwargs):
        matplotlib.pyplot.plot(self.argvals, numpy.transpose(self.data_matrix), *args, **kwargs)

    def __str__(self):
        return 'Data set:\t' + str(self.data_matrix) \
                 + '\nargvals:\t' + str(self.argvals) \
                 + '\ntime range:\t' + str(self.argvals_range)
