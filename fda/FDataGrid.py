"""This module defines a class for representing functional data as a series of
lists of values, each representing the observation of a function measured in a
list of discretisation points.

"""

import numpy
import matplotlib.pyplot


__author__ = "Miguel Carbajo Berrocal"
__email__ = "miguel.carbajo@estudiante.uam.es"


class FDataGrid:
    """ Class for representing functional data as a set of curves discretised
    in a grid of points.

    Attributes:
        data_matrix (numpy.ndarray): a matrix where each row contains the
            values of a functional datum evaluated at the
            points of discretisation.
        argvals (numpy.ndarray): an array containing the points of
            discretisation where values have been recorded or a list of lists
            with each of the list containing the points of dicretisation for
            each axis.
        argvals_range (tuple or list): contains the edges of the interval in
            which the functional data is considered to exist.
        names (list): list containing the names of the data set, x label, y
            label, z label and so on.

    Examples:
        The number of columns of data_matrix have to be the length of argvals.
        >>> FDataGrid(numpy.array([1,2,4,5,8]), range(6))
        Traceback (most recent call last):
            ....
        ValueError: Incorrect dimension in data_matrix and argvals arguments.


    """
    def __init__(self, data_matrix, argvals=None, argvals_range=None,
                 names=None):
        """
        Args:
            data_matrix (array_like): a matrix where each row contains the
                values of a functional datum evaluated at the
                points of discretisation.
            argvals (array_like, optional): an array containing the points of
                discretisation where values have been recorded or a list of
                    lists with each of the list containing the points of
                    dicretisation for each axis.
            argvals_range (tuple or list, optional): contains the edges of
                the interval in which the functional data is considered to
                exist.
            names (list): list containing the names of the data set, x label, y
                label, z label and so on.

        """
        self.data_matrix = numpy.asarray(data_matrix)
        if self.data_matrix.ndim == 1:
            self.data_matrix = numpy.array([self.data_matrix])
        # TODO check dimensionality

        if argvals is None:
            if self.data_matrix.ndim > 2:
                self.argvals = [numpy.linspace(0, 1, self.data_matrix.shape[i])
                                for i in range(1, self.data_matrix.ndim)]
            else:
                self.argvals = numpy.linspace(0, 1, self.data_matrix.shape[1])

        else:
            # Check that the dimension of the data matches the argvals list
            self.argvals = numpy.asarray(argvals)
            if self.data_matrix.ndim == 1 \
                    or (self.data_matrix.ndim == 2
                        and len(self.argvals) != self.data_matrix.shape[1]) \
                    or (self.data_matrix.ndim > 2
                        and self.data_matrix.ndim != len(self.argvals) + 1):
                raise ValueError("Incorrect dimension in data_matrix and "
                                 "argvals arguments.")

        if argvals_range is None:
            if self.data_matrix.ndim == 2:
                self.argvals_range = (self.argvals[0], self.argvals[-1])
            else:
                self.argvals_range = [(self.argvals[i][0], self.argvals[i][-1])
                                      for i in range(len(self.argvals))]
            # Default value for argvals_range is a list of tuples with the
                # first and last element of each list ofthe argvals
        else:
            self.argvals_range = argvals_range
            if len(self.argvals_range) != 2:
                raise ValueError("Incorrect value of argvals_range. It should"
                                 " have two elements.")
            if self.argvals_range[0] > self.argvals[0] \
                    or self.argvals_range[-1] < self.argvals[-1]:
                raise ValueError("Timestamps must be within the time range.")

        self.names = names
        if self.names is None:
            self.names = ['Data set', 'xlabel', 'ylabel']

        return

    def round(self, decimals=0):
        """ Evenly round to the given number of decimals.

        Args:
            decimals (int, optional): Number of decimal places to round to.
                If decimals is negative, it specifies the number of
                positions to the left of the decimal point. Defaults to 0.

        Returns:
            :obj:FDataGrid: Returns a FDataGrid object where all elements
            in its data_matrix and argvals are rounded .The real and
            imaginary parts of complex numbers are rounded separately.

        """
        return FDataGrid(self.data_matrix.round(decimals),
                         self.argvals.round(decimals),
                         self.argvals_range, self.names)

    def ndim(self):
        """ Number of dimensions of the data

        """
        return self.data_matrix.ndim

    def nrow(self):
        """ Number of rows of the data_matrix. Also the number of samples.

        Returns:
            int: Number of rows of the data_matrix.

        """
        return self.data_matrix.shape[0]

    def ncol(self):
        """ Number of columns of the data_matrix. Also the number of points
        of discretisation.

        Returns:
            int: Number of columns of the data_matrix.

        """
        return self.data_matrix.shape[1]

    def shape(self):
        """ Dimensions (aka shape) of the data_matrix.

        Returns:
            list of int: List containing the length of the matrix on each of
            its axis. If the matrix is 2 dimensional shape returns [number of
            rows, number of columns].

        """
        return self.data_matrix.shape

    def derivative(self, order=1):
        """ Derivative of a FDataGrid object.

        Its calculated using lagged differences. If we call :math:`D` the
        data_matrix, :math:`D^1` the derivative of order 1 and :math:`T` the
        vector
        contaning the points of discretisation; :math:`D^1` is
        calculated as it follows:

        .. math::

            D^{1}_{ij} = \\begin{cases}
            \\frac{D_{i1} - D_{i2}}{ T_{1} - T_{2}}  & \\mbox{if } j = 1 \\\\
            \\frac{D_{i(m-1)} - D_{im}}{ T_{m-1} - T_m}  & \\mbox{if }
                j = m \\\\
            \\frac{D_{i(j-1)} - D_{i(j+1)}}{ T_{j-1} - T_{j+1}} & \\mbox{if }
            1 < j < m
            \\end{cases}

        Where m is the number of columns of the matrix :math:`D`.

        Order > 1 derivatives are calculated by using derivative recursively.

        Args:
            order (int, optional): Order of the derivative. Defaults to one.

        Examples:
            First order derivative
            >>> fdata = FDataGrid([1,2,4,5,8], range(5))
            >>> fdata.derivative()
            FDataGrid(
                array([[ 1. ,  1.5,  1.5,  2. ,  3. ]])
                ,argvals=array([0, 1, 2, 3, 4])
                ,argvals_range=(0, 4)
                ,names=['Data set', 'xlabel', 'ylabel'])

            Second order derivative
            >>> fdata = FDataGrid([1,2,4,5,8], range(5))
            >>> fdata.derivative(2)
            FDataGrid(
                array([[ 0.5 ,  0.25,  0.25,  0.75,  1.  ]])
                ,argvals=array([0, 1, 2, 3, 4])
                ,argvals_range=(0, 4)
                ,names=['Data set', 'xlabel', 'ylabel'])

        """
        if order < 1:
            raise ValueError("The order of a derivative has to be greater "
                             "or equal than 1.")
        if self.ndim() > 2:
            raise NotImplementedError("Not implemented for 2 or more"
                                      " dimensional data.")
        if numpy.isnan(self.data_matrix).any():
            raise ValueError("The FDataGrid object cannot contain nan "
                             "elements.")
        data_matrix = self.data_matrix
        argvals = self.argvals
        for k in range(order):
            mdata = []
            for i in range(self.nrow()):
                arr = numpy.diff(data_matrix[i])/(argvals[1:] - argvals[:-1])
                arr = numpy.append(arr, arr[-1])
                arr[1:-1] += arr[:-2]
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
        return FDataGrid(self.data_matrix + other.data_matrix, self.argvals,
                         self.argvals_range, self.names)

    def __sub__(self, other):
        if not isinstance(other, FDataGrid):
            raise TypeError("Object type is not FDataGrid.")
        if self.data_matrix.shape[1] != other.data_matrix.shape[1]:
            raise ValueError("Error in columns dimensions")
        if self.argvals != other.argvals:
            raise ValueError("Error in argvals")
        return FDataGrid(self.data_matrix - other.data_matrix, self.argvals,
                         self.argvals_range, self.names)

    def __mul__(self, other):
        if not isinstance(other, FDataGrid):
            raise TypeError("Object type is not FDataGrid.")
        if self.data_matrix.shape[1] != other.data_matrix.shape[1]:
            raise ValueError("Error in columns dimensions")
        if self.argvals != other.argvals:
            raise ValueError("Error in argvals")
        return FDataGrid(self.data_matrix * other.data_matrix, self.argvals,
                         self.argvals_range, self.names)

    def __truediv__(self, other):
        if not isinstance(other, FDataGrid):
            raise TypeError("Object type is not FDataGrid.")
        if self.data_matrix.shape[1] != other.data_matrix.shape[1]:
            raise ValueError("Error in columns dimensions")
        if self.argvals != other.argvals:
            raise ValueError("Error in argvals")
        return FDataGrid(self.data_matrix / other.data_matrix, self.argvals,
                         self.argvals_range, self.names)

    def plot(self, *args, **kwargs):
        # TODO handle names
        matplotlib.pyplot.plot(self.argvals, numpy.transpose(self.data_matrix),
                               *args, **kwargs)

    def __str__(self):
        return 'Data set:\t' + str(self.data_matrix) \
                 + '\nargvals:\t' + str(self.argvals) \
                 + '\ntime range:\t' + str(self.argvals_range)

    def __repr__(self):
        return "FDataGrid(\n    " \
               + self.data_matrix.__repr__() \
               + "\n    ,argvals=" + self.argvals.__repr__() \
               + "\n    ,argvals_range=" + self.argvals_range.__repr__() \
               + "\n    ,names=" + self.names.__repr__() \
               + ")"
