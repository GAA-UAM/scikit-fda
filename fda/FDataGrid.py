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
        sample_points (numpy.ndarray): an array containing the points of
            discretisation where values have been recorded or a list of lists
            with each of the list containing the points of dicretisation for
            each axis.
        sample_range (tuple or list): contains the edges of the interval
            in which the functional data is considered to exist.
        names (list): list containing the names of the data set, x label, y
            label, z label and so on.

    Examples:
        The number of columns of data_matrix have to be the length of 
        sample_points.

        >>> FDataGrid(numpy.array([1,2,4,5,8]), range(6))
        Traceback (most recent call last):
            ....
        ValueError: Incorrect dimension in data_matrix and sample_points.

    """
    def __init__(self, data_matrix, sample_points=None, 
                 sample_range=None, names=None):
        """
        Args:
            data_matrix (array_like): a matrix where each row contains the
                values of a functional datum evaluated at the
                points of discretisation.
            sample_points (array_like, optional): an array containing the 
            points of discretisation where values have been recorded or a list 
            of lists with each of the list containing the points of
                dicretisation for each axis.
            sample_range (tuple or list, optional): contains the edges 
                of the interval in which the functional data is considered to
                exist.
            names (list): list containing the names of the data set, x label, y
                label, z label and so on.

        """
        self.data_matrix = numpy.atleast_2d(data_matrix)
        # TODO check dimensionality

        if sample_points is None:
            if self.data_matrix.ndim > 2:
                self.sample_points = [numpy.linspace(0, 1,
                                      self.data_matrix.shape[i]) for i
                                      in range(1, self.data_matrix.ndim)]
            else:
                self.sample_points = numpy.linspace(0, 1, 
                                                    self.data_matrix.shape[1])

        else:
            # Check that the dimension of the data matches the sample_points 
            # list
            self.sample_points = numpy.asarray(sample_points)
            if ((self.data_matrix.ndim == 2
                    and len(self.sample_points) != self.data_matrix.shape[1])
                or (self.data_matrix.ndim > 2
                    and self.data_matrix.ndim != len(self.sample_points) + 1)):
                raise ValueError("Incorrect dimension in data_matrix and "
                                 "sample_points.")

        if sample_range is None:
            if self.data_matrix.ndim == 2:
                self.sample_range = (self.sample_points[0], 
                                     self.sample_points[-1])
            else:
                self.sample_range = [(self.sample_points[i][0],
                                     self.sample_points[i][-1])
                                     for i in range(len(self.sample_points))]
            # Default value for sample_range is a list of tuples with
            # the first and last element of each list ofthe sample_points.
        else:
            self.sample_range = sample_range
            if len(self.sample_range) != 2:
                raise ValueError("Incorrect value of sample_range. It "
                                 "should have two elements.")
            if (self.sample_range[0] > self.sample_points[0]
                    or self.sample_range[-1] < self.sample_points[-1]):
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
            in its data_matrix and sample_points are rounded .The real and
            imaginary parts of complex numbers are rounded separately.

        """
        return FDataGrid(self.data_matrix.round(decimals),
                         self.sample_points.round(decimals),
                         self.sample_range, self.names)

    @property
    def ndim_domain(self):
        """ Number of dimensions of the domain.

        Returns:
            int: Number of dimensions of the domain.
        """
        return self.sample_points.ndim

    @property
    def ndim_image(self):
        """ Number of dimensions of the domain.

        Returns:
            int: Number of dimensions of the domain.
        """
        return self.data_matrix.ndim[(0,) * (1 + self.ndim_domain)]

    @property
    def ndim(self):
        """ Number of dimensions of the data matrix.

        Returns:
            int: Number of dimensions of the data matrix.
        """
        return self.data_matrix.ndim

    @property
    def n_samples(self):
        """ Number of rows of the data_matrix. Also the number of samples.

        Returns:
            int: Number of samples of the FDataGrid object. Also the number of 
                rows of the data_matrix.

        """
        return self.data_matrix.shape[0]

    @property
    def ncol(self):
        """ Number of columns of the data_matrix. Also the number of points
        of discretisation.

        Returns:
            int: Number of columns of the data_matrix.

        """
        return self.data_matrix.shape[1]

    @property
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
        vector contaning the points of discretisation; :math:`D^1` is
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
                ,sample_points=array([0, 1, 2, 3, 4])
                ,sample_range=(0, 4)
                ,names=['Data set - 1 derivative', 'xlabel', 'ylabel'])

            Second order derivative

            >>> fdata = FDataGrid([1,2,4,5,8], range(5))
            >>> fdata.derivative(2)
            FDataGrid(
                array([[ 0.5 ,  0.25,  0.25,  0.75,  1.  ]])
                ,sample_points=array([0, 1, 2, 3, 4])
                ,sample_range=(0, 4)
                ,names=['Data set - 2 derivative', 'xlabel', 'ylabel'])

        """
        if order < 1:
            raise ValueError("The order of a derivative has to be greater "
                             "or equal than 1.")
        if self.ndim > 2:
            raise NotImplementedError("Not implemented for 2 or more"
                                      " dimensional data.")
        if numpy.isnan(self.data_matrix).any():
            raise ValueError("The FDataGrid object cannot contain nan "
                             "elements.")
        data_matrix = self.data_matrix
        sample_points = self.sample_points
        for _ in range(order):
            mdata = []
            for i in range(self.n_samples):
                arr = numpy.diff(data_matrix[i])/(sample_points[1:] 
                                                  - sample_points[:-1])
                arr = numpy.append(arr, arr[-1])
                arr[1:-1] += arr[:-2]
                arr[1:-1] /= 2
                mdata.append(arr)
            data_matrix = numpy.array(mdata)

        names = [self.names[0] + ' - {} derivative'.format(order)]
        names += self.names[1:]

        return FDataGrid(data_matrix, sample_points, self.sample_range,
                         names)

    def __add__(self, other):
        if not isinstance(other, FDataGrid):
            raise NotImplemented
        if self.data_matrix.shape[1] != other.data_matrix.shape[1]:
            raise ValueError("Error in columns dimensions")
        if not numpy.array_equal(self.sample_points,
                                 other.sample_points):
            raise ValueError(
                "Sample points for both objects must be equal")
        return FDataGrid(self.data_matrix + other.data_matrix,
                         self.sample_points, self.sample_range,
                         self.names)

    def __sub__(self, other):
        if not isinstance(other, FDataGrid):
            raise NotImplemented
        if self.data_matrix.shape[1] != other.data_matrix.shape[1]:
            raise ValueError("Error in columns dimensions")
            # Checks
        if not numpy.array_equal(self.sample_points,
                                 other.sample_points):
            raise ValueError(
                "Sample points for both objects must be equal")
        return FDataGrid(self.data_matrix - other.data_matrix,
                         self.sample_points, self.sample_range, 
                         self.names)

    def __mul__(self, other):
        if not isinstance(other, FDataGrid):
            raise NotImplemented
        if self.data_matrix.shape[1] != other.data_matrix.shape[1]:
            raise ValueError("Error in columns dimensions")
        if not numpy.array_equal(self.sample_points,
                                 other.sample_points):
            raise ValueError(
                "Sample points for both objects must be equal")
        return FDataGrid(self.data_matrix * other.data_matrix, 
                         self.sample_points, self.sample_range, 
                         self.names)

    def __truediv__(self, other):
        if not isinstance(other, FDataGrid):
            raise NotImplemented
        if self.data_matrix.shape[1] != other.data_matrix.shape[1]:
            raise ValueError("Error in columns dimensions")
        if not numpy.array_equal(self.sample_points,
                                 other.sample_points):
            raise ValueError(
                "Sample points for both objects must be equal")
        return FDataGrid(self.data_matrix / other.data_matrix,
                         self.sample_points, self.sample_range, 
                         self.names)

    def plot(self, *args, **kwargs):
        _plot = matplotlib.pyplot.plot(self.sample_points,
                                       numpy.transpose(self.data_matrix),
                                       *args, **kwargs)
        ax = matplotlib.pyplot.gca()
        ax.set_title(self.names[0])
        ax.set_xlabel(self.names[1])
        ax.set_ylabel(self.names[2])
        return _plot

    def __str__(self):
        """ Return str(self). """
        return ('Data set:\t' + str(self.data_matrix)
                + '\nsample_points:\t' + str(self.sample_points)
                + '\ntime range:\t' + str(self.sample_range))

    def __repr__(self):
        """ Return repr(self). """
        return ("FDataGrid(\n    "
                + self.data_matrix.__repr__()
                + "\n    ,sample_points=" + self.sample_points.__repr__()
                + "\n    ,sample_range=" + self.sample_range.__repr__()
                + "\n    ,names=" + self.names.__repr__()
                + ")")

    def __getitem__(self, key):
        """ Return self[key]. """
        if isinstance(key, tuple) and len(key) > 1:
            return FDataGrid(self.data_matrix[key],
                             self.sample_points[key[1:1+self.ndim_domain]],
                             self.sample_range, self.names)
        return FDataGrid(self.data_matrix[key], self.sample_points,
                         self.sample_range, self.names)
