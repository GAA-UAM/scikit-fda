"""Discretised functional data module.

This module defines a class for representing functional data as a series of
lists of values, each representing the observation of a function measured in a
list of discretisation points.
"""

import numbers

import matplotlib.pyplot
import numpy
import scipy
import scipy.stats.mstats

from . import basis as fdbasis


__author__ = "Miguel Carbajo Berrocal"
__email__ = "miguel.carbajo@estudiante.uam.es"


def _list_of_arrays(original_array):
    """Convert to a list of arrays.

    If the original list is one-dimensional (e.g. [1, 2, 3]), return list to
    array (in this case [array([1, 2, 3])]).

    If the original list is two-dimensional (e.g. [[1, 2, 3], [4, 5]]), return
    a list containing other one-dimensional arrays (in this case
    [array([1, 2, 3]), array([4, 5, 6])]).

    In any other case the behaviour is unespecified.

    """
    new_array = numpy.array([numpy.asarray(i) for i in
                             numpy.atleast_1d(original_array)])

    # Special case: Only one array, expand dimension
    if len(new_array.shape) == 1 and not any(isinstance(s, numpy.ndarray)
                                             for s in new_array):
        new_array = numpy.atleast_2d(new_array)

    return list(new_array)


class FDataGrid:
    """Represents discretised functional data.

    Class for representing functional data as a set of curves discretised
    in a grid of points.

    Attributes:
        data_matrix (numpy.ndarray): a matrix where each entry of the first
        axis contains the values of a functional datum evaluated at the
        points of discretisation.
        sample_points (numpy.ndarray): 2 dimension matrix where each row
            contains the points of dicretisation for each axis of data_matrix.
        sample_range (numpy.ndarray): 2 dimension matrix where each row
            contains the bounds of the interval in which the functional data
            is considered to exist for each one of the axies.
        dataset_label (str): name of the dataset.
        axes_labels (list): list containing the labels of the different
            axis. The first element is the x label, the second the y label
            and so on.

    Examples:
        Representation of a functional data object with 2 samples
        representing a function :math:`f : \mathbb{R}\longmapsto\mathbb{R}`.

        >>> data_matrix = [[1, 2], [2, 3]]
        >>> sample_points = [2, 4]
        >>> FDataGrid(data_matrix, sample_points)
        FDataGrid(
            array([[1, 2],
                   [2, 3]]),
            sample_points=[array([2, 4])],
            ...)

        The number of columns of data_matrix have to be the length of
        sample_points.

        >>> FDataGrid(numpy.array([1,2,4,5,8]), range(6))
        Traceback (most recent call last):
            ....
        ValueError: Incorrect dimension in data_matrix and sample_points...



        FDataGrid support higher dimensional data both in the domain and image.
        Representation of a functional data object with 2 samples
        representing a function :math:`f : \mathbb{R}\longmapsto\mathbb{R}^2`.

        >>> data_matrix = [[[1, 0.3], [2, 0.4]], [[2, 0.5], [3, 0.6]]]
        >>> sample_points = [2, 4]
        >>> fd = FDataGrid(data_matrix, sample_points)
        >>> fd.ndim_domain, fd.ndim_image
        (1, 2)

        Representation of a functional data object with 2 samples
        representing a function :math:`f : \mathbb{R}^2\longmapsto\mathbb{R}`.
        >>> data_matrix = [[[1, 0.3], [2, 0.4]], [[2, 0.5], [3, 0.6]]]
        >>> sample_points = [[2, 4], [3,6]]
        >>> fd = FDataGrid(data_matrix, sample_points)
        >>> fd.ndim_domain, fd.ndim_image
        (2, 1)

    """
    def __init__(self, data_matrix, sample_points=None,
                 sample_range=None, dataset_label='Data set',
                 axes_labels=None):
        """
        Args:
            data_matrix (array_like): a matrix where each row contains the
                values of a functional datum evaluated at the
                points of discretisation.
            sample_points (array_like, optional): an array containing the
            points of discretisation where values have been recorded or a list
            of lists with each of the list containing the points of
                dicretisation for each axis.
            sample_range (tuple or list of tuples, optional): contains the
                edges of the interval in which the functional data is
                considered to exist (if the argument has 2 dimensions each
                row is interpreted as the limits of one of the dimension of
                the domain.
            dataset_label (str, optional): name of the dataset.
            axes_labels (list, optional): list containing the labels of the
                different axes. The first element is the x label, the second
                the y label and so on.

        """
        self.data_matrix = numpy.atleast_2d(data_matrix)

        if sample_points is None:
            self.sample_points = _list_of_arrays(
                [numpy.linspace(0, 1, self.data_matrix.shape[i]) for i in
                 range(1, self.data_matrix.ndim)])

        else:
            # Check that the dimension of the data matches the sample_points
            # list

            self.sample_points = _list_of_arrays(sample_points)

            data_shape = self.data_matrix.shape[1: 1 + self.ndim_domain]
            sample_points_shape = [len(i) for i in self.sample_points]

            if not numpy.array_equal(
                    data_shape,
                    sample_points_shape):
                raise ValueError("Incorrect dimension in data_matrix and "
                                 "sample_points. Data has shape {} and sample "
                                 "points have shape {}"
                                 .format(data_shape, sample_points_shape))

        if sample_range is None:
                self.sample_range = numpy.array(
                    [(self.sample_points[i][0], self.sample_points[i][-1])
                     for i in range(self.ndim_domain)])
            # Default value for sample_range is a list of tuples with
            # the first and last element of each list ofthe sample_points.
        else:
            self.sample_range = numpy.atleast_2d(sample_range)
            # sample range must by a 2 dimension matrix with as many rows as
            # dimensions in the domain and 2 columns
            if (self.sample_range.ndim != 2 or self.sample_range.shape[1] != 2
                    or self.sample_range.shape[0] != self.ndim_domain):
                raise ValueError("Incorrect shape of sample_range.")
            for i in range(self.ndim_domain):
                if (self.sample_range[i, 0] > self.sample_points[i][0]
                        or self.sample_range[i, -1] < self.sample_points[i]
                        [-1]):
                    raise ValueError("Sample points must be within the sample "
                                     "range.")

        self.dataset_label = dataset_label
        self.axes_labels = axes_labels

        return

    def round(self, decimals=0):
        """ Evenly round to the given number of decimals.

        Args:
            decimals (int, optional): Number of decimal places to round to.
                If decimals is negative, it specifies the number of
                positions to the left of the decimal point. Defaults to 0.

        Returns:
            :obj:FDataGrid: Returns a FDataGrid object where all elements
            in its data_matrix are rounded .The real and
            imaginary parts of complex numbers are rounded separately.

        """
        return FDataGrid(self.data_matrix.round(decimals),
                         self.sample_points,
                         self.sample_range, self.dataset_label,
                         self.axes_labels)

    @property
    def ndim_domain(self):
        """ Number of dimensions of the domain.

        Returns:
            int: Number of dimensions of the domain.
        """
        return len(self.sample_points)

    @property
    def ndim_image(self):
        """ Number of dimensions of the image

        Returns:
            int: Number of dimensions of the image.
        """
        try:
            # The dimension of the image is the length of the array that can
            #  be extracted from the data_matrix using all the dimensions of
            #  the domain.
            return self.data_matrix.shape[1 + self.ndim_domain]
        # If there is no array that means the dimension of the image is 1.
        except IndexError:
            return 1

    @property
    def ndim(self):
        """ Number of dimensions of the data matrix.

        Returns:
            int: Number of dimensions of the data matrix.
        """
        return self.data_matrix.ndim

    @property
    def nsamples(self):
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
                array([[1. , 1.5, 1.5, 2. , 3. ]]),
                sample_points=[array([0, 1, 2, 3, 4])],
                sample_range=array([[0, 4]]),
                dataset_label='Data set - 1 derivative',
                ...)

            Second order derivative

            >>> fdata = FDataGrid([1,2,4,5,8], range(5))
            >>> fdata.derivative(2)
            FDataGrid(
                array([[0.5 , 0.25, 0.25, 0.75, 1.  ]]),
                sample_points=[array([0, 1, 2, 3, 4])],
                sample_range=array([[0, 4]]),
                dataset_label='Data set - 2 derivative',
                ...)

        """
        if self.ndim_domain != 1:
            raise NotImplementedError(
                "This method only works when the dimension "
                "of the domain of the FDatagrid object is "
                "one.")
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
        sample_points = self.sample_points[0]
        for _ in range(order):
            mdata = []
            for i in range(self.nsamples):
                arr = numpy.diff(data_matrix[i]) / (sample_points[1:]
                                                    - sample_points[:-1])
                arr = numpy.append(arr, arr[-1])
                arr[1:-1] += arr[:-2]
                arr[1:-1] /= 2
                mdata.append(arr)
            data_matrix = numpy.array(mdata)

        dataset_label = "{} - {} derivative".format(self.dataset_label, order)

        return FDataGrid(data_matrix, sample_points, self.sample_range,
                         dataset_label, self.axes_labels)

    def __check_same_dimensions(self, other):
        if self.data_matrix.shape[1] != other.data_matrix.shape[1]:
            raise ValueError("Error in columns dimensions")
        if not numpy.array_equal(self.sample_points,
                                 other.sample_points):
            raise ValueError(
                "Sample points for both objects must be equal")

    def mean(self):
        return FDataGrid(self.data_matrix.mean(axis=0, keepdims=True),
                         self.sample_points, self.sample_range,
                         self.dataset_label, self.axes_labels)

    def gmean(self):
        """ Computes the geometric mean of all samples in the FDataGrid object.

            Returns:
                FDataGrid: A FDataGrid object with just one sample representing
                the geometric mean of all the samples in the original
                FDataGrid object.

            """
        return FDataGrid([scipy.stats.mstats.gmean(self.data_matrix, 0)],
                         self.sample_points, self.sample_range,
                         self.dataset_label, self.axes_labels)

    def __add__(self, other):
        if isinstance(other, (numpy.ndarray, numbers.Number)):
            data_matrix = other
        elif isinstance(other, FDataGrid):
            self.__check_same_dimensions(other)
            data_matrix = other.data_matrix
        else:
            return NotImplemented

        return FDataGrid(self.data_matrix + data_matrix,
                         self.sample_points, self.sample_range,
                         self.dataset_label, self.axes_labels)

    def __sub__(self, other):
        if isinstance(other, (numpy.ndarray, numbers.Number)):
            data_matrix = other
        elif isinstance(other, FDataGrid):
            self.__check_same_dimensions(other)
            data_matrix = other.data_matrix
        else:
            return NotImplemented

        return FDataGrid(self.data_matrix - data_matrix,
                         self.sample_points, self.sample_range,
                         self.dataset_label, self.axes_labels)

    def __mul__(self, other):
        if isinstance(other, (numpy.ndarray, numbers.Number)):
            data_matrix = other
        elif isinstance(other, FDataGrid):
            self.__check_same_dimensions(other)
            data_matrix = other.data_matrix
        else:
            return NotImplemented

        return FDataGrid(self.data_matrix * data_matrix,
                         self.sample_points, self.sample_range,
                         self.dataset_label, self.axes_labels)

    def __truediv__(self, other):
        if isinstance(other, (numpy.ndarray, numbers.Number)):
            data_matrix = other
        elif isinstance(other, FDataGrid):
            self.__check_same_dimensions(other)
            data_matrix = other.data_matrix
        else:
            return NotImplemented

        return FDataGrid(self.data_matrix / data_matrix,
                         self.sample_points, self.sample_range,
                         self.dataset_label, self.axes_labels)

    def concatenate(self, other):
        """Joins samples from a similar FDataGrid object.

        Joins samples from another FDataGrid object if it has the same
        dimensions and sampling points.

        Args:
            other (:obj:`FDataGrid`): another FDataGrid object.

        Returns:
            :obj:`FDataGrid`: FDataGrid object with the samples from the two
            original objects.

        Examples:
            >>> fd = FDataGrid([1,2,4,5,8], range(5))
            >>> fd_2 = FDataGrid([3,4,7,9,2], range(5))
            >>> fd.concatenate(fd_2)
            FDataGrid(
                array([[1, 2, 4, 5, 8],
                       [3, 4, 7, 9, 2]]),
                sample_points=[array([0, 1, 2, 3, 4])],
                ...

        """
        # Checks
        self.__check_same_dimensions(other)
        return FDataGrid(numpy.concatenate((self.data_matrix,
                                            other.data_matrix), axis=0),
                         self.sample_points,
                         self.sample_range,
                         self.dataset_label,
                         self.axes_labels)

    def _set_labels(self, ax):
        """Sets labels if any.

        Args:
            ax (axes object): axes object that implements set_title,
                set_xlable and set_ylabel or title, xlabel and ylabel.
            """
        if self.dataset_label is not None:
            try:
                ax.set_title(self.dataset_label)
            except AttributeError:
                try:
                    ax.title(self.dataset_label)
                except AttributeError:
                    pass

        if self.axes_labels is not None:
            try:
                ax.set_xlabel(self.axes_labels[0])
                ax.set_ylabel(self.axes_labels[1])
            except AttributeError:
                try:
                    ax.xlabel(self.axes_labels[0])
                    ax.ylabel(self.axes_labels[1])
                except AttributeError:
                    pass

    def plot(self, ax=None, **kwargs):
        """Plots the FDatGrid object.

        Args:
            ax (axis object, optional): axis over with the graphs are plotted.
                Defaults to matplotlib current axis.
            **kwargs: keyword arguments to be passed to the
                matplotlib.pyplot.plot function.

        Returns:
            List of lines that were added to the plot.

        """
        if self.ndim_domain != 1:
            raise NotImplementedError("Plot only supported for functional "
                                      "data with a domain dimension of 1.")
        if self.ndim_image != 1:
            raise NotImplementedError("Plot only supported for functional "
                                      "data with a image dimension of 1.")
        if ax is None:
            ax = matplotlib.pyplot.gca()
        _plot = ax.plot(self.sample_points[0],
                        numpy.transpose(self.data_matrix),
                        **kwargs)
        self._set_labels(ax)

        return _plot

    def scatter(self, ax=None, **kwargs):
        """Scatter plot of the FDatGrid object.

        Args:
            ax (axis object, optional): axis over with the graphs are plotted.
                Defaults to matplotlib current axis.
            **kwargs: keyword arguments to be passed to the
                matplotlib.pyplot.scatter function.

        Returns:
            :obj:`matplotlib.collections.PathCollection`

        """
        if self.ndim_domain != 1:
            raise NotImplementedError("Scatter only supported for functional "
                                      "data with a domain dimension of 1.")

        if self.ndim_image != 1:
            raise NotImplementedError("Scatter only supported for functional "
                                      "data with a image dimension of 1.")
        if ax is None:
            ax = matplotlib.pyplot.gca()
        _plot = None
        for i in range(self.nsamples):
            _plot = ax.scatter(self.sample_points[0],
                               self.data_matrix[i],
                               **kwargs)
        self._set_labels(ax)
        return _plot

    def to_basis(self, basis, **kwargs):
        """Returns the basis representation of the object.

        Args:
            basis(Basis): basis object in which the functional data are
                going to be represented.
            **kwargs: keyword arguments to be passed to
                FDataBasis.from_data().

        Returns:
            FDataBasis: Basis representation of the funtional data
            object.

        Examples:
            >>> import numpy as np
            >>> import fda
            >>> t = np.linspace(0, 1, 5)
            >>> x = np.sin(2 * np.pi * t) + np.cos(2 * np.pi * t)
            >>> x
            array([ 1.,  1., -1., -1.,  1.])

            >>> fd = FDataGrid(x, t)
            >>> basis = fda.basis.Fourier((0, 1), nbasis=3)
            >>> fd_b = fd.to_basis(basis)
            >>> fd_b.coefficients.round(2)
            array([[0.  , 0.71, 0.71]])

        """
        if self.ndim_domain > 1:
            raise NotImplementedError("Only support 1 dimension on the "
                                      "domain.")
        return fdbasis.FDataBasis.from_data(self.data_matrix,
                                            self.sample_points[0],
                                            basis,
                                            **kwargs)

    def __str__(self):
        """ Return str(self). """
        return ('Data set:    ' + str(self.data_matrix)
                + '\nsample_points:    ' + str(self.sample_points)
                + '\ntime range:    ' + str(self.sample_range))

    def __repr__(self):
        """ Return repr(self). """
        return ("FDataGrid("
                "\n{},"
                "\nsample_points={},"
                "\nsample_range={},"
                "\ndataset_label={},"
                "\naxes_labels={})"
                .format(repr(self.data_matrix),
                        repr(self.sample_points),
                        repr(self.sample_range),
                        repr(self.dataset_label),
                        repr(self.axes_labels))).replace('\n', '\n    ')

    def __getitem__(self, key):
        """ Return self[key]. """
        if isinstance(key, tuple):
            # If there are not values for every dimension, the remaining ones
            # are kept
            key += (slice(None),) * (self.ndim_domain + 1 - len(key))

            sample_points = [self.sample_points[i][subkey]
                             for i, subkey in enumerate(
                                 key[1:1 + self.ndim_domain])]

            return FDataGrid(self.data_matrix[key],
                             sample_points,
                             self.sample_range, self.dataset_label,
                             self.axes_labels)
        return FDataGrid(self.data_matrix[key], self.sample_points,
                         self.sample_range, self.dataset_label,
                         self.axes_labels)
