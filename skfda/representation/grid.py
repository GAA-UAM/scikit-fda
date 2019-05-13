"""Discretised functional data module.

This module defines a class for representing functional data as a series of
lists of values, each representing the observation of a function measured in a
list of discretisation points.

"""

import numbers

import copy
import numpy
import scipy.stats.mstats


from . import basis as fdbasis
from .interpolation import SplineInterpolator
from . import FData
from .._utils import _list_of_arrays


__author__ = "Miguel Carbajo Berrocal"
__email__ = "miguel.carbajo@estudiante.uam.es"


class FDataGrid(FData):
    r"""Represent discretised functional data.

    Class for representing functional data as a set of curves discretised
    in a grid of points.

    Attributes:
        data_matrix (numpy.ndarray): a matrix where each entry of the first
            axis contains the values of a functional datum evaluated at the
            points of discretisation.
        sample_points (numpy.ndarray): 2 dimension matrix where each row
            contains the points of dicretisation for each axis of data_matrix.
        domain_range (numpy.ndarray): 2 dimension matrix where each row
            contains the bounds of the interval in which the functional data
            is considered to exist for each one of the axies.
        dataset_label (str): name of the dataset.
        axes_labels (list): list containing the labels of the different
            axis.
        extrapolation (str or Extrapolation): defines the default type of
            extrapolation. By default None, which does not apply any type of
            extrapolation. See `Extrapolation` for detailled information of the
            types of extrapolation.
        interpolator (GridInterpolator): Defines the type of interpolation
            applied in `evaluate`.
        keepdims (bool):

    Examples:
        Representation of a functional data object with 2 samples
        representing a function :math:`f : \mathbb{R}\longmapsto\mathbb{R}`.

        >>> data_matrix = [[1, 2], [2, 3]]
        >>> sample_points = [2, 4]
        >>> FDataGrid(data_matrix, sample_points)
        FDataGrid(
            array([[[1],
                    [2]],
        <BLANKLINE>
                   [[2],
                    [3]]]),
            sample_points=[array([2, 4])],
            domain_range=array([[2, 4]]),
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
                 domain_range=None, dataset_label=None,
                 axes_labels=None, extrapolation=None,
                 interpolator=None, keepdims=False):

        """Construct a FDataGrid object.

        Args:
            data_matrix (array_like): a matrix where each row contains the
                values of a functional datum evaluated at the
                points of discretisation.
            sample_points (array_like, optional): an array containing the
                points of discretisation where values have been recorded or a list
                of lists with each of the list containing the points of
                dicretisation for each axis.
            domain_range (tuple or list of tuples, optional): contains the
                edges of the interval in which the functional data is
                considered to exist (if the argument has 2 dimensions each
                row is interpreted as the limits of one of the dimension of
                the domain).
            dataset_label (str, optional): name of the dataset.
            axes_labels (list, optional): list containing the labels of the
                different axes. The length of the list must be equal to the sum of the
                number of dimensions of the domain plus the number of dimensions
                of the image.

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

            if not numpy.array_equal(data_shape, sample_points_shape):
                raise ValueError("Incorrect dimension in data_matrix and "
                                 "sample_points. Data has shape {} and sample "
                                 "points have shape {}"
                                 .format(data_shape, sample_points_shape))


        self._sample_range = numpy.array(
            [(self.sample_points[i][0], self.sample_points[i][-1])
             for i in range(self.ndim_domain)])

        if domain_range is None:
                self._domain_range = self.sample_range
            # Default value for domain_range is a list of tuples with
            # the first and last element of each list ofthe sample_points.
        else:
            self._domain_range = numpy.atleast_2d(domain_range)
            # sample range must by a 2 dimension matrix with as many rows as
            # dimensions in the domain and 2 columns
            if (self._domain_range.ndim != 2 or self._domain_range.shape[1] != 2
                    or self._domain_range.shape[0] != self.ndim_domain):
                raise ValueError("Incorrect shape of domain_range.")
            for i in range(self.ndim_domain):
                if (self._domain_range[i, 0] > self.sample_points[i][0]
                        or self._domain_range[i, -1] < self.sample_points[i]
                        [-1]):
                    raise ValueError("Sample points must be within the domain "
                                     "range.")

        # Adjust the data matrix if the dimension of the image is one
        if self.data_matrix.ndim == 1 + self.ndim_domain:
            self.data_matrix = self.data_matrix[..., numpy.newaxis]

        if axes_labels is not None and len(axes_labels) != (self.ndim_domain + self.ndim_image):
            raise ValueError("There must be a label for each of the"
                              "dimensions of the domain and the image.")

        self.interpolator = interpolator


        super().__init__(extrapolation, dataset_label, axes_labels, keepdims)


        return

    def round(self, decimals=0):
        """Evenly round to the given number of decimals.

        Args:
            decimals (int, optional): Number of decimal places to round to.
                If decimals is negative, it specifies the number of
                positions to the left of the decimal point. Defaults to 0.

        Returns:
            :obj:FDataGrid: Returns a FDataGrid object where all elements
            in its data_matrix are rounded .The real and
            imaginary parts of complex numbers are rounded separately.

        """
        return self.copy(data_matrix=self.data_matrix.round(decimals))

    @property
    def ndim_domain(self):
        """Return number of dimensions of the domain.

        Returns:
            int: Number of dimensions of the domain.

        """
        return len(self.sample_points)

    @property
    def ndim_image(self):
        """Return number of dimensions of the image.

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
        """Return number of dimensions of the data matrix.

        Returns:
            int: Number of dimensions of the data matrix.

        """
        return self.data_matrix.ndim

    @property
    def nsamples(self):
        """Return number of rows of the data_matrix. Also the number of samples.

        Returns:
            int: Number of samples of the FDataGrid object. Also the number of
                rows of the data_matrix.

        """
        return self.data_matrix.shape[0]

    @property
    def ncol(self):
        """Return number of columns of the data_matrix.

        Also the number of points of discretisation.

        Returns:
            int: Number of columns of the data_matrix.

        """
        return self.data_matrix.shape[1]

    @property
    def sample_range(self):
        """Return the edges of the interval in which the functional data is
            considered to exist by the sample points.

            Do not have to be equal to the domain_range.
        """
        return self._sample_range

    @property
    def domain_range(self):
        """Return the edges of the interval in which the functional data is
            considered to exist by the sample points.

            Do not have to be equal to the sample_range.
        """
        return self._domain_range

    @property
    def shape(self):
        """Dimensions (aka shape) of the data_matrix.

        Returns:
            list of int: List containing the length of the matrix on each of
            its axis. If the matrix is 2 dimensional shape returns [number of
            rows, number of columns].

        """
        return self.data_matrix.shape

    @property
    def interpolator(self):
        """Defines the type of interpolation applied in `evaluate`."""
        return self._interpolator

    @interpolator.setter
    def interpolator(self, new_interpolator):
        """Sets the interpolator of the FDataGrid."""
        if new_interpolator is None:
            new_interpolator = SplineInterpolator()

        self._interpolator = new_interpolator
        self._interpolator_evaluator = None

    @property
    def _evaluator(self):
        """Return the evaluator constructed by the interpolator."""

        if self._interpolator_evaluator is None:
            self._interpolator_evaluator = self._interpolator.evaluator(self)

        return self._interpolator_evaluator


    def _evaluate(self, eval_points, *, derivative=0):
        """"Evaluate the object or its derivatives at a list of values.

        Args:
            eval_points (array_like): List of points where the functions are
                evaluated. If a matrix of shape nsample x eval_points is given
                each sample is evaluated at the values in the corresponding row
                in eval_points.
            derivative (int, optional): Order of the derivative. Defaults to 0.

        Returns:
            (numpy.darray): Matrix whose rows are the values of the each
            function at the values specified in eval_points.

        """

        return self._evaluator.evaluate(eval_points, derivative=derivative)

    def _evaluate_composed(self, eval_points, *, derivative=0):
        """"Evaluate the object or its derivatives at a list of values.

        Args:
            eval_points (array_like): List of points where the functions are
                evaluated. If a matrix of shape nsample x eval_points is given
                each sample is evaluated at the values in the corresponding row
                in eval_points.
            derivative (int, optional): Order of the derivative. Defaults to 0.

        Returns:
            (numpy.darray): Matrix whose rows are the values of the each
            function at the values specified in eval_points.

        """

        return self._evaluator.evaluate_composed(eval_points,
                                                 derivative=derivative)

    def derivative(self, order=1):
        r"""Differentiate a FDataGrid object.

        It is calculated using lagged differences. If we call :math:`D` the
        data_matrix, :math:`D^1` the derivative of order 1 and :math:`T` the
        vector contaning the points of discretisation; :math:`D^1` is
        calculated as it follows:

        .. math::

            D^{1}_{ij} = \begin{cases}
            \frac{D_{i1} - D_{i2}}{ T_{1} - T_{2}}  & \mbox{if } j = 1 \\
            \frac{D_{i(m-1)} - D_{im}}{ T_{m-1} - T_m}  & \mbox{if }
                j = m \\
            \frac{D_{i(j-1)} - D_{i(j+1)}}{ T_{j-1} - T_{j+1}} & \mbox{if }
            1 < j < m
            \end{cases}

        Where m is the number of columns of the matrix :math:`D`.

        Order > 1 derivatives are calculated by using derivative recursively.

        Args:
            order (int, optional): Order of the derivative. Defaults to one.

        Examples:
            First order derivative

            >>> fdata = FDataGrid([1,2,4,5,8], range(5))
            >>> fdata.derivative()
            FDataGrid(
                array([[[ 1. ],
                        [ 1.5],
                        [ 1.5],
                        [ 2. ],
                        [ 3. ]]]),
                sample_points=[array([0, 1, 2, 3, 4])],
                domain_range=array([[0, 4]]),
                ...)

            Second order derivative

            >>> fdata = FDataGrid([1,2,4,5,8], range(5))
            >>> fdata.derivative(2)
            FDataGrid(
                array([[[ 0.5 ],
                        [ 0.25],
                        [ 0.25],
                        [ 0.75],
                        [ 1.  ]]]),
                sample_points=[array([0, 1, 2, 3, 4])],
                domain_range=array([[0, 4]]),
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
        if self.ndim_domain > 1 or self.ndim_image > 1:
            raise NotImplementedError("Not implemented for 2 or more"
                                      " dimensional data.")
        if numpy.isnan(self.data_matrix).any():
            raise ValueError("The FDataGrid object cannot contain nan "
                             "elements.")
        data_matrix = self.data_matrix[..., 0]
        sample_points = self.sample_points[0]
        for _ in range(order):
            mdata = []
            for i in range(self.nsamples):
                arr = (numpy.diff(data_matrix[i]) /
                       (sample_points[1:]
                        - sample_points[:-1]))
                arr = numpy.append(arr, arr[-1])
                arr[1:-1] += arr[:-2]
                arr[1:-1] /= 2
                mdata.append(arr)
            data_matrix = numpy.array(mdata)

        if self.dataset_label:
            dataset_label = "{} - {} derivative".format(self.dataset_label,
                                                        order)
        else:
            dataset_label = None

        return self.copy(data_matrix=data_matrix, sample_points=sample_points,
                         dataset_label=dataset_label)

    def __check_same_dimensions(self, other):
        if self.data_matrix.shape[1] != other.data_matrix.shape[1]:
            raise ValueError("Error in columns dimensions")
        if not numpy.array_equal(self.sample_points, other.sample_points):
            raise ValueError("Sample points for both objects must be equal")

    def mean(self):
        """Compute the mean of all the samples.

        Returns:
            FDataGrid : A FDataGrid object with just one sample representing
            the mean of all the samples in the original object.

        """
        return self.copy(data_matrix=self.data_matrix.mean(axis=0,
                                                           keepdims=True))

    def var(self):
        """Compute the variance of a set of samples in a FDataGrid object.

        Returns:
            FDataGrid: A FDataGrid object with just one sample representing the
            variance of all the samples in the original FDataGrid object.

        """
        return self.copy(data_matrix=[numpy.var(self.data_matrix, 0)])

    def cov(self):
        """Compute the covariance.

        Calculates the covariance matrix representing the covariance of the
        functional samples at the observation points.

        Returns:
            numpy.darray: Matrix of covariances.

        """

        if self.dataset_label is not None:
            dataset_label = self.dataset_label + ' - covariance'
        else:
            dataset_label = None

        return self.copy(data_matrix=numpy.cov(self.data_matrix,
                                               rowvar=False)[numpy.newaxis, ...],
                         sample_points=[self.sample_points[0],
                                        self.sample_points[0]],
                         domain_range=[self.domain_range[0],
                                       self.domain_range[0]],
                         dataset_label=dataset_label)

    def gmean(self):
        """Compute the geometric mean of all samples in the FDataGrid object.

        Returns:
            FDataGrid: A FDataGrid object with just one sample representing
            the geometric mean of all the samples in the original
            FDataGrid object.

        """
        return self.copy(data_matrix=
                         [scipy.stats.mstats.gmean(self.data_matrix, 0)])

    def __add__(self, other):
        """Addition for FDataGrid object.

        It supports other FDataGrid objects, numpy.ndarray and numbers.

        """
        if isinstance(other, (numpy.ndarray, numbers.Number)):
            data_matrix = other
        elif isinstance(other, FDataGrid):
            self.__check_same_dimensions(other)
            data_matrix = other.data_matrix
        else:
            return NotImplemented

        return self.copy(data_matrix=self.data_matrix + data_matrix)

    def __radd__(self, other):
        """Addition for FDataGrid object.

        It supports other FDataGrid objects, numpy.ndarray and numbers.

        """

        return self.__add__(other)

    def __sub__(self, other):
        """Subtraction for FDataGrid object.

        It supports other FDataGrid objects, numpy.ndarray and numbers.

        """
        if isinstance(other, (numpy.ndarray, numbers.Number)):
            data_matrix = other
        elif isinstance(other, FDataGrid):
            self.__check_same_dimensions(other)
            data_matrix = other.data_matrix
        else:
            return NotImplemented

        return self.copy(data_matrix=self.data_matrix - data_matrix)

    def __rsub__(self, other):
        """Right Subtraction for FDataGrid object.

        It supports other FDataGrid objects, numpy.ndarray and numbers.

        """
        if isinstance(other, (numpy.ndarray, numbers.Number)):
            data_matrix = other
        elif isinstance(other, FDataGrid):
            self.__check_same_dimensions(other)
            data_matrix = other.data_matrix
        else:
            return NotImplemented

        return self.copy(data_matrix=data_matrix - self.data_matrix)

    def __mul__(self, other):
        """Multiplication for FDataGrid object.

        It supports other FDataGrid objects, numpy.ndarray and numbers.

        """
        if isinstance(other, (numpy.ndarray, numbers.Number)):
            data_matrix = other
        elif isinstance(other, FDataGrid):
            self.__check_same_dimensions(other)
            data_matrix = other.data_matrix
        else:
            return NotImplemented

        return self.copy(data_matrix=self.data_matrix * data_matrix)

    def __rmul__(self, other):
        """Multiplication for FDataGrid object.

        It supports other FDataGrid objects, numpy.ndarray and numbers.

        """
        return self.__mul__(other)

    def __truediv__(self, other):
        """Division for FDataGrid object.

        It supports other FDataGrid objects, numpy.ndarray and numbers.

        """
        if isinstance(other, (numpy.ndarray, numbers.Number)):
            data_matrix = other
        elif isinstance(other, FDataGrid):
            self.__check_same_dimensions(other)
            data_matrix = other.data_matrix
        else:
            return NotImplemented

        return self.copy(data_matrix=self.data_matrix / data_matrix)

    def __rtruediv__(self, other):
        """Division for FDataGrid object.

        It supports other FDataGrid objects, numpy.ndarray and numbers.

        """
        if isinstance(other, (numpy.ndarray, numbers.Number)):
            data_matrix = other
        elif isinstance(other, FDataGrid):
            self.__check_same_dimensions(other)
            data_matrix = other.data_matrix
        else:
            return NotImplemented

        return self.copy(data_matrix=data_matrix / self.data_matrix)


    def concatenate(self, other):
        """Join samples from a similar FDataGrid object.

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
                array([[[1],
                        [2],
                        [4],
                        [5],
                        [8]],
            <BLANKLINE>
                       [[3],
                        [4],
                        [7],
                        [9],
                        [2]]]),
                sample_points=[array([0, 1, 2, 3, 4])],
                domain_range=array([[0, 4]]),
                ...)

        """
        # Checks
        self.__check_same_dimensions(other)

        return self.copy(data_matrix=numpy.concatenate((self.data_matrix,
                                                        other.data_matrix),
                                                       axis=0))


    def scatter(self, fig=None, ax=None, nrows=None, ncols=None, **kwargs):
        """Scatter plot of the FDatGrid object.

        Args:
            fig (figure object, optional): figure over with the graphs are plotted in case ax is not specified.
                If None and ax is also None, the figure is initialized.
            ax (list of axis objects, optional): axis over where the graphs are plotted. If None, see param fig.
            nrows(int, optional): designates the number of rows of the figure to plot the different dimensions of the
                image. Only specified if fig and ax are None.
            ncols(int, optional): designates the number of columns of the figure to plot the different dimensions of the
                image. Only specified if fig and ax are None.
            **kwargs: keyword arguments to be passed to the matplotlib.pyplot.scatter function;

        Returns:
            fig (figure object): figure object in which the graphs are plotted in case ax is None.
            ax (axes object): axes in which the graphs are plotted.

        """
        fig, ax = self.generic_plotting_checks(fig, ax, nrows, ncols)

        if self.ndim_domain == 1:
            for i in range(self.ndim_image):
                for j in range(self.nsamples):
                    ax[i].scatter(self.sample_points[0], self.data_matrix[j, :, i].T, **kwargs)
        else:
            X = self.sample_points[0]
            Y = self.sample_points[1]
            X, Y = numpy.meshgrid(X, Y)
            for i in range(self.ndim_image):
                for j in range(self.nsamples):
                    ax[i].scatter(X, Y, self.data_matrix[j, :, :, i].T, **kwargs)

        self.set_labels(fig, ax)

        return fig, ax


    def to_basis(self, basis, **kwargs):
        """Return the basis representation of the object.

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
            >>> import skfda
            >>> t = np.linspace(0, 1, 5)
            >>> x = np.sin(2 * np.pi * t) + np.cos(2 * np.pi * t)
            >>> x
            array([ 1.,  1., -1., -1.,  1.])

            >>> fd = FDataGrid(x, t)
            >>> basis = skfda.representation.basis.Fourier((0, 1), nbasis=3)
            >>> fd_b = fd.to_basis(basis)
            >>> fd_b.coefficients.round(2)
            array([[ 0.  , 0.71, 0.71]])

        """
        if self.ndim_domain > 1:
            raise NotImplementedError("Only support 1 dimension on the "
                                      "domain.")
        elif self.ndim_image > 1:
            raise NotImplementedError("Only support 1 dimension on the "
                                      "image.")
        return fdbasis.FDataBasis.from_data(self.data_matrix[..., 0],
                                            self.sample_points[0],
                                            basis,
                                            keepdims=self.keepdims,
                                            **kwargs)

    def to_grid(self, sample_points=None):
        """Return the discrete representation of the object.

        Args:
            sample_points (array_like, optional):  2 dimension matrix where each
            row contains the points of dicretisation for each axis of
            data_matrix.

        Returns:
              FDataGrid: Discrete representation of the functional data
              object.

        """
        if sample_points is None:
            sample_points = self.sample_points

        return self.copy(data_matrix=self.evaluate(sample_points, grid=True),
                         sample_points=sample_points)



    def copy(self, *, data_matrix=None, sample_points=None,
                 domain_range=None, dataset_label=None,
                 axes_labels=None, extrapolation=None,
                 interpolator=None, keepdims=None):
        """Returns a copy of the FDataGrid.

        If an argument is provided the corresponding attribute in the new copy
        is updated.

        """

        if data_matrix is None:
            # The data matrix won't be writeable
            data_matrix = self.data_matrix

        if sample_points is None:
            # Sample points won`t be writeable
            sample_points = self.sample_points

        if domain_range is None:
            domain_range = copy.deepcopy(self.domain_range)

        if dataset_label is None:
            dataset_label = copy.copy(self.dataset_label)

        if axes_labels is None:
            axes_labels = copy.copy(self.axes_labels)

        if extrapolation is None:
            extrapolation = self.extrapolation

        if interpolator is None:
            interpolator = self.interpolator

        if keepdims is None:
            keepdims = self.keepdims

        return FDataGrid(data_matrix, sample_points=sample_points,
                     domain_range=domain_range, dataset_label=dataset_label,
                     axes_labels=axes_labels, extrapolation=extrapolation,
                     interpolator=interpolator, keepdims=keepdims)


    def shift(self, shifts, *, restrict_domain=False, extrapolation=None,
              eval_points=None):
        """Perform a shift of the curves.

        Args:
            shifts (array_like or numeric): List with the shifts
                corresponding for each sample or numeric with the shift to apply
                to all samples.
            restrict_domain (bool, optional): If True restricts the domain to
                avoid evaluate points outside the domain using extrapolation.
                Defaults uses extrapolation.
            extrapolation (str or Extrapolation, optional): Controls the
                extrapolation mode for elements outside the domain range.
                By default uses the method defined in fd. See extrapolation to
                more information.
            eval_points (array_like, optional): Set of points where
                the functions are evaluated to obtain the discrete
                representation of the object to operate. If an empty list the
                current sample_points are used to unificate the domain of the
                shifted data.

        Returns:
            :class:`FDataGrid` with the shifted data.
        """


        if numpy.isscalar(shifts):
            shifts = [shifts]

        shifts = numpy.array(shifts)

        # Case unidimensional treated as the multidimensional
        if self.ndim_domain == 1 and shifts.ndim == 1 and shifts.shape[0] != 1:
            shifts = shifts[:, numpy.newaxis]

        # Case same shift for all the curves
        if shifts.shape[0] == self.ndim_domain and shifts.ndim ==1:

            # Column vector with shapes
            shifts = numpy.atleast_2d(shifts).T

            sample_points = self.sample_points + shifts
            domain_range = self.domain_range + shifts

            return self.copy(sample_points=sample_points,
                             domain_range=domain_range)


        if shifts.shape[0] != self.nsamples:
            raise ValueError(f"shifts vector ({shifts.shape[0]}) must have the "
                             f"same length than the number of samples "
                             f"({self.nsamples})")

        if eval_points is None:
            eval_points = self.sample_points



        if restrict_domain:
            domain = numpy.asarray(self.domain_range)
            a = domain[:,0] - numpy.atleast_1d(numpy.min(numpy.min(shifts, axis=1), 0))
            b = domain[:,1] - numpy.atleast_1d(numpy.max(numpy.max(shifts, axis=1), 0))

            domain = numpy.vstack((a,b)).T

            eval_points = [eval_points[i][
                numpy.logical_and(eval_points[i] >= domain[i,0],
                                  eval_points[i] <= domain[i,1])]
                           for i in range(self.ndim_domain)]

        else:
            domain = self.domain_range

        eval_points = numpy.asarray(eval_points)


        eval_points_repeat = numpy.repeat(eval_points[numpy.newaxis, :],
                                       self.nsamples, axis=0)

        # Solve problem with cartesian and matrix indexing
        if self.ndim_domain > 1:
            shifts[:,:2] = numpy.flip(shifts[:,:2], axis=1)

        shifts = numpy.repeat(shifts[..., numpy.newaxis],
                              eval_points.shape[1], axis=2)

        eval_points_shifted = eval_points_repeat + shifts


        grid = True if self.ndim_domain > 1 else False

        data_matrix = self.evaluate(eval_points_shifted,
                                    extrapolation=extrapolation,
                                    aligned_evaluation=False,
                                    grid=True)


        return self.copy(data_matrix=data_matrix, sample_points=eval_points,
                         domain_range=domain)

    def compose(self, fd, *, eval_points=None):
        """Composition of functions.

        Performs the composition of functions.

        Args:
            fd (:class:`FData`): FData object to make the composition. Should
                have the same number of samples and image dimension equal to 1.
            eval_points (array_like): Points to perform the evaluation.
        """

        if self.ndim_domain != fd.ndim_image:
            raise ValueError(f"Dimension of codomain of first function do not "
                             f"match with the domain of the second function "
                             f"({self.ndim_domain})!=({fd.ndim_image}).")

        # All composed with same function
        if fd.nsamples == 1 and self.nsamples != 1:
            fd = fd.copy(data_matrix=numpy.repeat(fd.data_matrix, self.nsamples,
                                               axis=0))

        if fd.ndim_domain == 1:
            if eval_points is None:
                try:
                    eval_points = fd.sample_points[0]
                except:
                    eval_points = numpy.linspace(*fd.domain_range[0], 201)

            eval_points_transformation = fd(eval_points, keepdims=False)
            data_matrix = self(eval_points_transformation,
                               aligned_evaluation=False)
        else:
            if eval_points is None:
                eval_points = fd.sample_points

            grid_transformation = fd(eval_points, grid=True, keepdims=True)

            lengths = [len(ax) for ax in eval_points]

            eval_points_transformation =  numpy.empty((self.nsamples,
                                                       numpy.prod(lengths),
                                                       self.ndim_domain))


            for i in range(self.nsamples):
                eval_points_transformation[i] = numpy.array(
                    list(map(numpy.ravel, grid_transformation[i].T))
                    ).T

            data_flatten = self(eval_points_transformation,
                               aligned_evaluation=False)

            data_matrix = data_flatten.reshape((self.nsamples, *lengths,
                                                self.ndim_image))


        return self.copy(data_matrix=data_matrix,
                         sample_points=eval_points,
                         domain_range=fd.domain_range)



    def __str__(self):
        """Return str(self)."""
        return ('Data set:    ' + str(self.data_matrix)
                + '\nsample_points:    ' + str(self.sample_points)
                + '\ntime range:    ' + str(self.domain_range))

    def __repr__(self):
        """Return repr(self)."""
        return (f"FDataGrid("
                f"\n{repr(self.data_matrix)},"
                f"\nsample_points={repr(self.sample_points)},"
                f"\ndomain_range={repr(self.domain_range)},"
                f"\ndataset_label={repr(self.dataset_label)},"
                f"\naxes_labels={repr(self.axes_labels)},"
                f"\nextrapolation={repr(self.extrapolation)},"
                f"\ninterpolator={repr(self.interpolator)},"
                f"\nkeepdims={repr(self.keepdims)})").replace('\n', '\n    ')

    def __getitem__(self, key):
        """Return self[key]."""
        if isinstance(key, tuple):
            # If there are not values for every dimension, the remaining ones
            # are kept
            key += (slice(None),) * (self.ndim_domain + 1 - len(key))

            sample_points = [self.sample_points[i][subkey]
                             for i, subkey in enumerate(
                                 key[1:1 + self.ndim_domain])]

            return self.copy(data_matrix=self.data_matrix[key],
                             sample_points=sample_points)

        if isinstance(key, int):
            return self.copy(data_matrix=self.data_matrix[key:key + 1])

        else:
            return self.copy(data_matrix=self.data_matrix[key])

    def __array_ufunc__(self, ufunc, method, *inputs, **kwargs):

        for i in inputs:
            if isinstance(i, FDataGrid) and not numpy.all(i.sample_points ==
                                                          self.sample_points):
                return NotImplemented

        new_inputs = [i.data_matrix if isinstance(i, FDataGrid)
                      else i for i in inputs]

        outputs = kwargs.pop('out', None)
        if outputs:
            new_outputs = [o.data_matrix if isinstance(o, FDataGrid)
                           else o for o in outputs]
            kwargs['out'] = tuple(new_outputs)
        else:
            new_outputs = (None,) * ufunc.nout

        results = getattr(ufunc, method)(*new_inputs, **kwargs)
        if results is NotImplemented:
            return NotImplemented

        if ufunc.nout == 1:
            results = (results,)

        results = tuple((result
                         if output is None else output)
                        for result, output in zip(results, new_outputs))

        results = [self.copy(data_matrix=r) for r in results]

        return results[0] if len(results) == 1 else results
