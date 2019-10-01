"""Discretised functional data module.

This module defines a class for representing functional data as a series of
lists of values, each representing the observation of a function measured in a
list of discretisation points.

"""

import copy
import numbers

import pandas.api.extensions
import scipy.stats.mstats

import numpy as np

from . import FData
from . import basis as fdbasis
from .._utils import _list_of_arrays, constants
from .interpolation import SplineInterpolator


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

        >>> FDataGrid(np.array([1,2,4,5,8]), range(6))
        Traceback (most recent call last):
            ....
        ValueError: Incorrect dimension in data_matrix and sample_points...


        FDataGrid support higher dimensional data both in the domain and image.
        Representation of a functional data object with 2 samples
        representing a function :math:`f : \mathbb{R}\longmapsto\mathbb{R}^2`.

        >>> data_matrix = [[[1, 0.3], [2, 0.4]], [[2, 0.5], [3, 0.6]]]
        >>> sample_points = [2, 4]
        >>> fd = FDataGrid(data_matrix, sample_points)
        >>> fd.dim_domain, fd.dim_codomain
        (1, 2)

        Representation of a functional data object with 2 samples
        representing a function :math:`f : \mathbb{R}^2\longmapsto\mathbb{R}`.

        >>> data_matrix = [[[1, 0.3], [2, 0.4]], [[2, 0.5], [3, 0.6]]]
        >>> sample_points = [[2, 4], [3,6]]
        >>> fd = FDataGrid(data_matrix, sample_points)
        >>> fd.dim_domain, fd.dim_codomain
        (2, 1)

    """

    class _CoordinateIterator:
        """Internal class to iterate through the image coordinates."""

        def __init__(self, fdatagrid):
            """Create an iterator through the image coordinates."""
            self._fdatagrid = fdatagrid

        def __iter__(self):
            """Return an iterator through the image coordinates."""

            for i in range(len(self)):
                yield self[i]

        def __getitem__(self, key):
            """Get a specific coordinate."""
            axes_labels = self._fdatagrid._get_labels_coordinates(key)

            return self._fdatagrid.copy(
                data_matrix=self._fdatagrid.data_matrix[..., key],
                axes_labels=axes_labels)

        def __len__(self):
            """Return the number of coordinates."""
            return self._fdatagrid.dim_codomain

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
                points of discretisation where values have been recorded or a
                list of lists with each of the list containing the points of
                dicretisation for each axis.
            domain_range (tuple or list of tuples, optional): contains the
                edges of the interval in which the functional data is
                considered to exist (if the argument has 2 dimensions each
                row is interpreted as the limits of one of the dimension of
                the domain).
            dataset_label (str, optional): name of the dataset.
            axes_labels (list, optional): list containing the labels of the
                different axes. The length of the list must be equal to the sum
                of the number of dimensions of the domain plus the number of
                dimensions of the image.
        """
        self.data_matrix = np.atleast_2d(data_matrix)

        if sample_points is None:
            self.sample_points = _list_of_arrays(
                [np.linspace(0, 1, self.data_matrix.shape[i]) for i in
                 range(1, self.data_matrix.ndim)])

        else:
            # Check that the dimension of the data matches the sample_points
            # list

            self.sample_points = _list_of_arrays(sample_points)

            data_shape = self.data_matrix.shape[1: 1 + self.dim_domain]
            sample_points_shape = [len(i) for i in self.sample_points]

            if not np.array_equal(data_shape, sample_points_shape):
                raise ValueError("Incorrect dimension in data_matrix and "
                                 "sample_points. Data has shape {} and sample "
                                 "points have shape {}"
                                 .format(data_shape, sample_points_shape))

        self._sample_range = np.array(
            [(self.sample_points[i][0], self.sample_points[i][-1])
             for i in range(self.dim_domain)])

        if domain_range is None:
            self._domain_range = self.sample_range
            # Default value for domain_range is a list of tuples with
            # the first and last element of each list ofthe sample_points.
        else:
            self._domain_range = np.atleast_2d(domain_range)
            # sample range must by a 2 dimension matrix with as many rows as
            # dimensions in the domain and 2 columns
            if (self._domain_range.ndim != 2
                    or self._domain_range.shape[1] != 2
                    or self._domain_range.shape[0] != self.dim_domain):
                raise ValueError("Incorrect shape of domain_range.")
            for i in range(self.dim_domain):
                if (self._domain_range[i, 0] > self.sample_points[i][0]
                        or self._domain_range[i, -1] < self.sample_points[i]
                        [-1]):
                    raise ValueError("Sample points must be within the domain "
                                     "range.")

        # Adjust the data matrix if the dimension of the image is one
        if self.data_matrix.ndim == 1 + self.dim_domain:
            self.data_matrix = self.data_matrix[..., np.newaxis]

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
    def dim_domain(self):
        """Return number of dimensions of the domain.

        Returns:
            int: Number of dimensions of the domain.

        """
        return len(self.sample_points)

    @property
    def dim_codomain(self):
        """Return number of dimensions of the image.

        Returns:
            int: Number of dimensions of the image.

        """
        try:
            # The dimension of the image is the length of the array that can
            #  be extracted from the data_matrix using all the dimensions of
            #  the domain.
            return self.data_matrix.shape[1 + self.dim_domain]
        # If there is no array that means the dimension of the image is 1.
        except IndexError:
            return 1

    @property
    def coordinates(self):
        r"""Returns an object to access to the image coordinates.

        If the functional object contains multivariate samples
        :math:`f: \mathbb{R}^n \rightarrow \mathbb{R}^d`, this class allows
        iterate and get coordinates of the vector
        :math:`f = (f_0, ..., f_{d-1})`.

        Examples:

            We will construct a dataset of curves in :math:`\mathbb{R}^3`

            >>> from skfda.datasets import make_multimodal_samples
            >>> fd = make_multimodal_samples(dim_codomain=3, random_state=0)
            >>> fd.dim_codomain
            3

            The functions of this dataset are vectorial functions
            :math:`f(t) = (f_0(t), f_1(t), f_2(t))`. We can obtain a specific
            component of the vector, for example, the first one.

            >>> fd_0 = fd.coordinates[0]
            >>> fd_0
            FDataGrid(...)

            The object returned has image dimension equal to 1

            >>> fd_0.dim_codomain
            1

            Or we can get multiple components, it can be accesed as a 1-d
            numpy array of coordinates, for example, :math:`(f_0(t), f_1(t))`.

            >>> fd_01 = fd.coordinates[0:2]
            >>> fd_01.dim_codomain
            2

            We can use this method to iterate throught all the coordinates.

            >>> for fd_i in fd.coordinates:
            ...     fd_i.dim_codomain
            1
            1
            1

            This object can be used to split a FDataGrid in a list with
            their components.

            >>> fd_list = list(fd.coordinates)
            >>> len(fd_list)
            3

        """

        return FDataGrid._CoordinateIterator(self)

    @property
    def n_samples(self):
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
        if self.dim_domain != 1:
            raise NotImplementedError(
                "This method only works when the dimension "
                "of the domain of the FDatagrid object is "
                "one.")
        if order < 1:
            raise ValueError("The order of a derivative has to be greater "
                             "or equal than 1.")
        if self.dim_domain > 1 or self.dim_codomain > 1:
            raise NotImplementedError("Not implemented for 2 or more"
                                      " dimensional data.")
        if np.isnan(self.data_matrix).any():
            raise ValueError("The FDataGrid object cannot contain nan "
                             "elements.")
        data_matrix = self.data_matrix[..., 0]
        sample_points = self.sample_points[0]
        for _ in range(order):
            mdata = []
            for i in range(self.n_samples):
                arr = (np.diff(data_matrix[i]) /
                       (sample_points[1:]
                        - sample_points[:-1]))
                arr = np.append(arr, arr[-1])
                arr[1:-1] += arr[:-2]
                arr[1:-1] /= 2
                mdata.append(arr)
            data_matrix = np.array(mdata)

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
        if not np.array_equal(self.sample_points, other.sample_points):
            raise ValueError("Sample points for both objects must be equal")

    def mean(self, weights=None):
        """Compute the mean of all the samples.

        weights (array-like, optional): List of weights.
        Returns:
            FDataGrid : A FDataGrid object with just one sample representing
            the mean of all the samples in the original object.

        """
        if weights is not None:

            return self.copy(data_matrix=np.average(
                self.data_matrix, weights=weights, axis=0)[np.newaxis, ...]
            )

        return self.copy(data_matrix=self.data_matrix.mean(axis=0,
                                                           keepdims=True))

    def var(self):
        """Compute the variance of a set of samples in a FDataGrid object.

        Returns:
            FDataGrid: A FDataGrid object with just one sample representing the
            variance of all the samples in the original FDataGrid object.

        """
        return self.copy(data_matrix=[np.var(self.data_matrix, 0)])

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

        return self.copy(data_matrix=np.cov(self.data_matrix,
                                            rowvar=False)[np.newaxis, ...],
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
        return self.copy(data_matrix=[
            scipy.stats.mstats.gmean(self.data_matrix, 0)])

    def __eq__(self, other):
        """Comparison of FDataGrid objects"""
        if not isinstance(other, FDataGrid):
            return NotImplemented

        if not np.array_equal(self.data_matrix, other.data_matrix):
            return False

        if len(self.sample_points) != len(other.sample_points):
            return False

        for a, b in zip(self.sample_points, other.sample_points):
            if not np.array_equal(a, b):
                return False

        if not np.array_equal(self.domain_range, other.domain_range):
            return False

        if self.dataset_label != other.dataset_label:
            return False

        if self.axes_labels is None or other.axes_labels is None:
            # Both must be None
            if self.axes_labels is not other.axes_labels:
                return False
        else:
            if len(self.axes_labels) != len(other.axes_labels):
                return False

            for a, b in zip(self.axes_labels, other.axes_labels):
                if a != b:
                    return False

        if self.extrapolation != other.extrapolation:
            return False

        if self.interpolator != other.interpolator:
            return False

        return True

    def __add__(self, other):
        """Addition for FDataGrid object.

        It supports other FDataGrid objects, numpy.ndarray and numbers.

        """
        if isinstance(other, (np.ndarray, numbers.Number)):
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
        if isinstance(other, (np.ndarray, numbers.Number)):
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
        if isinstance(other, (np.ndarray, numbers.Number)):
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
        if isinstance(other, (np.ndarray, numbers.Number)):
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
        if isinstance(other, (np.ndarray, numbers.Number)):
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
        if isinstance(other, (np.ndarray, numbers.Number)):
            data_matrix = other
        elif isinstance(other, FDataGrid):
            self.__check_same_dimensions(other)
            data_matrix = other.data_matrix
        else:
            return NotImplemented

        return self.copy(data_matrix=data_matrix / self.data_matrix)

    def concatenate(self, *others, as_coordinates=False):
        """Join samples from a similar FDataGrid object.

        Joins samples from another FDataGrid object if it has the same
        dimensions and sampling points.

        Args:
            others (:obj:`FDataGrid`): Objects to be concatenated.
            as_coordinates (boolean, optional):  If False concatenates as
                new samples, else, concatenates the other functions as
                new components of the image. Defaults to false.

        Returns:
            :obj:`FDataGrid`: FDataGrid object with the samples from the
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
        if not as_coordinates:
            for other in others:
                self.__check_same_dimensions(other)

        elif not all([np.array_equal(self.sample_points, other.sample_points)
                      for other in others]):
            raise ValueError("All the FDataGrids must be sampled in the  same "
                             "sample points.")

        elif any([self.n_samples != other.n_samples for other in others]):

            raise ValueError(f"All the FDataGrids must contain the same "
                             f"number of samples {self.n_samples} to "
                             f"concatenate as a new coordinate.")

        data = [self.data_matrix] + [other.data_matrix for other in others]

        if as_coordinates:
            return self.copy(data_matrix=np.concatenate(data, axis=-1),
                             axes_labels=(
                                 self._join_labels_coordinates(*others)))

        else:
            return self.copy(data_matrix=np.concatenate(data, axis=0))

    def scatter(self, *args, **kwargs):
        """Scatter plot of the FDatGrid object.

        Args:
            fig (figure object, optional): figure over with the graphs are
                plotted in case ax is not specified. If None and ax is also
                None, the figure is initialized.
            axes (list of axis objects, optional): axis over where the graphs
                are plotted. If None, see param fig.
            n_rows(int, optional): designates the number of rows of the figure
                to plot the different dimensions of the image. Only specified
                if fig and ax are None.
            n_cols(int, optional): designates the number of columns of the
                figure to plot the different dimensions of the image. Only
                specified if fig and ax are None.
            kwargs: keyword arguments to be passed to the
                matplotlib.pyplot.scatter function;

        Returns:
            fig (figure): figure object in which the graphs are plotted.


        """
        from ..exploratory.visualization.representation import plot_scatter

        return plot_scatter(self, *args, **kwargs)

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
            >>> basis = skfda.representation.basis.Fourier(n_basis=3)
            >>> fd_b = fd.to_basis(basis)
            >>> fd_b.coefficients.round(2)
            array([[ 0.  , 0.71, 0.71]])

        """
        if self.dim_domain > 1:
            raise NotImplementedError("Only support 1 dimension on the "
                                      "domain.")
        elif self.dim_codomain > 1:
            raise NotImplementedError("Only support 1 dimension on the "
                                      "image.")

        # Readjust the domain range if there was not an explicit one
        if basis._domain_range is None:
            basis = basis.copy()
            basis.domain_range = self.domain_range

        return fdbasis.FDataBasis.from_data(self.data_matrix[..., 0],
                                            self.sample_points[0],
                                            basis,
                                            keepdims=self.keepdims,
                                            **kwargs)

    def to_grid(self, sample_points=None):
        """Return the discrete representation of the object.

        Args:
            sample_points (array_like, optional):  2 dimension matrix where
            each row contains the points of dicretisation for each axis of
            data_matrix.

        Returns:
              FDataGrid: Discrete representation of the functional data
              object.

        """
        if sample_points is None:
            sample_points = self.sample_points

        return self.copy(data_matrix=self.evaluate(sample_points, grid=True),
                         sample_points=sample_points)

    def copy(self, *,
             deep=False,  # For Pandas compatibility
             data_matrix=None, sample_points=None,
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
                         domain_range=domain_range,
                         dataset_label=dataset_label,
                         axes_labels=axes_labels, extrapolation=extrapolation,
                         interpolator=interpolator, keepdims=keepdims)

    def shift(self, shifts, *, restrict_domain=False, extrapolation=None,
              eval_points=None):
        """Perform a shift of the curves.

        Args:
            shifts (array_like or numeric): List with the shifts
                corresponding for each sample or numeric with the shift to
                apply to all samples.
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

        if np.isscalar(shifts):
            shifts = [shifts]

        shifts = np.array(shifts)

        # Case unidimensional treated as the multidimensional
        if self.dim_domain == 1 and shifts.ndim == 1 and shifts.shape[0] != 1:
            shifts = shifts[:, np.newaxis]

        # Case same shift for all the curves
        if shifts.shape[0] == self.dim_domain and shifts.ndim == 1:

            # Column vector with shapes
            shifts = np.atleast_2d(shifts).T

            sample_points = self.sample_points + shifts
            domain_range = self.domain_range + shifts

            return self.copy(sample_points=sample_points,
                             domain_range=domain_range)
        if shifts.shape[0] != self.n_samples:
            raise ValueError(f"shifts vector ({shifts.shape[0]}) must have the"
                             f" same length than the number of samples "
                             f"({self.n_samples})")

        if eval_points is None:
            eval_points = self.sample_points

        if restrict_domain:
            domain = np.asarray(self.domain_range)
            a = domain[:, 0] - np.atleast_1d(np.min(np.min(shifts, axis=1), 0))
            b = domain[:, 1] - np.atleast_1d(np.max(np.max(shifts, axis=1), 0))

            domain = np.vstack((a, b)).T

            eval_points = [eval_points[i][
                np.logical_and(eval_points[i] >= domain[i, 0],
                               eval_points[i] <= domain[i, 1])]
                           for i in range(self.dim_domain)]

        else:
            domain = self.domain_range

        eval_points = np.asarray(eval_points)

        eval_points_repeat = np.repeat(eval_points[np.newaxis, :],
                                       self.n_samples, axis=0)

        # Solve problem with cartesian and matrix indexing
        if self.dim_domain > 1:
            shifts[:, :2] = np.flip(shifts[:, :2], axis=1)

        shifts = np.repeat(shifts[..., np.newaxis],
                           eval_points.shape[1], axis=2)

        eval_points_shifted = eval_points_repeat + shifts

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

        if self.dim_domain != fd.dim_codomain:
            raise ValueError(f"Dimension of codomain of first function do not "
                             f"match with the domain of the second function "
                             f"({self.dim_domain})!=({fd.dim_codomain}).")

        # All composed with same function
        if fd.n_samples == 1 and self.n_samples != 1:
            fd = fd.copy(data_matrix=np.repeat(fd.data_matrix, self.n_samples,
                                               axis=0))

        if fd.dim_domain == 1:
            if eval_points is None:
                try:
                    eval_points = fd.sample_points[0]
                except AttributeError:
                    eval_points = np.linspace(*fd.domain_range[0],
                                              constants.N_POINTS_COARSE_MESH)

            eval_points_transformation = fd(eval_points, keepdims=False)
            data_matrix = self(eval_points_transformation,
                               aligned_evaluation=False)
        else:
            if eval_points is None:
                eval_points = fd.sample_points

            grid_transformation = fd(eval_points, grid=True, keepdims=True)

            lengths = [len(ax) for ax in eval_points]

            eval_points_transformation = np.empty((self.n_samples,
                                                   np.prod(lengths),
                                                   self.dim_domain))

            for i in range(self.n_samples):
                eval_points_transformation[i] = np.array(
                    list(map(np.ravel, grid_transformation[i].T))
                ).T

            data_flatten = self(eval_points_transformation,
                                aligned_evaluation=False)

            data_matrix = data_flatten.reshape((self.n_samples, *lengths,
                                                self.dim_codomain))

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

        if self.axes_labels is None:
            axes_labels = None
        else:
            axes_labels = self.axes_labels.tolist()

        return (f"FDataGrid("
                f"\n{repr(self.data_matrix)},"
                f"\nsample_points={repr(self.sample_points)},"
                f"\ndomain_range={repr(self.domain_range)},"
                f"\ndataset_label={repr(self.dataset_label)},"
                f"\naxes_labels={repr(axes_labels)},"
                f"\nextrapolation={repr(self.extrapolation)},"
                f"\ninterpolator={repr(self.interpolator)},"
                f"\nkeepdims={repr(self.keepdims)})").replace('\n', '\n    ')

    def __getitem__(self, key):
        """Return self[key]."""
        if isinstance(key, tuple):
            # If there are not values for every dimension, the remaining ones
            # are kept
            key += (slice(None),) * (self.dim_domain + 1 - len(key))

            sample_points = [self.sample_points[i][subkey]
                             for i, subkey in enumerate(
                                 key[1:1 + self.dim_domain])]

            return self.copy(data_matrix=self.data_matrix[key],
                             sample_points=sample_points)

        if isinstance(key, numbers.Integral):  # To accept also numpy ints
            key = int(key)
            return self.copy(data_matrix=self.data_matrix[key:key + 1])

        else:
            return self.copy(data_matrix=self.data_matrix[key])

    #####################################################################
    # Numpy methods
    #####################################################################

    def __array_ufunc__(self, ufunc, method, *inputs, **kwargs):

        for i in inputs:
            if isinstance(i, FDataGrid) and not np.all(i.sample_points ==
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

    #####################################################################
    # Pandas ExtensionArray methods
    #####################################################################
    @property
    def dtype(self):
        """The dtype for this extension array, FDataGridDType"""
        return FDataGridDType

    @property
    def nbytes(self) -> int:
        """
        The number of bytes needed to store this object in memory.
        """
        return self.data_matrix.nbytes() + sum(
            p.nbytes() for p in self.sample_points)


class FDataGridDType(pandas.api.extensions.ExtensionDtype):
    """
    DType corresponding to FDataGrid in Pandas
    """
    name = 'functional data (grid)'
    kind = 'O'
    type = FDataGrid
    na_value = None

    @classmethod
    def construct_from_string(cls, string):
        if string == cls.name:
            return cls()
        else:
            raise TypeError("Cannot construct a '{}' from "
                            "'{}'".format(cls, string))

    @classmethod
    def construct_array_type(cls):
        return FDataGrid
