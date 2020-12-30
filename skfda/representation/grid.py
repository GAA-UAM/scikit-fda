"""Discretised functional data module.

This module defines a class for representing functional data as a series of
lists of values, each representing the observation of a function measured in a
list of discretisation points.

"""

import copy
import numbers
import warnings
from typing import Any

import findiff
import numpy as np
import pandas.api.extensions
import scipy.stats.mstats

from .._utils import (
    _check_array_key,
    _domain_range,
    _int_to_real,
    _tuple_of_arrays,
    constants,
)
from . import basis as fdbasis
from ._functional_data import FData
from .interpolation import SplineInterpolation

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
        grid_points (numpy.ndarray): 2 dimension matrix where each row
            contains the points of dicretisation for each axis of data_matrix.
        domain_range (numpy.ndarray): 2 dimension matrix where each row
            contains the bounds of the interval in which the functional data
            is considered to exist for each one of the axies.
        dataset_name (str): name of the dataset.
        argument_names (tuple): tuple containing the names of the different
            arguments.
        coordinate_names (tuple): tuple containing the names of the different
            coordinate functions.
        extrapolation (str or Extrapolation): defines the default type of
            extrapolation. By default None, which does not apply any type of
            extrapolation. See `Extrapolation` for detailled information of the
            types of extrapolation.
        interpolation (GridInterpolation): Defines the type of interpolation
            applied in `evaluate`.

    Examples:
        Representation of a functional data object with 2 samples
        representing a function :math:`f : \mathbb{R}\longmapsto\mathbb{R}`,
        with 3 discretization points.

        >>> data_matrix = [[1, 2, 3], [4, 5, 6]]
        >>> grid_points = [2, 4, 5]
        >>> FDataGrid(data_matrix, grid_points)
        FDataGrid(
            array([[[ 1.],
                    [ 2.],
                    [ 3.]],
        <BLANKLINE>
                   [[ 4.],
                    [ 5.],
                    [ 6.]]]),
            grid_points=(array([ 2., 4., 5.]),),
            domain_range=((2.0, 5.0),),
            ...)

        The number of columns of data_matrix have to be the length of
        grid_points.

        >>> FDataGrid(np.array([1,2,4,5,8]), range(6))
        Traceback (most recent call last):
            ....
        ValueError: Incorrect dimension in data_matrix and grid_points...


        FDataGrid support higher dimensional data both in the domain and image.
        Representation of a functional data object with 2 samples
        representing a function :math:`f : \mathbb{R}\longmapsto\mathbb{R}^2`.

        >>> data_matrix = [[[1, 0.3], [2, 0.4]], [[2, 0.5], [3, 0.6]]]
        >>> grid_points = [2, 4]
        >>> fd = FDataGrid(data_matrix, grid_points)
        >>> fd.dim_domain, fd.dim_codomain
        (1, 2)

        Representation of a functional data object with 2 samples
        representing a function :math:`f : \mathbb{R}^2\longmapsto\mathbb{R}`.

        >>> data_matrix = [[[1, 0.3], [2, 0.4]], [[2, 0.5], [3, 0.6]]]
        >>> grid_points = [[2, 4], [3,6]]
        >>> fd = FDataGrid(data_matrix, grid_points)
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

            s_key = key
            if isinstance(s_key, int):
                s_key = slice(s_key, s_key + 1)

            coordinate_names = np.array(
                self._fdatagrid.coordinate_names)[s_key]

            return self._fdatagrid.copy(
                data_matrix=self._fdatagrid.data_matrix[..., key],
                coordinate_names=coordinate_names)

        def __len__(self):
            """Return the number of coordinates."""
            return self._fdatagrid.dim_codomain

    def __init__(self, data_matrix, grid_points=None,
                 *,
                 sample_points=None,
                 domain_range=None,
                 dataset_label=None,
                 dataset_name=None,
                 argument_names=None,
                 coordinate_names=None,
                 sample_names=None,
                 axes_labels=None, extrapolation=None,
                 interpolation=None):
        """Construct a FDataGrid object.

        Args:
            data_matrix (array_like): a matrix where each row contains the
                values of a functional datum evaluated at the
                points of discretisation.
            grid_points (array_like, optional): an array containing the
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
        if sample_points is not None:
            warnings.warn("Parameter sample_points is deprecated. Use the "
                          "parameter grid_points instead.",
                          DeprecationWarning)
            grid_points = sample_points

        self.data_matrix = _int_to_real(np.atleast_2d(data_matrix))

        if grid_points is None:
            self.grid_points = _tuple_of_arrays(
                [np.linspace(0., 1., self.data_matrix.shape[i]) for i in
                 range(1, self.data_matrix.ndim)])

        else:
            # Check that the dimension of the data matches the grid_points
            # list

            self.grid_points = _tuple_of_arrays(grid_points)

            data_shape = self.data_matrix.shape[1: 1 + self.dim_domain]
            grid_points_shape = [len(i) for i in self.grid_points]

            if not np.array_equal(data_shape, grid_points_shape):
                raise ValueError("Incorrect dimension in data_matrix and "
                                 "grid_points. Data has shape {} and grid "
                                 "points have shape {}"
                                 .format(data_shape, grid_points_shape))

        self._sample_range = np.array(
            [(s[0], s[-1]) for s in self.grid_points])

        if domain_range is None:
            domain_range = self.sample_range
            # Default value for domain_range is a list of tuples with
            # the first and last element of each list of the grid_points.

        self._domain_range = _domain_range(domain_range)

        if len(self._domain_range) != self.dim_domain:
            raise ValueError("Incorrect shape of domain_range.")

        for i in range(self.dim_domain):
            if (self._domain_range[i][0] > self.grid_points[i][0]
                    or self._domain_range[i][-1] < self.grid_points[i]
                    [-1]):
                raise ValueError("Sample points must be within the domain "
                                 "range.")

        # Adjust the data matrix if the dimension of the image is one
        if self.data_matrix.ndim == 1 + self.dim_domain:
            self.data_matrix = self.data_matrix[..., np.newaxis]

        self.interpolation = interpolation

        super().__init__(extrapolation=extrapolation,
                         dataset_label=dataset_label,
                         dataset_name=dataset_name,
                         axes_labels=axes_labels,
                         argument_names=argument_names,
                         coordinate_names=coordinate_names,
                         sample_names=sample_names)

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
    def sample_points(self):
        warnings.warn("Parameter sample_points is deprecated. Use the "
                      "parameter grid_points instead.",
                      DeprecationWarning)
        return self.grid_points

    @property
    def dim_domain(self):
        return len(self.grid_points)

    @property
    def dim_codomain(self):
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
    def interpolation(self):
        """Defines the type of interpolation applied in `evaluate`."""
        return self._interpolation

    @interpolation.setter
    def interpolation(self, new_interpolation):
        """Sets the interpolation of the FDataGrid."""
        if new_interpolation is None:
            new_interpolation = SplineInterpolation()

        self._interpolation = new_interpolation

    def _evaluate(self, eval_points, *, aligned=True):

        return self.interpolation.evaluate(self, eval_points,
                                           aligned=aligned)

    def derivative(self, *, order=1):
        r"""Differentiate a FDataGrid object.

        It is calculated using central finite differences when possible. In
        the extremes, forward and backward finite differences with accuracy
        2 are used.

        Args:
            order (int, optional): Order of the derivative. Defaults to one.

        Examples:
            First order derivative

            >>> fdata = FDataGrid([1,2,4,5,8], range(5))
            >>> fdata.derivative()
            FDataGrid(
                array([[[ 0.5],
                        [ 1.5],
                        [ 1.5],
                        [ 2. ],
                        [ 4. ]]]),
                grid_points=(array([ 0., 1., 2., 3., 4.]),),
                domain_range=((0.0, 4.0),),
                ...)

            Second order derivative

            >>> fdata = FDataGrid([1,2,4,5,8], range(5))
            >>> fdata.derivative(order=2)
            FDataGrid(
                array([[[ 3.],
                        [ 1.],
                        [-1.],
                        [ 2.],
                        [ 5.]]]),
                grid_points=(array([ 0., 1., 2., 3., 4.]),),
                domain_range=((0.0, 4.0),),
                ...)

        """
        order_list = np.atleast_1d(order)
        if order_list.ndim != 1 or len(order_list) != self.dim_domain:
            raise ValueError("The order for each partial should be specified.")

        operator = findiff.FinDiff(*[(1 + i, p, o)
                                     for i, (p, o) in enumerate(
                                         zip(self.grid_points, order_list))])
        data_matrix = operator(self.data_matrix.astype(float))

        if self.dataset_name:
            dataset_name = "{} - {} derivative".format(self.dataset_name,
                                                       order)
        else:
            dataset_name = None

        fdatagrid = self.copy(data_matrix=data_matrix,
                              dataset_name=dataset_name)

        return fdatagrid

    def __check_same_dimensions(self, other):
        if self.data_matrix.shape[1:-1] != other.data_matrix.shape[1:-1]:
            raise ValueError("Error in columns dimensions")
        if not np.array_equal(self.grid_points, other.grid_points):
            raise ValueError("Sample points for both objects must be equal")

    def sum(self, *, axis=None, out=None, keepdims=False, skipna=False,
            min_count=0):
        """Compute the sum of all the samples.

        Returns:
            FDataGrid : A FDataGrid object with just one sample representing
            the sum of all the samples in the original object.

        Examples:

            >>> from skfda import FDataGrid
            >>> data_matrix = [[0.5, 1, 2, .5], [1.5, 1, 4, .5]]
            >>> FDataGrid(data_matrix).sum()
            FDataGrid(
                array([[[ 2.],
                        [ 2.],
                        [ 6.],
                        [ 1.]]]),
                ...)

        """
        super().sum(axis=axis, out=out, keepdims=keepdims, skipna=skipna)

        data = (np.nansum(self.data_matrix, axis=0, keepdims=True) if skipna
                else np.sum(self.data_matrix, axis=0, keepdims=True))

        if min_count > 0:
            valid = ~np.isnan(self.data_matrix)
            n_valid = np.sum(valid, axis=0)
            data[n_valid < min_count] = np.NaN

        return self.copy(data_matrix=data,
                         sample_names=(None,))

    def var(self):
        """Compute the variance of a set of samples in a FDataGrid object.

        Returns:
            FDataGrid: A FDataGrid object with just one sample representing the
            variance of all the samples in the original FDataGrid object.

        """
        return self.copy(data_matrix=[np.var(self.data_matrix, 0)],
                         sample_names=("variance",))

    def cov(self):
        """Compute the covariance.

        Calculates the covariance matrix representing the covariance of the
        functional samples at the observation points.

        Returns:
            numpy.darray: Matrix of covariances.

        """

        if self.dataset_name is not None:
            dataset_name = self.dataset_name + ' - covariance'
        else:
            dataset_name = None

        if self.dim_domain != 1 or self.dim_codomain != 1:
            raise NotImplementedError("Covariance only implemented "
                                      "for univariate functions")

        return self.copy(data_matrix=np.cov(self.data_matrix[..., 0],
                                            rowvar=False)[np.newaxis, ...],
                         grid_points=[self.grid_points[0],
                                      self.grid_points[0]],
                         domain_range=[self.domain_range[0],
                                       self.domain_range[0]],
                         dataset_name=dataset_name,
                         argument_names=self.argument_names * 2,
                         sample_names=("covariance",))

    def gmean(self):
        """Compute the geometric mean of all samples in the FDataGrid object.

        Returns:
            FDataGrid: A FDataGrid object with just one sample representing
            the geometric mean of all the samples in the original
            FDataGrid object.

        """
        return self.copy(data_matrix=[
            scipy.stats.mstats.gmean(self.data_matrix, 0)],
            sample_names=("geometric mean",))

    def equals(self, other):
        """Comparison of FDataGrid objects"""
        if not super().equals(other):
            return False

        if not np.array_equal(self.data_matrix, other.data_matrix):
            return False

        if len(self.grid_points) != len(other.grid_points):
            return False

        for a, b in zip(self.grid_points, other.grid_points):
            if not np.array_equal(a, b):
                return False

        if not np.array_equal(self.domain_range, other.domain_range):
            return False

        if self.interpolation != other.interpolation:
            return False

        return True

    def __eq__(self, other):
        """Elementwise equality of FDataGrid"""

        if not isinstance(self, type(other)) or self.dtype != other.dtype:
            if other is pandas.NA:
                return self.isna()
            if pandas.api.types.is_list_like(other) and not isinstance(
                other, (pandas.Series, pandas.Index, pandas.DataFrame),
            ):
                return np.concatenate([x == y for x, y in zip(self, other)])
            else:
                return NotImplemented

        if len(self) != len(other) and len(self) != 1 and len(other) != 1:
            raise ValueError(f"Different lengths: "
                             f"len(self)={len(self)} and "
                             f"len(other)={len(other)}")

        return np.all(self.data_matrix == other.data_matrix,
                      axis=tuple(range(1, self.data_matrix.ndim)))

    def _get_op_matrix(self, other):
        if isinstance(other, numbers.Number):
            return other
        elif isinstance(other, np.ndarray):

            if other.shape == () or other.shape == (1,):
                return other
            elif other.shape == (self.n_samples,):
                other_index = ((slice(None),) + (np.newaxis,) *
                               (self.data_matrix.ndim - 1))

                return other[other_index]
            else:
                return None

        elif isinstance(other, FDataGrid):
            self.__check_same_dimensions(other)
            return other.data_matrix
        else:
            return None

    def __add__(self, other):
        """Addition for FDataGrid object.

        It supports other FDataGrid objects, numpy.ndarray and numbers.

        """

        data_matrix = self._get_op_matrix(other)
        if data_matrix is None:
            return NotImplemented

        return self._copy_op(other, data_matrix=self.data_matrix + data_matrix)

    def __radd__(self, other):
        """Addition for FDataGrid object.

        It supports other FDataGrid objects, numpy.ndarray and numbers.

        """

        return self.__add__(other)

    def __sub__(self, other):
        """Subtraction for FDataGrid object.

        It supports other FDataGrid objects, numpy.ndarray and numbers.

        """
        data_matrix = self._get_op_matrix(other)
        if data_matrix is None:
            return NotImplemented

        return self._copy_op(other, data_matrix=self.data_matrix - data_matrix)

    def __rsub__(self, other):
        """Right Subtraction for FDataGrid object.

        It supports other FDataGrid objects, numpy.ndarray and numbers.

        """
        data_matrix = self._get_op_matrix(other)
        if data_matrix is None:
            return NotImplemented

        return self.copy(data_matrix=data_matrix - self.data_matrix)

    def __mul__(self, other):
        """Multiplication for FDataGrid object.

        It supports other FDataGrid objects, numpy.ndarray and numbers.

        """
        data_matrix = self._get_op_matrix(other)
        if data_matrix is None:
            return NotImplemented

        return self._copy_op(other, data_matrix=self.data_matrix * data_matrix)

    def __rmul__(self, other):
        """Multiplication for FDataGrid object.

        It supports other FDataGrid objects, numpy.ndarray and numbers.

        """
        return self.__mul__(other)

    def __truediv__(self, other):
        """Division for FDataGrid object.

        It supports other FDataGrid objects, numpy.ndarray and numbers.

        """
        data_matrix = self._get_op_matrix(other)
        if data_matrix is None:
            return NotImplemented

        return self._copy_op(other, data_matrix=self.data_matrix / data_matrix)

    def __rtruediv__(self, other):
        """Division for FDataGrid object.

        It supports other FDataGrid objects, numpy.ndarray and numbers.

        """
        data_matrix = self._get_op_matrix(other)
        if data_matrix is None:
            return NotImplemented

        return self._copy_op(other, data_matrix=data_matrix / self.data_matrix)

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
                array([[[ 1.],
                        [ 2.],
                        [ 4.],
                        [ 5.],
                        [ 8.]],
            <BLANKLINE>
                       [[ 3.],
                        [ 4.],
                        [ 7.],
                        [ 9.],
                        [ 2.]]]),
                grid_points=(array([ 0., 1., 2., 3., 4.]),),
                domain_range=((0.0, 4.0),),
                ...)

        """
        # Checks
        if not as_coordinates:
            for other in others:
                self.__check_same_dimensions(other)

        elif not all([np.array_equal(self.grid_points, other.grid_points)
                      for other in others]):
            raise ValueError("All the FDataGrids must be sampled in the  same "
                             "sample points.")

        elif any([self.n_samples != other.n_samples for other in others]):

            raise ValueError(f"All the FDataGrids must contain the same "
                             f"number of samples {self.n_samples} to "
                             f"concatenate as a new coordinate.")

        data = [self.data_matrix] + [other.data_matrix for other in others]

        if as_coordinates:

            coordinate_names = [fd.coordinate_names for fd in [self, *others]]

            return self.copy(data_matrix=np.concatenate(data, axis=-1),
                             coordinate_names=sum(coordinate_names, ()))

        else:

            sample_names = [fd.sample_names for fd in [self, *others]]

            return self.copy(data_matrix=np.concatenate(data, axis=0),
                             sample_names=sum(sample_names, ()))

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
            >>> x = np.sin(2 * np.pi * t) + np.cos(2 * np.pi * t) + 2
            >>> x
            array([ 3.,  3.,  1.,  1.,  3.])

            >>> fd = FDataGrid(x, t)
            >>> basis = skfda.representation.basis.Fourier(n_basis=3)
            >>> fd_b = fd.to_basis(basis)
            >>> fd_b.coefficients.round(2)
            array([[ 2.  , 0.71, 0.71]])

        """
        from ..preprocessing.smoothing import BasisSmoother

        if self.dim_domain != basis.dim_domain:
            raise ValueError(f"The domain of the function has "
                             f"dimension {self.dim_domain} "
                             f"but the domain of the basis has "
                             f"dimension {basis.dim_domain}")
        elif self.dim_codomain != basis.dim_codomain:
            raise ValueError(f"The codomain of the function has "
                             f"dimension {self.dim_codomain} "
                             f"but the codomain of the basis has "
                             f"dimension {basis.dim_codomain}")

        # Readjust the domain range if there was not an explicit one
        if basis._domain_range is None:
            basis = basis.copy(domain_range=self.domain_range)

        smoother = BasisSmoother(
            basis=basis,
            **kwargs,
            return_basis=True)

        return smoother.fit_transform(self)

    def to_grid(self, grid_points=None, *, sample_points=None):

        if sample_points is not None:
            warnings.warn("Parameter sample_points is deprecated. Use the "
                          "parameter grid_points instead.",
                          DeprecationWarning)
            grid_points = sample_points

        if grid_points is None:
            grid_points = self.grid_points

        return self.copy(data_matrix=self.evaluate(grid_points, grid=True),
                         grid_points=grid_points)

    def copy(self, *,
             deep=False,  # For Pandas compatibility
             data_matrix=None,
             grid_points=None,
             sample_points=None,
             domain_range=None,
             dataset_name=None,
             argument_names=None,
             coordinate_names=None,
             sample_names=None,
             extrapolation=None,
             interpolation=None):
        """Returns a copy of the FDataGrid.

        If an argument is provided the corresponding attribute in the new copy
        is updated.

        """

        if sample_points is not None:
            warnings.warn("Parameter sample_points is deprecated. Use the "
                          "parameter grid_points instead.",
                          DeprecationWarning)
            grid_points = sample_points

        if data_matrix is None:
            # The data matrix won't be writeable
            data_matrix = self.data_matrix

        if grid_points is None:
            # Sample points won`t be writeable
            grid_points = self.grid_points

        if domain_range is None:
            domain_range = copy.deepcopy(self.domain_range)

        if dataset_name is None:
            dataset_name = self.dataset_name

        if argument_names is None:
            # Tuple, immutable
            argument_names = self.argument_names

        if coordinate_names is None:
            # Tuple, immutable
            coordinate_names = self.coordinate_names

        if sample_names is None:
            # Tuple, immutable
            sample_names = self.sample_names

        if extrapolation is None:
            extrapolation = self.extrapolation

        if interpolation is None:
            interpolation = self.interpolation

        return FDataGrid(data_matrix, grid_points=grid_points,
                         domain_range=domain_range,
                         dataset_name=dataset_name,
                         argument_names=argument_names,
                         coordinate_names=coordinate_names,
                         sample_names=sample_names,
                         extrapolation=extrapolation,
                         interpolation=interpolation)

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
                current grid_points are used to unificate the domain of the
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

            grid_points = self.grid_points + shifts
            domain_range = self.domain_range + shifts

            return self.copy(grid_points=grid_points,
                             domain_range=domain_range)
        if shifts.shape[0] != self.n_samples:
            raise ValueError(f"shifts vector ({shifts.shape[0]}) must have the"
                             f" same length than the number of samples "
                             f"({self.n_samples})")

        if eval_points is None:
            eval_points = self.grid_points
        else:
            eval_points = np.atleast_2d(eval_points)

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
                                    aligned=False,
                                    grid=True)

        return self.copy(data_matrix=data_matrix, grid_points=eval_points,
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
                    eval_points = fd.grid_points[0]
                except AttributeError:
                    eval_points = np.linspace(*fd.domain_range[0],
                                              constants.N_POINTS_COARSE_MESH)

            eval_points_transformation = fd(eval_points)
            data_matrix = self(eval_points_transformation,
                               aligned=False)
        else:
            if eval_points is None:
                eval_points = fd.grid_points

            grid_transformation = fd(eval_points, grid=True)

            lengths = [len(ax) for ax in eval_points]

            eval_points_transformation = np.empty((self.n_samples,
                                                   np.prod(lengths),
                                                   self.dim_domain))

            for i in range(self.n_samples):
                eval_points_transformation[i] = np.array(
                    list(map(np.ravel, grid_transformation[i].T))
                ).T

            data_matrix = self(eval_points_transformation,
                               aligned=False)

        return self.copy(data_matrix=data_matrix,
                         grid_points=eval_points,
                         domain_range=fd.domain_range,
                         argument_names=fd.argument_names)

    def __str__(self):
        """Return str(self)."""
        return ('Data set:    ' + str(self.data_matrix)
                + '\ngrid_points:    ' + str(self.grid_points)
                + '\ntime range:    ' + str(self.domain_range))

    def __repr__(self):
        """Return repr(self)."""

        return (f"FDataGrid("
                f"\n{repr(self.data_matrix)},"
                f"\ngrid_points={repr(self.grid_points)},"
                f"\ndomain_range={repr(self.domain_range)},"
                f"\ndataset_name={repr(self.dataset_name)},"
                f"\nargument_names={repr(self.argument_names)},"
                f"\ncoordinate_names={repr(self.coordinate_names)},"
                f"\nextrapolation={repr(self.extrapolation)},"
                f"\ninterpolation={repr(self.interpolation)})").replace(
                    '\n', '\n    ')

    def __getitem__(self, key):
        """Return self[key]."""

        key = _check_array_key(self.data_matrix, key)

        return self.copy(data_matrix=self.data_matrix[key],
                         sample_names=np.array(self.sample_names)[key])

    #####################################################################
    # Numpy methods
    #####################################################################

    def __array_ufunc__(self, ufunc, method, *inputs, **kwargs):

        for i in inputs:
            if isinstance(i, FDataGrid) and not np.array_equal(
                    i.grid_points, self.grid_points):
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
        return FDataGridDType(
            grid_points=self.grid_points,
            domain_range=self.domain_range,
            dim_codomain=self.dim_codomain)

    @property
    def nbytes(self) -> int:
        """
        The number of bytes needed to store this object in memory.
        """
        return self.data_matrix.nbytes + sum(
            p.nbytes for p in self.grid_points)

    def isna(self):
        """
        A 1-D array indicating if each value is missing.

        Returns:
            na_values (np.ndarray): Positions of NA.
        """
        return np.all(np.isnan(self.data_matrix),
                      axis=tuple(range(1, self.data_matrix.ndim)))


class FDataGridDType(pandas.api.extensions.ExtensionDtype):
    """
    DType corresponding to FDataGrid in Pandas
    """
    name = 'FDataGrid'
    kind = 'O'
    type = FDataGrid
    na_value = pandas.NA

    def __init__(self, grid_points, dim_codomain, domain_range=None) -> None:
        grid_points = _tuple_of_arrays(grid_points)

        self.grid_points = tuple(tuple(s) for s in grid_points)

        if domain_range is None:
            domain_range = np.array(
                [(s[0], s[-1]) for s in self.grid_points])

        self.domain_range = _domain_range(domain_range)
        self.dim_codomain = dim_codomain

    @classmethod
    def construct_array_type(cls):
        return FDataGrid

    def _na_repr(self) -> FDataGrid:

        shape = ((1,)
                 + tuple(len(s) for s in self.grid_points)
                 + (self.dim_codomain,))

        data_matrix = np.full(shape=shape, fill_value=np.NaN)

        return FDataGrid(
            grid_points=self.grid_points,
            domain_range=self.domain_range,
            data_matrix=data_matrix)

    def __eq__(self, other: Any) -> bool:
        """
        Rules for equality (similar to categorical):
        1) Any FData is equal to the string 'category'
        2) Any FData is equal to itself
        3) Otherwise, they are equal if the arguments are equal.
        6) Any other comparison returns False
        """
        if isinstance(other, str):
            return other == self.name
        elif other is self:
            return True
        else:
            return (isinstance(other, FDataGridDType)
                    and self.dim_codomain == other.dim_codomain
                    and self.domain_range == other.domain_range
                    and self.grid_points == other.grid_points)

    def __hash__(self) -> int:
        return hash((self.grid_points,
                     self.domain_range, self.dim_codomain))
