"""Module for functional data manipulation.

Defines the abstract class that should be implemented by the funtional data
objects of the package and contains some commons methods.
"""

from abc import ABC, abstractmethod
from enum import Enum
import numbers

import numpy

class Extrapolation(Enum):
    r"""Enum with extrapolation types. Defines the extrapolation mode for
        elements outside the domain range.
    """

    periodic = "periodic" #: Extends the domain range periodically.
    const = "const" #: Uses the boundary value.
    exception = "exception" #: Raise an exception

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

# This class could exteds in a future the numpy class to work with their
# functions.

class FData(ABC):
    """Defines the structure of a functional data object.

    Attributes:
        nsamples (int): Number of samples.
        ndim_domain (int): Dimension of the domain.
        ndim_image (int): Dimension of the image.
        extrapolation (Extrapolation): Default extrapolation mode.
        dataset_label (str): name of the dataset.
        axes_labels (list): list containing the labels of the different
            axis. The first element is the x label, the second the y label
            and so on.
        keepdims (bool): Default value of argument keepdims in
            :func:`evaluate".

    """

    @property
    @abstractmethod
    def nsamples(self):
        """Return the number of samples.

        Returns:
            int: Number of samples of the FDataGrid object. Also the number of
                rows of the data_matrix.

        """
        pass

    @property
    @abstractmethod
    def ndim_domain(self):
        """Return number of dimensions of the domain.

        Returns:
            int: Number of dimensions of the domain.

        """
        pass

    @property
    @abstractmethod
    def ndim_image(self):
        """Return number of dimensions of the image.

        Returns:
            int: Number of dimensions of the image.

        """
        pass

    def ndim_codomain(self):
        """Return number of dimensions of the image.

        Returns:
            int: Number of dimensions of the image.

        """
        return self.ndim_image

    @property
    @abstractmethod
    def extrapolation(self):
        """Return the default type of extrapolation of the object

        Returns:
            Extrapolation: Type of extrapolation

        """
        pass

    @property
    @abstractmethod
    def domain_range(self):
        """Return the domain range of the object

        Returns:
            List of tuples with the ranges for each domain dimension.
        """
        pass

    @abstractmethod
    def evaluate(self, eval_points, *, derivative=0, extrapolation=None,
                 grid=False, keepdims=None):
        """Evaluate the object or its derivatives at a list of values or a grid.

        Args:
            eval_points (array_like): List of points where the functions are
                evaluated. If a matrix of shape nsample x eval_points is given
                each sample is evaluated at the values in the corresponding row
                in eval_points.
            derivative (int, optional): Order of the derivative. Defaults to 0.
            extrapolation (str or Extrapolation, optional): Controls the
                extrapolation mode for elements outside the domain range. By
                default it is used the mode defined during the instance of the
                object.
            grid (bool, optional): Whether to evaluate the results on a grid
                spanned by the input arrays, or at points specified by the input
                arrays. If true the eval_points should be a list of size
                ndim_domain with the corresponding times for each axis. The
                return matrix has shape nsamples x len(t1) x len(t2) x ... x
                len(t_ndim_domain) x ndim_image. If the domain dimension is 1
                the parameter has no efect. Defaults to False.
            keepdims (bool, optional): If the image dimension is equal to 1 and
                keepdims is True the return matrix has shape
                nsamples x eval_points x 1 else nsamples x eval_points.
                By default is used the value given during the instance of the
                object.

        Returns:
            (numpy.darray): Matrix whose rows are the values of the each
            function at the values specified in eval_points.

        """
        pass

    def __call__(self, eval_points, *, derivative=0, extrapolation=None,
                 grid=False, keepdims=False):
        """Evaluate the object or its derivatives at a list of values or a grid.
        This method is a wrapper of :meth:`evaluate`.

        Args:
            eval_points (array_like): List of points where the functions are
                evaluated. If a matrix of shape nsample x eval_points is given
                each sample is evaluated at the values in the corresponding row
                in eval_points.
            derivative (int, optional): Order of the derivative. Defaults to 0.
            extrapolation (str or Extrapolation, optional): Controls the
                extrapolation mode for elements outside the domain range. By
                default it is used the mode defined during the instance of the
                object.
            grid (bool, optional): Whether to evaluate the results on a grid
                spanned by the input arrays, or at points specified by the input
                arrays. If true the eval_points should be a list of size
                ndim_domain with the corresponding times for each axis. The
                return matrix has shape nsamples x len(t1) x len(t2) x ... x
                len(t_ndim_domain) x ndim_image. If the domain dimension is 1
                the parameter has no efect. Defaults to False.
            keepdims (bool, optional): If the image dimension is equal to 1 and
                keepdims is True the return matrix has shape
                nsamples x eval_points x 1 else nsamples x eval_points.
                By default is used the value given during the instance of the
                object.

        Returns:
            (numpy.ndarray): Matrix whose rows are the values of the each
            function at the values specified in eval_points.

        """

        return self.evaluate(eval_points, derivative=derivative,
                             extrapolation=extrapolation, grid=grid,
                             keepdims=keepdims)

    def _extrapolate_time(self, eval_points, extrapolation=None,
                          in_place=False):
        """Modifies the eval_points during the evaluation to apply
        extrapolation.

            Args:
                eval_points (ndarray): List or matrix to apply extrapolation.
                extrapolation (str or Extrapolation): Type of extrapolation to
                    apply. See :class:`Extrapolation`.
                in_place (bool, optional): Select if modificate in place the
                    eval_points or return a copy. Default to false.

            Returns:
                (numpy.ndarray): Arrays with the corresponding eval_points
                modificated.
        """

        if extrapolation is None:
            extrapolation = self.extrapolation
        else:
            extrapolation = Extrapolation(extrapolation)

        # Do nothing
        if extrapolation is None:
            return eval_points

        domain_range = self.domain_range[0]

        # Creates a copy of the object or uses the given array
        out = eval_points if not in_place else eval_points.copy()

        if extrapolation is Extrapolation.periodic:

            numpy.subtract(out, domain_range[0], out)
            numpy.mod(out, domain_range[1] - domain_range[0], out)
            numpy.add(out, domain_range[0], out)

        # Case boundary value
        elif extrapolation is Extrapolation.const:
            out[out <= domain_range[0]] = domain_range[0]
            out[out >= domain_range[1]] = domain_range[1]

        # Case raise exception
        elif extrapolation is Extrapolation.exception:
            if (numpy.any(out < domain_range[0]) or
                numpy.any(out > domain_range[1])):

                raise ValueError("Attempt to evaluate a value outside the "
                                 "domain range.")

        return out

    @abstractmethod
    def derivative(self, order=1):
        r"""Differentiate a FDataGrid object.


        Args:
            order (int, optional): Order of the derivative. Defaults to one.
        """
        pass

    @abstractmethod
    def shift(self, shifts, *, restrict_domain=False, extrapolation=None,
              discretization_points=None, **kwargs):
        """Perform a shift of the curves.

        Args:
            shifts (array_like or numeric): List with the shift corresponding
                for each sample or numeric with the shift to apply to all
                samples.
            restrict_domain (bool, optional): If True restricts the domain to
                avoid evaluate points outside the domain using extrapolation.
                Defaults uses extrapolation.
            extrapolation (str or Extrapolation, optional): Controls the
                extrapolation mode for elements outside the domain range.
                By default uses the method defined in fd. See extrapolation to
                more information.
            discretization_points (array_like, optional): Set of points where
                the functions are evaluated to obtain the discrete
                representation of the object to operate. If an empty list is
                passed it calls numpy.linspace with bounds equal to the ones
                defined in fd.domain_range and the number of points the maximum
                between 201 and 10 times the number of basis plus 1.

        Returns:
            :obj:`FDataBasis` with the registered functional data.
        """
        pass

    @abstractmethod
    def plot(self, ax=None, derivative=0, **kwargs):
        """Plot the FData object.

        Args:
            ax (axis object, optional): axis over with the graphs are plotted.
                Defaults to matplotlib current axis.
            **kwargs: keyword arguments to be passed to the
                matplotlib.pyplot.plot function.

        Returns:
            List of lines that were added to the plot.

        """
        pass

    def _set_labels(self, ax):
        """Set labels if any.

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


    @abstractmethod
    def copy(self, **kwargs):
        pass

    @abstractmethod
    def mean(self):
        """Compute the mean of all the samples.

        Returns:
            FData : A FData object with just one sample representing
            the mean of all the samples in the original object.

        """
        pass

    @abstractmethod
    def gmean(self):
        """Compute the geometric mean of all samples in the FDataGrid object.

        Returns:
            FData: A FData object with just one sample representing
            the geometric mean of all the samples in the original
            FData object.

        """
        pass

    @abstractmethod
    def var(self):
        """Compute the variance of a set of samples in a FDataGrid object.

        Returns:
            FDataGrid: A FDataGrid object with just one sample representing the
            variance of all the samples in the original FDataGrid object.

        """
        pass

    @abstractmethod
    def cov(self):
        """Compute the covariance.

        Calculates the covariance matrix representing the covariance of the
        functional samples at the observation points.

        Returns:
            numpy.darray: Matrix of covariances.

        """
        pass

    @abstractmethod
    def round(self, decimals=0):
        """Evenly round to the given number of decimals.

        Args:
            decimals (int, optional): Number of decimal places to round to.
                If decimals is negative, it specifies the number of
                positions to the left of the decimal point. Defaults to 0.

        Returns:
            :obj:FData: Returns a FData object where all elements
            in its data_matrix are rounded .The real and
            imaginary parts of complex numbers are rounded separately.

        """
        pass

    @abstractmethod
    def to_grid(self, eval_points=None):
        """Return the discrete representation of the object.

        Args:
            eval_points (array_like, optional): Set of points where the
                functions are evaluated.

        Returns:
              FDataGrid: Discrete representation of the functional data
              object.
        """

        pass

    @abstractmethod
    def to_basis(self, basis, eval_points=None, **kwargs):
        """Return the basis representation of the object.

        Args:
            basis(Basis): basis object in which the functional data are
                going to be represented.
            **kwargs: keyword arguments to be passed to
                FDataBasis.from_data().

        Returns:
            FDataBasis: Basis representation of the funtional data
            object.
        """

        pass

    @abstractmethod
    def concatenate(self, other):
        """Join samples from a similar FData object.

        Joins samples from another FData object if it has the same
        dimensions and has compatible representations.

        Args:
            other (:class:`FData`): another FData object.

        Returns:
            :class:`FData`: FData object with the samples from the two
            original objects.
        """

        pass

    @abstractmethod
    def __repr__(self):
        """Return repr(self)."""

        pass

    @abstractmethod
    def __str__(self):
        """Return str(self)."""

        pass

    @abstractmethod
    def __getitem__(self, key):
        """Return self[key]."""

        pass

    @abstractmethod
    def __add__(self, other):
        """Addition for FData object."""

        pass

    @abstractmethod
    def __radd__(self, other):
        """Addition for FData object."""

        pass

    @abstractmethod
    def __sub__(self, other):
        """Subtraction for FData object."""

        pass

    @abstractmethod
    def __rsub__(self, other):
        """Right subtraction for FData object."""

        pass

    @abstractmethod
    def __mul__(self, other):
        """Multiplication for FData object."""

        pass

    @abstractmethod
    def __rmul__(self, other):
        """Multiplication for FData object."""

        pass

    @abstractmethod
    def __truediv__(self, other):
        """Division for FData object."""

        pass
