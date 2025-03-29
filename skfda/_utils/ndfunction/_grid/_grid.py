"""Implementation of functions discretized in a grid of values."""
from __future__ import annotations

from functools import cached_property
import math
from collections.abc import Sequence
from typing import TYPE_CHECKING, Mapping, TypeVar, overload

import numpy as np
import pandas as pd
from typing_extensions import override

from .._array_api import Array, DType, Shape, array_namespace, ArrayNamespace
from .._ndfunction import NDFunction
from ..interpolation import SplineInterpolation
from ..utils.validation import check_grid_points, check_region
from .._region import AxisAlignedBox

"""Discretized functional data module.

This module defines a class for representing functional data as a series of
lists of values, each representing the observation of a function measured in a
list of discretization points.

"""

if TYPE_CHECKING:

    from ....typing._base import LabelTupleLike
    from .._region import Region
    from ..evaluator import Evaluator
    from ..extrapolation import AcceptedExtrapolation, ExtrapolationLike
    from ..typing import (
        GridPoints,
        GridPointsLike,
        InputNamesLike,
        OutputNamesLike,
    )
    from .basis import Basis, FDataBasis

A = TypeVar("A", bound=Array[Shape, DType])
T = TypeVar("T", bound="GridDiscretizedFunction")


def _infer_shapes(
    shape: tuple[int, ...] | None = None,
    input_shape: tuple[int, ...] | None = None,
    output_shape: tuple[int, ...] | None = None,
    *,
    grid_values_shape: tuple[int, ...],
    grid_points_shape: tuple[int, ...] | None = None,
    domain_shape: tuple[int, ...] | None = None,
) -> tuple[tuple[int, ...], tuple[int, ...], tuple[int, ...]]:
    """
    Infer the correct shapes from the supplied parameters.

    Infers the values of ``shape``, ``input_shape`` and ``output_shape`` from
    the remaining arguments.

    Args:
        shape: The shape of the array of functions, if present.
        input_shape: The input shape of the functions, if present.
        output_shape: The output shape of the functions, if present.
        grid_values_shape: Shape of the grid values.
        grid_points_shape: Shape of the grid points, if passed.
        domain_shape: Shape of the domain, if passed.

    Returns:
        Inferred ``shape``, ``input_shape`` and ``output_shape``, in that
        order.

    Examples:
        If everything is present, it should be returned as is:

        >>> _infer_shapes(
        ...     shape=(3, 4),
        ...     input_shape=(2, 2),
        ...     output_shape=(5, 6, 7),
        ...     grid_values_shape=(3, 4, 10, 11, 12, 13, 5, 6, 7),
        ... )
        ((3, 4), (2, 2), (5, 6, 7))

        Infer shape from the other two and the grid values:

        >>> _infer_shapes(
        ...     input_shape=(2, 2),
        ...     output_shape=(5, 6, 7),
        ...     grid_values_shape=(3, 4, 10, 11, 12, 13, 5, 6, 7),
        ... )
        ((3, 4), (2, 2), (5, 6, 7))

        Infer output shape from the other two and the grid values:

        >>> _infer_shapes(
        ...     shape=(3, 4),
        ...     input_shape=(2, 2),
        ...     grid_values_shape=(3, 4, 10, 11, 12, 13, 5, 6, 7),
        ... )
        ((3, 4), (2, 2), (5, 6, 7))

        Infer (raveled) input shape from the other two and the grid values:

        >>> _infer_shapes(
        ...     shape=(3, 4),
        ...     output_shape=(5, 6, 7),
        ...     grid_values_shape=(3, 4, 10, 11, 12, 13, 5, 6, 7),
        ... )
        ((3, 4), (4,), (5, 6, 7))

        Infer input shape from the grid points:

        >>> _infer_shapes(
        ...     shape=(3, 4),
        ...     output_shape=(5, 6, 7),
        ...     grid_values_shape=(3, 4, 10, 11, 12, 13, 5, 6, 7),
        ...     grid_points_shape=(2, 2),
        ... )
        ((3, 4), (2, 2), (5, 6, 7))

        Infer input shape from the domain:

        >>> _infer_shapes(
        ...     shape=(3, 4),
        ...     output_shape=(5, 6, 7),
        ...     grid_values_shape=(3, 4, 10, 11, 12, 13, 5, 6, 7),
        ...     domain_shape=(2, 2),
        ... )
        ((3, 4), (2, 2), (5, 6, 7))

        Infer one scalar-valued function when not enough info is given:

        >>> _infer_shapes(
        ...     grid_values_shape=(3, 4, 10, 11, 12, 13, 5, 6, 7),
        ... )
        ((), (9,), ())

    """
    # First we will try to infer the shapes (shape, input shape and
    # output shape, given all available information).

    # Try to infer input shape, if None.
    if input_shape is None:

        # Infer from grid points.
        if grid_points_shape is not None:
            input_shape = grid_points_shape

        # Infer from domain.
        elif domain_shape is not None:
            input_shape = domain_shape

        # Infer from remaining shapes, falling back to default values
        # if there is no other choice.
        else:
            # We do not have enough info, unless the other two are defined.
            # We will assume by default one function, scalar response, and
            # vector input.
            if shape is None:
                shape = ()

            if output_shape is None:
                output_shape = ()

            input_shape = (
                len(grid_values_shape)  # Assume vector (raveled) shape.
                - len(shape)  # If shape is present, it matches the left part.
                - len(output_shape),  # If output_shape is present, it matches
                                      # the right part.
            )

    # We have the input shape. We can deduce the shape from the output
    # shape or vice versa.
    match (shape, output_shape):
        case (None, None):
            # Not enough info. Let's assume both are one and roll with
            # it.
            shape = ()
            output_shape = ()
        case (None, _):
            # Last dimensions of grid values must be raveled input and
            # output shape. So, shape must be the remaining first ones.
            last_index = - math.prod(input_shape) - len(output_shape)
            shape = grid_values_shape[:last_index]
        case (_, None):
            # First dimensions of grid values must be shape and raveled
            # input. So, output shape must be the remaining last ones.
            first_index = len(shape) + math.prod(input_shape)
            output_shape = grid_values_shape[first_index:]

        case _:
            # Nothing left to infer.
            pass

    return shape, input_shape, output_shape


class GridDiscretizedFunction(NDFunction[A]):  # noqa: WPS214
    r"""
    Array of functions discretized on a grid.

    Attributes:
        grid_values: A tensor containing the values of the functions at the
            intersections of the grid. Its shape has three parts:
                - The shape of the array of functions (:attr:`shape`).
                - A shape (:math:`M_0`, ..., :math:`M_P`), where :math:`P` is
                  the size of the raveled input, and :math:`M_p` the number of
                  grid points in the :math:`p`-th coordinate of the raveled
                  input.
                - The shape of the output (:attr:`output_shape`).
        grid_points: Object array with the same shape as the input
            (:attr:`ìnput_shape`). The value at each location is an array with
            the ordered grid points at that location.
        domain_range: 2 dimension matrix where each row
            contains the bounds of the interval in which the functional data
            is considered to exist for each one of the axes.
        dataset_name: name of the dataset.
        argument_names: tuple containing the names of the different
            arguments.
        coordinate_names: tuple containing the names of the different
            coordinate functions.
        extrapolation: defines the default type of
            extrapolation. By default None, which does not apply any type of
            extrapolation. See `Extrapolation` for detailled information of the
            types of extrapolation.
        interpolation: Defines the type of interpolation
            applied in `evaluate`.

    Examples:
        Representation of a functional data object with 2 samples
        representing a function :math:`f : \mathbb{R}\longmapsto\mathbb{R}`,
        with 3 discretization points.

        >>> from skfda import FDataGrid
        >>> import numpy as np
        >>> data_matrix = [[1, 2, 3], [4, 5, 6]]
        >>> grid_points = np.array([2, 4, 5])
        >>> FDataGrid(data_matrix, grid_points)
        FDataGrid(
            array([[[ 1.],
                    [ 2.],
                    [ 3.]],
        <BLANKLINE>
                   [[ 4.],
                    [ 5.],
                    [ 6.]]]),
            grid_points=array([array([ 2, 4, 5])], dtype=object),
            domain_range=((2.0, 5.0),),
            ...)

        The number of columns of data_matrix have to be the length of
        grid_points.

        >>> FDataGrid(np.array([1,2,4,5,8]), np.arange(6))
        Traceback (most recent call last):
            ....
        ValueError: The number of grid points for each dimension...


        FDataGrid support higher dimensional data both in the domain and image.
        Representation of a functional data object with 2 samples
        representing a function :math:`f : \mathbb{R}\longmapsto\mathbb{R}^2`.

        >>> data_matrix = [[[1, 0.3], [2, 0.4]], [[2, 0.5], [3, 0.6]]]
        >>> grid_points = np.array([2, 4])
        >>> fd = FDataGrid(data_matrix, grid_points)
        >>> fd.dim_domain, fd.dim_codomain
        (1, 2)

        Representation of a functional data object with 2 samples
        representing a function :math:`f : \mathbb{R}^2\longmapsto\mathbb{R}`.

        >>> data_matrix = [[[1, 0.3], [2, 0.4]], [[2, 0.5], [3, 0.6]]]
        >>> grid_points = [np.array([2, 4]), np.array([3,6])]
        >>> fd = FDataGrid(data_matrix, grid_points)
        >>> fd.dim_domain, fd.dim_codomain
        (2, 1)

    """

    def __init__(  # noqa: WPS211
        self,
        grid_values: A,
        grid_points: GridPointsLike[A] | None = None,
        *,
        domain: Region[A] | None = None,
        input_names: InputNamesLike = None,
        output_names: OutputNamesLike = None,
        extrapolation: ExtrapolationLike[A] | None = None,
        interpolation: Evaluator[A] | None = None,
        shape: tuple[int, ...] | None = None,
        input_shape: tuple[int, ...] | None = None,
        output_shape: tuple[int, ...] | None = None,
    ):
        self.grid_values = grid_values

        if domain is not None:
            domain = check_region(domain)

        if grid_points is not None:
            grid_points = check_grid_points(grid_points)

        shape, input_shape, output_shape = _infer_shapes(
            shape=shape,
            input_shape=input_shape,
            output_shape=output_shape,
            grid_values_shape=grid_values.shape,
            grid_points_shape=(
                None
                if grid_points is None
                else grid_points.shape
            ),
            domain_shape=(
                None
                if domain is None
                else domain.bounding_box[0].shape
            ),
        )

        # Check that grid_values start with the given shape.
        middle_part_index = len(shape)
        if grid_values.shape[:middle_part_index] != shape:
            msg = (
                f"The grid values shape ({grid_values.shape}) is not "
                f"compatible with the given shape ({shape}). All dimensions "
                f"starting from the left should match exactly."
            )
            raise ValueError(msg)

        # Check that grid_values end with the given output_shape.
        output_shape_index = len(grid_values.shape) - len(output_shape)
        if grid_values.shape[output_shape_index:] != output_shape:
            msg = (
                f"The grid values shape ({grid_values.shape}) is not "
                f"compatible with the given output shape ({output_shape}). "
                f"All dimensions starting from the right should match exactly."
            )
            raise ValueError(msg)

        # Check that grid_values have a shape with the raveled input shape
        # length in the middle.
        middle_shape = grid_values.shape[middle_part_index:output_shape_index]
        raveled_input_shape_len = math.prod(input_shape)
        if len(middle_shape) != raveled_input_shape_len:
            msg = (
                f"The grid values shape ({grid_values.shape}) is not "
                f"compatible with the given input shape ({input_shape}). "
                f"The middle part of the grid values shape ({middle_shape}) "
                f"should have the same length as the raveled input "
                f"({raveled_input_shape_len})."
            )
            raise ValueError(msg)

        if domain is None:
            if grid_points is None:
                # Set the domain as the unit square.
                lower = self.array_backend.zeros(shape=input_shape)
                upper = self.array_backend.ones(shape=input_shape)
            else:
                # Set the domain as the bounding box of the grid points.
                lower = np.vectorize(self.array_backend.min)(grid_points)
                upper = np.vectorize(self.array_backend.max)(grid_points)

            domain = AxisAlignedBox(lower=lower, upper=upper)
        else:
            # Check that the provided domain shape is correct.
            lower, upper = domain.bounding_box
            if lower.shape != input_shape:
                msg = (
                    f"The shape of the domain ({lower.shape}) does not match "
                    f"the shape of the input ({input_shape})."
                )
                raise ValueError(msg)

        if grid_points is None:
            # Create equispaced gridpoints in the domain, with the number of
            # points according to the shape of the grid values.
            grid_points_num = self.array_backend.reshape(
                self.array_backend.asarray(middle_shape),
                shape=input_shape,
            )
            lower, upper = domain.bounding_box

            grid_points = np.vectorize(
                self.array_backend.linspace,
                otypes=(np.object_,),
            )(lower, upper, grid_points_num)
        else:
            # Check that the grid points shapes and lengths match their
            # intended values, and that they are in the domain.
            if grid_points.shape != input_shape:
                msg = (
                    f"The shape of the grid points ({grid_points.shape}) does "
                    f"not match the input shape ({input_shape})."
                )
                raise ValueError(msg)

            grid_points_num = np.vectorize(len)(grid_points)
            grid_values_num = np.reshape(middle_shape, shape=input_shape)
            if not np.all(grid_points_num == grid_values_num):
                msg = (
                    f"The number of grid points for each dimension "
                    f"({grid_points_num}) does not match the number of values "
                    f"provided for each dimension ({grid_values_num})."
                )
                raise ValueError(msg)

            lower, upper = domain.bounding_box

            # Set the domain as the bounding box of the grid points.
            # TODO: Should we disallow this (or make it even more strict)?
            min_grid_points = np.vectorize(self.array_backend.min)(grid_points)
            max_grid_points = np.vectorize(self.array_backend.max)(grid_points)

            points_outside = self.array_backend.any(
                (min_grid_points < lower)
                | (max_grid_points > upper),
            )
            if points_outside:
                raise ValueError(
                    f"There are grid points outside the domain's bounding box "
                    f"({lower}, {upper})."
                )

        self._shape = shape
        self._input_shape = input_shape
        self._output_shape = output_shape
        self.grid_points = grid_points
        self._domain = domain
        self.interpolation = interpolation

        # We check this at the end because we do not know the shape before.
        super().__init__(
            extrapolation=extrapolation,
            input_names=input_names,
            output_names=output_names,
        )

    @override
    @cached_property
    def array_backend(self) -> ArrayNamespace[A]:
        return array_namespace(self.grid_values)
    
    @override
    @property
    def shape(self) -> tuple[int, ...]:
        return self._shape

    @override
    @property
    def input_shape(self) -> tuple[int, ...]:
        return self._input_shape

    @override
    @property
    def output_shape(self) -> tuple[int, ...]:
        return self._output_shape

    def round(  # noqa: WPS125
        self,
        decimals: int = 0,
        out: FDataGrid | None = None,
    ) -> FDataGrid:
        """Evenly round to the given number of decimals.

        .. deprecated:: 0.6
            Use :func:`numpy.round` function instead.

        Args:
            decimals: Number of decimal places to round to.
                If decimals is negative, it specifies the number of
                positions to the left of the decimal point. Defaults to 0.
            out: FDataGrid where to place the result, if any.

        Returns:
            Returns a FDataGrid object where all elements
            in its data_matrix are rounded.

        """
        out_matrix = None if out is None else out.data_matrix

        if (
            out is not None
            and (
                self.domain_range != out.domain_range
                or not all(
                    np.array_equal(a, b)
                    for a, b in zip(self.grid_points, out.grid_points)
                )
                or self.data_matrix.shape != out.data_matrix.shape
            )
        ):
            raise ValueError("out parameter is not valid")

        data_matrix = np.round(
            self.data_matrix,
            decimals=decimals,
            out=out_matrix,
        )

        return self.copy(data_matrix=data_matrix) if out is None else out

    @property
    def sample_points(self) -> GridPoints:
        warnings.warn(
            "Parameter sample_points is deprecated. Use the "
            "parameter grid_points instead.",
            DeprecationWarning,
        )
        return self.grid_points

    @property
    def dim_domain(self) -> int:
        return self.grid_points.size

    @property
    def dim_codomain(self) -> int:
        try:
            # The dimension of the image is the length of the array that can
            #  be extracted from the data_matrix using all the dimensions of
            #  the domain.
            return self.data_matrix.shape[1 + self.dim_domain]
        # If there is no array that means the dimension of the image is 1.
        except IndexError:
            return 1

    @property
    def coordinates(self: T) -> _CoordinateIterator[T]:
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

        return _CoordinateIterator(self)

    @property
    def n_samples(self) -> int:
        """
        Return the number of samples.

        This is also the number of rows of the data_matrix.

        Returns:
            Number of samples of the FDataGrid object.

        """
        return self.data_matrix.shape[0]

    @property
    def domain_range(self) -> DomainRange:
        """
        Return the :term:`domain range` of the function.

        It does not have to be equal to the `sample_range`.

        """
        return self._domain_range

    @property
    def interpolation(self) -> Evaluator:
        """Define the type of interpolation applied in `evaluate`."""
        return self._interpolation

    @interpolation.setter
    def interpolation(self, new_interpolation: Optional[Evaluator]) -> None:

        if new_interpolation is None:
            new_interpolation = SplineInterpolation()

        self._interpolation = new_interpolation

    def _evaluate(
        self,
        eval_points: NDArrayFloat,
        *,
        aligned: bool = True,
    ) -> NDArrayFloat:

        return self.interpolation(
            self,
            eval_points,
            aligned=aligned,
        )

    def derivative(
        self: T,
        *,
        order: int = 1,
        method: Optional[Basis] = None,
    ) -> T:
        """
        Differentiate a FDataGrid object.

        By default, it is calculated using central finite differences when
        possible. In the extremes, forward and backward finite differences
        with accuracy 2 are used.

        Args:
            order: Order of the derivative. Defaults to one.
            method: Method to use to compute the derivative. If ``None``
                (the default), finite differences are used. In a basis
                object is passed the grid is converted to a basis
                representation and the derivative is evaluated using that
                representation.

        Returns:
            Derivative function.

        Examples:
            First order derivative

            >>> from skfda import FDataGrid
            >>> import numpy as np

            >>> fdata = FDataGrid([1,2,4,5,8], np.arange(5))
            >>> fdata.derivative()
            FDataGrid(
                array([[[ 0.5],
                        [ 1.5],
                        [ 1.5],
                        [ 2. ],
                        [ 4. ]]]),
                grid_points=array([array([ 0, 1, 2, 3, 4])], dtype=object),
                domain_range=((0.0, 4.0),),
                ...)

            Second order derivative

            >>> fdata = FDataGrid([1,2,4,5,8], np.arange(5))
            >>> fdata.derivative(order=2)
            FDataGrid(
                array([[[ 3.],
                        [ 1.],
                        [-1.],
                        [ 2.],
                        [ 5.]]]),
                grid_points=array([array([ 0, 1, 2, 3, 4])], dtype=object),
                domain_range=((0.0, 4.0),),
                ...)

        """
        order_list = np.atleast_1d(order)
        if order_list.ndim != 1 or len(order_list) != self.dim_domain:
            raise ValueError("The order for each partial should be specified.")

        if method is None:
            operator = findiff.FinDiff(*[
                (1 + i, *z)
                for i, z in enumerate(
                    zip(self.grid_points, order_list),
                )
            ])
            data_matrix = operator(self.data_matrix.astype(float))
        else:
            data_matrix = self.to_basis(method).derivative(
                order=order,
            )(
                self.grid_points,
                grid=True,
            )

        return self.copy(
            data_matrix=data_matrix,
        )

    def integrate(
        self: T,
        *,
        domain: Optional[DomainRange] = None,
    ) -> NDArrayFloat:
        """
        Integration of the FData object.

        The integration is performed over the whole domain. Thus, for a
        function of several variables this will be a multiple integral.

        For a vector valued function the vector of integrals will be
        returned.

        Args:
            domain: Domain range where we want to integrate.
                By default is None as we integrate on the whole domain.

        Returns:
            NumPy array of size (``n_samples``, ``dim_codomain``)
            with the integrated data.

        Examples:
            >>> from skfda import FDataGrid
            >>> import numpy as np
            >>> fdata = FDataGrid([1,2,4,5,8], np.arange(5))
            >>> fdata.integrate()
            array([[ 15.]])
        """
        if domain is not None:
            data = self.restrict(domain)
        else:
            data = self

        integrand = data.data_matrix

        for g in data.grid_points[::-1]:
            integrand = scipy.integrate.simpson(
                integrand,
                x=g,
                axis=-2,
            )

        return integrand

    def _check_same_dimensions(self: T, other: T) -> None:
        if self.data_matrix.shape[1:-1] != other.data_matrix.shape[1:-1]:
            raise ValueError("Error in columns dimensions")
        if not grid_points_equal(self.grid_points, other.grid_points):
            raise ValueError("Grid points for both objects must be equal")

    def _get_points_and_values(self: T) -> Tuple[NDArrayFloat, NDArrayFloat]:
        return (
            cartesian_product(self.grid_points),
            self.data_matrix.reshape((self.n_samples, -1)).T,
        )

    def _get_input_points(self: T) -> GridPoints:
        return self.grid_points

    def sum(  # noqa: WPS125
        self: T,
        *,
        axis: Optional[int] = None,
        out: None = None,
        keepdims: bool = False,
        skipna: bool = False,
        min_count: int = 0,
    ) -> T:
        """Compute the sum of all the samples.

        Args:
            axis: Used for compatibility with numpy. Must be None or 0.
            out: Used for compatibility with numpy. Must be None.
            keepdims: Used for compatibility with numpy. Must be False.
            skipna: Wether the NaNs are ignored or not.
            min_count: Number of valid (non NaN) data to have in order
                for the a variable to not be NaN when `skipna` is
                `True`.

        Returns:
            A FDataGrid object with just one sample representing
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

        data = (
            np.nansum(self.data_matrix, axis=0, keepdims=True) if skipna
            else np.sum(self.data_matrix, axis=0, keepdims=True)
        )

        if min_count > 0:
            valid = ~np.isnan(self.data_matrix)
            n_valid = np.sum(valid, axis=0)
            data[n_valid < min_count] = np.NaN

        return self.copy(
            data_matrix=data,
            sample_names=(None,),
        )

    def var(self: T, correction: int = 0) -> T:
        """Compute the variance of a set of samples in a FDataGrid object.

        Args:
            correction: degrees of freedom adjustment. The divisor used in the
                calculation is `N - correction`, where `N` represents the
                number of elements. Default: `0`.

        Returns:
            A FDataGrid object with just one sample representing the
            variance of all the samples in the original FDataGrid object.

        """
        return self.copy(
            data_matrix=np.array([np.var(
                self.data_matrix,
                axis=0,
                ddof=correction,
            )]),
            sample_names=("variance",),
        )

    @overload
    def cov(  # noqa: WPS451
        self: T,
        s_points: NDArrayFloat,
        t_points: NDArrayFloat,
        /,
        correction: int = 0,
    ) -> NDArrayFloat:
        pass

    @overload
    def cov(  # noqa: WPS451
        self: T,
        /,
        correction: int = 0,
    ) -> Callable[[NDArrayFloat, NDArrayFloat], NDArrayFloat]:
        pass

    def cov(  # noqa: WPS320, WPS451
        self: T,
        s_points: Optional[NDArrayFloat] = None,
        t_points: Optional[NDArrayFloat] = None,
        /,
        correction: int = 0,
    ) -> Union[
        Callable[[NDArrayFloat, NDArrayFloat], NDArrayFloat],
        NDArrayFloat,
    ]:
        """Compute the covariance.

        Calculates the covariance matrix representing the covariance of the
        functional samples at the observation points.
        if s_points and t_points are both specified, this method returns the
        covariance function evaluated at the grid points formed by the
        cartesian product of s_points and t_points.

        Args:
            s_points: Grid points where the covariance function is evaluated.
            t_points: Grid points where the covariance function is evaluated.
            correction: degrees of freedom adjustment. The divisor used in the
                calculation is `N - correction`, where `N` represents the
                number of elements. Default: `0`.

        Returns:
            Covariance function.

        """
        # To avoid circular imports
        from ..misc.covariances import EmpiricalGrid
        cov_function = EmpiricalGrid(self, correction=correction)
        if s_points is None or t_points is None:
            return cov_function
        return cov_function(s_points, t_points)

    def gmean(self: T) -> T:
        """Compute the geometric mean of all samples in the FDataGrid object.

        Returns:
            A FDataGrid object with just one sample representing
            the geometric mean of all the samples in the original
            FDataGrid object.

        """
        return self.copy(
            data_matrix=[
                scipy.stats.mstats.gmean(self.data_matrix, 0),
            ],
            sample_names=("geometric mean",),
        )

    def equals(self, other: object) -> bool:
        """Comparison of FDataGrid objects."""
        if not super().equals(other):
            return False

        other = cast(FDataGrid, other)

        if not np.array_equal(self.data_matrix, other.data_matrix):
            return False

        # Comparison of the domain
        if (
            not np.array_equal(self.domain_range, other.domain_range)
            or len(self.grid_points) != len(other.grid_points)
            or not all(
                np.array_equal(a, b)
                for a, b in zip(self.grid_points, other.grid_points)
            )
        ):
            return False

        return self.interpolation == other.interpolation

    def _eq_elemenwise(self: T, other: T) -> NDArrayBool:
        """Elementwise equality of FDataGrid."""
        return np.all(  # type: ignore[no-any-return]
            self.data_matrix == other.data_matrix,
            axis=tuple(range(1, self.data_matrix.ndim)),
        )

    def _get_op_matrix(
        self,
        other: Union[T, NDArrayFloat, NDArrayInt, float],
    ) -> Union[None, float, NDArrayFloat, NDArrayInt]:
        if isinstance(other, numbers.Real):
            return float(other)
        elif isinstance(other, np.ndarray):

            if other.shape in {(), (1,)}:
                return other
            elif other.shape == (self.n_samples,):
                other_index = (
                    (slice(None),) + (np.newaxis,)
                    * (self.data_matrix.ndim - 1)
                )

                return other[other_index]
            elif other.shape == (
                self.n_samples,
                self.dim_codomain,
            ):
                other_index = (
                    (slice(None),) + (np.newaxis,)
                    * (self.data_matrix.ndim - 2)
                    + (slice(None),)
                )

                return other[other_index]

            raise ValueError(
                f"Invalid dimensions in operator between "
                f"FDataGrid (data_matrix.shape={self.data_matrix.shape}) "
                f"and Numpy array (shape={other.shape})",
            )

        elif isinstance(other, FDataGrid):
            self._check_same_dimensions(other)
            return other.data_matrix

        return None

    def __add__(
        self: T,
        other: Union[T, NDArrayFloat, NDArrayInt, float],
    ) -> T:

        data_matrix = self._get_op_matrix(other)
        if data_matrix is None:
            return NotImplemented

        return self._copy_op(other, data_matrix=self.data_matrix + data_matrix)

    def __radd__(
        self: T,
        other: Union[T, NDArrayFloat, NDArrayInt, float],
    ) -> T:

        return self.__add__(other)

    def __sub__(
        self: T,
        other: Union[T, NDArrayFloat, NDArrayInt, float],
    ) -> T:

        data_matrix = self._get_op_matrix(other)
        if data_matrix is None:
            return NotImplemented

        return self._copy_op(other, data_matrix=self.data_matrix - data_matrix)

    def __rsub__(
        self: T,
        other: Union[T, NDArrayFloat, NDArrayInt, float],
    ) -> T:

        data_matrix = self._get_op_matrix(other)
        if data_matrix is None:
            return NotImplemented

        return self.copy(data_matrix=data_matrix - self.data_matrix)

    def __mul__(
        self: T,
        other: Union[T, NDArrayFloat, NDArrayInt, float],
    ) -> T:

        data_matrix = self._get_op_matrix(other)
        if data_matrix is None:
            return NotImplemented

        return self._copy_op(other, data_matrix=self.data_matrix * data_matrix)

    def __rmul__(
        self: T,
        other: Union[T, NDArrayFloat, NDArrayInt, float],
    ) -> T:

        return self.__mul__(other)

    def __truediv__(
        self: T,
        other: Union[T, NDArrayFloat, NDArrayInt, float],
    ) -> T:

        data_matrix = self._get_op_matrix(other)
        if data_matrix is None:
            return NotImplemented

        return self._copy_op(other, data_matrix=self.data_matrix / data_matrix)

    def __rtruediv__(
        self: T,
        other: Union[T, NDArrayFloat, NDArrayInt, float],
    ) -> T:

        data_matrix = self._get_op_matrix(other)
        if data_matrix is None:
            return NotImplemented

        return self._copy_op(other, data_matrix=data_matrix / self.data_matrix)

    def __neg__(self: T) -> T:
        """Negation of FData object."""
        return self.copy(data_matrix=-self.data_matrix)

    def concatenate(self: T, *others: T, as_coordinates: bool = False) -> T:
        """Join samples from a similar FDataGrid object.

        Joins samples from another FDataGrid object if it has the same
        dimensions and sampling points.

        Args:
            others: Objects to be concatenated.
            as_coordinates:  If False concatenates as
                new samples, else, concatenates the other functions as
                new components of the image. Defaults to false.

        Returns:
            FDataGrid object with the samples from the original objects.

        Examples:
            >>> from skfda import FDataGrid
            >>> import numpy as np
            >>> fd = FDataGrid([1,2,4,5,8], np.arange(5))
            >>> fd_2 = FDataGrid([3,4,7,9,2], np.arange(5))
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
                grid_points=array([array([ 0, 1, 2, 3, 4])], dtype=object),
                domain_range=((0.0, 4.0),),
                ...)

        """
        # Checks
        if not as_coordinates:
            for other in others:
                self._check_same_dimensions(other)

        elif not all(
            grid_points_equal(self.grid_points, other.grid_points)
            for other in others
        ):
            raise ValueError(
                "All the FDataGrids must be sampled in the  same "
                "grid points.",
            )

        elif any(self.n_samples != other.n_samples for other in others):

            raise ValueError(
                f"All the FDataGrids must contain the same "
                f"number of samples {self.n_samples} to "
                f"concatenate as a new coordinate.",
            )

        data = [self.data_matrix] + [other.data_matrix for other in others]

        if as_coordinates:

            coordinate_names = [fd.coordinate_names for fd in (self, *others)]

            return self.copy(
                data_matrix=np.concatenate(data, axis=-1),
                coordinate_names=sum(coordinate_names, ()),
            )

        sample_names = [fd.sample_names for fd in (self, *others)]

        return self.copy(
            data_matrix=np.concatenate(data, axis=0),
            sample_names=sum(sample_names, ()),
        )

    def scatter(self, *args: Any, **kwargs: Any) -> Figure:
        """Scatter plot of the FDatGrid object.

        Args:
            args: Positional arguments to be passed to the class
                :class:`~skfda.exploratory.visualization.representation.ScatterPlot`.
            kwargs: Keyword arguments to be passed to the class
                :class:`~skfda.exploratory.visualization.representation.ScatterPlot`.

        Returns:
            Figure object in which the graphs are plotted.


        """
        from ..exploratory.visualization.representation import ScatterPlot

        return ScatterPlot(self, *args, **kwargs).plot()

    def to_basis(self, basis: Basis, **kwargs: Any) -> FDataBasis:
        """Return the basis representation of the object.

        Args:
            basis(Basis): basis object in which the functional data are
                going to be represented.
            kwargs: keyword arguments to be passed to
                FDataBasis.from_data().

        Returns:
            FDataBasis: Basis representation of the funtional data
            object.

        Examples:
            >>> from skfda import FDataGrid
            >>> import numpy as np
            >>> import skfda
            >>> t = np.linspace(0, 1, 5)
            >>> x = np.sin(2 * np.pi * t) + np.cos(2 * np.pi * t) + 2
            >>> x
            array([ 3.,  3.,  1.,  1.,  3.])

            >>> fd = FDataGrid(x, t)
            >>> basis = skfda.representation.basis.FourierBasis(n_basis=3)
            >>> fd_b = fd.to_basis(basis)
            >>> fd_b.coefficients.round(2)
            array([[ 2.  , 0.71, 0.71]])

        """
        from ..preprocessing.smoothing import BasisSmoother

        if self.dim_domain != basis.dim_domain:
            raise ValueError(
                f"The domain of the function has "
                f"dimension {self.dim_domain} "
                f"but the domain of the basis has "
                f"dimension {basis.dim_domain}",
            )
        elif self.dim_codomain != basis.dim_codomain:
            raise ValueError(
                f"The codomain of the function has "
                f"dimension {self.dim_codomain} "
                f"but the codomain of the basis has "
                f"dimension {basis.dim_codomain}",
            )

        # Readjust the domain range if there was not an explicit one
        if not basis.is_domain_range_fixed():
            basis = basis.copy(domain_range=self.domain_range)

        smoother = BasisSmoother(
            basis=basis,
            **kwargs,
            return_basis=True,
        )

        return smoother.fit_transform(self)

    def to_grid(  # noqa: D102
        self: T,
        grid_points: Optional[GridPointsLike] = None,
        *,
        sample_points: Optional[GridPointsLike] = None,
    ) -> T:

        if sample_points is not None:
            warnings.warn(
                "Parameter sample_points is deprecated. Use the "
                "parameter grid_points instead.",
                DeprecationWarning,
            )
            grid_points = sample_points

        grid_points = (
            self.grid_points
            if grid_points is None
            else check_grid_points(grid_points)
        )

        return self.copy(
            data_matrix=self(grid_points, grid=True),
            grid_points=grid_points,
        )

    def copy(  # noqa: WPS211
        self: T,
        *,
        deep: bool = False,  # For Pandas compatibility
        data_matrix: Optional[ArrayLike] = None,
        grid_points: Optional[GridPointsLike] = None,
        sample_points: Optional[GridPointsLike] = None,
        domain_range: Optional[DomainRangeLike] = None,
        dataset_name: Optional[str] = None,
        argument_names: Optional[LabelTupleLike] = None,
        coordinate_names: Optional[LabelTupleLike] = None,
        sample_names: Optional[LabelTupleLike] = None,
        extrapolation: Optional[ExtrapolationLike] = None,
        interpolation: Optional[Evaluator] = None,
    ) -> T:
        """
        Return a copy of the FDataGrid.

        If an argument is provided the corresponding attribute in the new copy
        is updated.

        """
        if sample_points is not None:
            warnings.warn(
                "Parameter sample_points is deprecated. Use the "
                "parameter grid_points instead.",
                DeprecationWarning,
            )
            grid_points = sample_points

        if data_matrix is None:
            # The data matrix won't be writeable
            data_matrix = self.data_matrix

        if grid_points is None:
            # Grid points won`t be writeable
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

        return FDataGrid(
            data_matrix,
            grid_points=grid_points,
            domain_range=domain_range,
            dataset_name=dataset_name,
            argument_names=argument_names,
            coordinate_names=coordinate_names,
            sample_names=sample_names,
            extrapolation=extrapolation,
            interpolation=interpolation,
        )

    def restrict(
        self: T,
        domain_range: DomainRangeLike,
        *,
        with_bounds: bool = False,
    ) -> T:
        """
        Restrict the functions to a new domain range.

        Args:
            domain_range: New domain range.
            with_bounds: Whether or not to ensure domain boundaries
                appear in `grid_points`.

        Returns:
            Restricted function.

        """
        from ..misc.validation import validate_domain_range

        domain_range = validate_domain_range(domain_range)
        assert all(
            c <= a < b <= d  # noqa: WPS228
            for ((a, b), (c, d)) in zip(domain_range, self.domain_range)
        )

        # Eliminate points outside the new range.
        slice_list = []
        for (a, b), dim_points in zip(domain_range, self.grid_points):
            ia = np.searchsorted(dim_points, a)
            ib = np.searchsorted(dim_points, b, 'right')
            slice_list.append(slice(ia, ib))
        grid_points = [g[s] for g, s in zip(self.grid_points, slice_list)]
        data_matrix = self.data_matrix[(slice(None),) + tuple(slice_list)]

        # Ensure that boundaries are in grid_points.
        if with_bounds:
            # Update `grid_points`
            for dim, (a, b) in enumerate(domain_range):
                dim_points = grid_points[dim]
                left = [a] if a < dim_points[0] else []
                right = [b] if b > dim_points[-1] else []
                grid_points[dim] = np.concatenate((left, dim_points, right))
            # Evaluate
            data_matrix = self(grid_points, grid=True)

        return self.copy(
            domain_range=domain_range,
            grid_points=grid_points,
            data_matrix=data_matrix,
        )

    def shift(
        self,
        shifts: Union[ArrayLike, float],
        *,
        restrict_domain: bool = False,
        extrapolation: AcceptedExtrapolation = "default",
        grid_points: Optional[GridPointsLike] = None,
    ) -> FDataGrid:
        r"""
        Perform a shift of the curves.

        The i-th shifted function :math:`y_i` has the form

        .. math::
            y_i(t) = x_i(t + \delta_i)

        where :math:`x_i` is the i-th original function and :math:`delta_i` is
        the shift performed for that function, that must be a vector in the
        domain space.

        Note that a positive shift moves the graph of the function in the
        negative direction and vice versa.

        Args:
            shifts: List with the shifts
                corresponding for each sample or numeric with the shift to
                apply to all samples.
            restrict_domain: If True restricts the domain to avoid the
                evaluation of points outside the domain using extrapolation.
                Defaults uses extrapolation.
            extrapolation: Controls the
                extrapolation mode for elements outside the domain range.
                By default uses the method defined in fd. See extrapolation to
                more information.
            grid_points: Grid of points where
                the functions are evaluated to obtain the discrete
                representation of the object to operate. If ``None`` the
                current grid_points are used to unificate the domain of the
                shifted data.

        Returns:
            Shifted functions.

        Examples:
            >>> import numpy as np
            >>> import skfda
            >>>
            >>> t = np.linspace(0, 1, 6)
            >>> x = np.array([t, t**2, t**3])
            >>> fd = skfda.FDataGrid(x, t)
            >>> fd.domain_range[0]
            (0.0, 1.0)
            >>> fd.grid_points[0]
            array([ 0. ,  0.2,  0.4,  0.6,  0.8,  1. ])
            >>> fd.data_matrix[..., 0]
            array([[ 0.   ,  0.2  ,  0.4  ,  0.6  ,  0.8  ,  1.   ],
                   [ 0.   ,  0.04 ,  0.16 ,  0.36 ,  0.64 ,  1.   ],
                   [ 0.   ,  0.008,  0.064,  0.216,  0.512,  1.   ]])

            Shift all curves by the same amount:

            >>> shifted = fd.shift(0.2)
            >>> shifted.domain_range[0]
            (0.0, 1.0)
            >>> shifted.grid_points[0]
            array([ 0. ,  0.2,  0.4,  0.6,  0.8,  1. ])
            >>> shifted.data_matrix[..., 0]
            array([[ 0.2  ,  0.4  ,  0.6  ,  0.8  ,  1.   ,  1.2  ],
                   [ 0.04 ,  0.16 ,  0.36 ,  0.64 ,  1.   ,  1.36 ],
                   [ 0.008,  0.064,  0.216,  0.512,  1.   ,  1.488]])


            Different shift per curve:

            >>> shifted = fd.shift([-0.2, 0.0, 0.2])
            >>> shifted.domain_range[0]
            (0.0, 1.0)
            >>> shifted.grid_points[0]
            array([ 0. ,  0.2,  0.4,  0.6,  0.8,  1. ])
            >>> shifted.data_matrix[..., 0]
            array([[-0.2  ,  0.   ,  0.2  ,  0.4  ,  0.6  ,  0.8  ],
                   [ 0.   ,  0.04 ,  0.16 ,  0.36 ,  0.64 ,  1.   ],
                   [ 0.008,  0.064,  0.216,  0.512,  1.   ,  1.488]])

            It is possible to restrict the domain to prevent the need for
            extrapolations:

            >>> shifted = fd.shift([-0.3, 0.1, 0.2], restrict_domain=True)
            >>> shifted.domain_range[0]
            (0.3, 0.8)

        """
        grid_points = (
            self.grid_points if grid_points is None
            else grid_points
        )

        return super().shift(
            shifts=shifts,
            restrict_domain=restrict_domain,
            extrapolation=extrapolation,
            grid_points=grid_points,
        )

    def compose(
        self: T,
        fd: T,
        *,
        eval_points: Optional[GridPointsLike] = None,
    ) -> T:
        """Composition of functions.

        Performs the composition of functions.

        Args:
            fd: FData object to make the composition. Should
                have the same number of samples and image dimension equal to 1.
            eval_points: Points to perform the evaluation.

        Returns:
            Function representing the composition.

        """
        if self.dim_domain != fd.dim_codomain:
            raise ValueError(
                f"Dimension of codomain of first function do not "
                f"match with the domain of the second function "
                f"{self.dim_domain} != {fd.dim_codomain}.",
            )

        # All composed with same function
        if fd.n_samples == 1 and self.n_samples != 1:
            fd = fd.copy(data_matrix=np.repeat(
                fd.data_matrix,
                self.n_samples,
                axis=0,
            ))

        if fd.dim_domain == 1:
            if eval_points is None:
                try:
                    eval_points = fd.grid_points[0]
                except AttributeError:
                    eval_points = np.linspace(
                        *fd.domain_range[0],
                        constants.N_POINTS_COARSE_MESH,
                    )

            eval_points_transformation = fd(eval_points)
            data_matrix = self(
                eval_points_transformation,
                aligned=False,
            )
        else:
            eval_points = (
                fd.grid_points
                if eval_points is None
                else check_grid_points(eval_points)
            )

            grid_transformation = fd(eval_points, grid=True)

            lengths = [len(ax) for ax in eval_points]

            eval_points_transformation = np.empty((
                self.n_samples,
                np.prod(lengths),
                self.dim_domain,
            ))

            for i in range(self.n_samples):
                eval_points_transformation[i] = np.array(
                    list(map(np.ravel, grid_transformation[i].T)),
                ).T

            data_matrix = self(
                eval_points_transformation,
                aligned=False,
            )

        return self.copy(
            data_matrix=data_matrix,
            grid_points=eval_points,
            domain_range=fd.domain_range,
            argument_names=fd.argument_names,
        )

    def __str__(self) -> str:
        """Return str(self)."""
        return (
            f"Data set:    {self.data_matrix}\n"
            f"grid_points:    {self.grid_points}\n"
            f"time range:    {self.domain_range}"
        )

    def __repr__(self) -> str:
        """Return repr(self)."""
        return (
            f"FDataGrid("  # noqa: WPS221
            f"\n{self.data_matrix!r},"
            f"\ngrid_points={self.grid_points!r},"
            f"\ndomain_range={self.domain_range!r},"
            f"\ndataset_name={self.dataset_name!r},"
            f"\nargument_names={self.argument_names!r},"
            f"\ncoordinate_names={self.coordinate_names!r},"
            f"\nextrapolation={self.extrapolation!r},"
            f"\ninterpolation={self.interpolation!r})"
        ).replace(
            '\n',
            '\n    ',
        )

    def __getitem__(
        self: T,
        key: Union[int, slice, NDArrayInt, NDArrayBool],
    ) -> T:
        """Return self[key]."""
        key = _check_array_key(self.data_matrix, key)

        return self.copy(
            data_matrix=self.data_matrix[key],
            sample_names=list(np.array(self.sample_names)[key]),
        )

    #####################################################################
    # Numpy methods
    #####################################################################

    def __array_ufunc__(
        self,
        ufunc: Any,
        method: str,
        *inputs: Any,
        **kwargs: Any,
    ) -> Any:

        for i in inputs:
            if (
                isinstance(i, FDataGrid)
                and not grid_points_equal(i.grid_points, self.grid_points)
            ):
                return NotImplemented

        new_inputs = [
            i.data_matrix if isinstance(i, FDataGrid)
            else self._get_op_matrix(i) for i in inputs
        ]

        outputs = kwargs.pop('out', None)
        if outputs:
            new_outputs = [
                o.data_matrix if isinstance(o, FDataGrid)
                else o for o in outputs
            ]
            kwargs['out'] = tuple(new_outputs)
        else:
            new_outputs = (None,) * ufunc.nout

        results = getattr(ufunc, method)(*new_inputs, **kwargs)
        if results is NotImplemented:
            return NotImplemented

        if ufunc.nout == 1:
            results = (results,)

        results = tuple(
            (result if output is None else output)
            for result, output in zip(results, new_outputs)
        )

        results = [self.copy(data_matrix=r) for r in results]

        return results[0] if len(results) == 1 else results

    #####################################################################
    # Pandas ExtensionArray methods
    #####################################################################

    def _take_allow_fill(
        self: T,
        indices: NDArrayInt,
        fill_value: T,
    ) -> T:
        result = self.copy()
        result.data_matrix = np.full(
            (len(indices),) + self.data_matrix.shape[1:],
            np.nan,
        )

        positive_mask = indices >= 0
        result.data_matrix[positive_mask] = self.data_matrix[
            indices[positive_mask]
        ]

        if fill_value is not self.dtype.na_value:
            result.data_matrix[~positive_mask] = fill_value.data_matrix[0]

        return result

    @property
    def dtype(self) -> GridDiscretizedFunctionDType:
        """The dtype for this extension array, GridDiscretizedFunctionDType"""
        return GridDiscretizedFunctionDType(
            grid_points=self.grid_points,
            domain_range=self.domain_range,
            dim_codomain=self.dim_codomain,
        )

    @property
    def nbytes(self) -> int:
        """
        The number of bytes needed to store this object in memory.
        """
        return self.data_matrix.nbytes + sum(
            p.nbytes for p in self.grid_points
        )

    def isna(self) -> NDArrayBool:
        """
        Return a 1-D array indicating if each value is missing.

        Returns:
            na_values: Positions of NA.
        """
        return np.all(  # type: ignore[no-any-return]
            np.isnan(self.data_matrix),
            axis=tuple(range(1, self.data_matrix.ndim)),
        )


class GridDiscretizedFunctionDType(
    pd.api.extensions.ExtensionDtype,  # type: ignore[misc]
):
    """DType corresponding to FDataGrid in Pandas."""

    name = 'GridDiscretizedFunction'
    kind = 'O'
    type = GridDiscretizedFunction  # noqa: WPS125
    na_value = pd.NA

    def __init__(
        self,
        grid_points: GridPointsLike,
        dim_codomain: int,
        domain_range: Optional[DomainRangeLike] = None,
    ) -> None:
        from ..misc.validation import validate_domain_range

        self.grid_points = check_grid_points(grid_points)

        if domain_range is None:
            domain_range = tuple((s[0], s[-1]) for s in self.grid_points)

        self.domain_range = validate_domain_range(domain_range)
        self.dim_codomain = dim_codomain

    @classmethod
    def construct_array_type(cls) -> Type[FDataGrid]:  # noqa: D102
        return FDataGrid

    def _na_repr(self) -> FDataGrid:

        shape = (
            (1,)
            + tuple(len(s) for s in self.grid_points)
            + (self.dim_codomain,)
        )

        data_matrix = np.full(shape=shape, fill_value=np.NaN)

        return FDataGrid(
            grid_points=self.grid_points,
            domain_range=self.domain_range,
            data_matrix=data_matrix,
        )

    def __eq__(self, other: Any) -> bool:
        """
        Compare dtype equality.

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

        return (
            isinstance(other, GridDiscretizedFunctionDType)
            and self.dim_codomain == other.dim_codomain
            and self.domain_range == other.domain_range
            and grid_points_equal(self.grid_points, other.grid_points)
        )

    def __hash__(self) -> int:
        # Grid points are not currently hashed
        return hash((self.domain_range, self.dim_codomain))


class _CoordinateIterator(Sequence[T]):
    """Internal class to iterate through the image coordinates."""

    def __init__(self, fdatagrid: T) -> None:
        """Create an iterator through the image coordinates."""
        self._fdatagrid = fdatagrid

    def __getitem__(
        self,
        key: Union[int, slice, NDArrayInt, NDArrayBool],
    ) -> T:
        """Get a specific coordinate."""
        s_key = key
        if isinstance(s_key, int):
            s_key = slice(s_key, s_key + 1)

        coordinate_names = np.array(self._fdatagrid.coordinate_names)[s_key]

        return self._fdatagrid.copy(
            data_matrix=self._fdatagrid.data_matrix[..., key],
            coordinate_names=tuple(coordinate_names),
        )

    def __len__(self) -> int:
        """Return the number of coordinates."""
        return self._fdatagrid.dim_codomain
