"""Discretised functional data module.

This module defines a class for representing discretized irregular data,
in which the observations may be made in different grid points in each
data function, and the overall density of the observations may be low

"""
from __future__ import annotations

import itertools
import numbers
from typing import (
    Any,
    Literal,
    Optional,
    Sequence,
    Tuple,
    Type,
    TypeVar,
    Union,
)

import numpy as np
import pandas.api.extensions
import scipy
from matplotlib.figure import Figure

from .._utils import _cartesian_product, _check_array_key, _to_grid_points
from ..typing._base import (
    DomainRange,
    DomainRangeLike,
    GridPoints,
    GridPointsLike,
    LabelTupleLike,
)
from ..typing._numpy import ArrayLike, NDArrayBool, NDArrayFloat, NDArrayInt
from ._functional_data import FData
from .basis import Basis, FDataBasis
from .evaluator import Evaluator
from .extrapolation import ExtrapolationLike
from .grid import FDataGrid
from .interpolation import SplineInterpolation

T = TypeVar("T", bound='FDataIrregular')
IrregularToBasisConversionType = Literal[
    "function-wise", "mixed-effects", "mixed-effects-minimize",
]

######################
# Auxiliary functions#
######################


def _reduceat(
    ufunc,
    array: ArrayLike,
    indices: ArrayLike,
    axis: int = 0,
    dtype=None,
    out=None,
    *,
    value_empty,
):
    """
    Wrapped `np.ufunc.reduceat` to manage some edge cases.

    The edge cases are the one described in the doc of
    `np.ufunc.reduceat`. Different behaviours are the following:
        - No exception is raised when `indices[i] < 0` or
            `indices[i] >=len(array)`. Instead, the corresponding value
            is `value_empty`.
        - When not in the previous case, the result is `value_empty` if
            `indices[i] == indices[i+1]` and otherwise, the same as
            `ufunc.reduce(array[indices[i]:indices[i+1]])`. This means
            that an exception is still be raised if `indices[i] >
            indices[i+1]`.

    Note: The `value_empty` must be convertible to the `dtype` (either
          provided or inferred from the `ufunc` operations).
    """
    array = np.asarray(array)
    indices = np.asarray(indices)

    n = array.shape[axis]
    good_axis_idx = (
        (indices >= 0) & (indices < n) & (np.diff(indices, append=n) > 0)
    )

    good_idx = [slice(None)] * array.ndim
    good_idx[axis] = good_axis_idx
    good_idx = tuple(good_idx)

    reduceat_out = ufunc.reduceat(
        array, indices[good_axis_idx], axis=axis, dtype=dtype
    )

    out_shape = list(array.shape)
    out_shape[axis] = len(indices)
    out_dtype = dtype or reduceat_out.dtype

    if out is None:
        out = np.full(out_shape, value_empty, dtype=out_dtype)
    else:
        out.astype(out_dtype, copy=False)
        out.fill(value_empty)

    out[good_idx] = reduceat_out

    return out


def _get_sample_range_from_data(
    start_indices: NDArrayInt,
    points: NDArrayFloat,
) -> DomainRangeLike:
    """Compute the domain ranges of each sample.

    Args:
        start_indices: start_indices of the FDataIrregular object.
        points: points of the FDataIrregular object.

    Returns:
        DomainRange: (sample_range) a tuple of tuples of 2-tuples where
            sample_range[f][d] = (min_point, max_point) is the domain range for
            the function f in dimension d.
    """
    return np.stack(
        [
            _reduceat(
                ufunc,
                points,
                start_indices,
                value_empty=np.nan,
                dtype=float,
            )
            for ufunc in (np.fmin, np.fmax)
        ],
        axis=-1,
    )


def _get_domain_range_from_sample_range(
    sample_range: DomainRangeLike,
) -> DomainRange:
    """Compute the domain range of the whole dataset.

    Args:
        sample_range: sample_range of the FDataIrregular object.

    Returns:
        DomainRange: (domain_range) a tuple of 2-tuples where
            domain_range[d] = (min_point, max_point) is the domain range for
            the dimension d.
    """
    sample_range_array = np.asarray(sample_range)
    min_arguments = np.nanmin(sample_range_array[..., 0], axis=0)
    max_arguments = np.nanmax(sample_range_array[..., 1], axis=0)
    return tuple(zip(min_arguments, max_arguments))


######################
# FDataIrregular#
######################


class FDataIrregular(FData):  # noqa: WPS214
    r"""Represent discretised functional data of an irregular or sparse nature.

    Class for representing irregular functional data in a compact manner,
    allowing basic operations, representation and conversion to basis format.

    Attributes:
        start_indices: A unidimensional array which stores the index of
            the functional_values and functional_values arrays where the data
            of each individual curve of the sample begins.
        points: An array of every argument of the domain for
            every curve in the sample. Each row contains an observation.
        values: An array of every value of the codomain for
            every curve in the sample. Each row contains an observation.
        domain_range: 2 dimension matrix where each row
            contains the bounds of the interval in which the functional data
            is considered to exist for each one of the axes.
        dataset_name: Name of the dataset.
        argument_names: Tuple containing the names of the different
            arguments.
        coordinate_names: Tuple containing the names of the different
            coordinate functions.
        extrapolation: Defines the default type of
            extrapolation. By default None, which does not apply any type of
            extrapolation. See `Extrapolation` for detailled information of the
            types of extrapolation.
        interpolation: Defines the type of interpolation
            applied in `evaluate`.

    Raises:
        ValueError:
            - if `points` and `values` lengths don't match
            - if `start_indices` does'nt start with `0`, or is decreasing
                somewhere, or ends with a value greater than or equal to
                `len(points)`.

    Examples:
        Representation of an irregular functional data object with 2 samples
        representing a function :math:`f : \mathbb{R}\longmapsto\mathbb{R}`,
        with 2 and 3 discretization points respectively.

        >>> indices = [0, 2]
        >>> arguments = [[1], [2], [3], [4], [5]]
        >>> values = [[1], [2], [3], [4], [5]]
        >>> FDataIrregular(indices, arguments, values)
        FDataIrregular(
            start_indices=array([ 0, 2]),
            points=array([[ 1],
                [ 2],
                [ 3],
                [ 4],
                [ 5]]),
            values=array([[ 1],
                [ 2],
                [ 3],
                [ 4],
                [ 5]]),
            domain_range=((1.0, 5.0),),
            ...)

        The number of arguments and values must be the same.

        >>> indices = [0,2]
        >>> arguments = np.arange(5).reshape(-1, 1)
        >>> values = np.arange(6).reshape(-1, 1)
        >>> FDataIrregular(indices, arguments, values)
        Traceback (most recent call last):
            ....
        ValueError: Dimension mismatch ...

        The indices in start_indices must point to correct rows
        in points and values.

        >>> indices = [0,7]
        >>> arguments = np.arange(5).reshape(-1, 1)
        >>> values = np.arange(5).reshape(-1, 1)
        >>> FDataIrregular(indices, arguments, values)
        Traceback (most recent call last):
            ....
        ValueError: Index in start_indices out of bounds...

        FDataIrregular supports higher dimensional data both in the domain
        and in the codomain (image).

        Representation of a functional data object with 2 samples
        representing a function :math:`f : \mathbb{R}\longmapsto\mathbb{R}^2`.

        >>> indices = [0, 2]
        >>> arguments = [[1.], [2.], [3.], [4.], [5.]]
        >>> values = [[1., 1.], [2., 2.], [3., 3.], [4., 4.], [5., 5.]]
        >>> fd = FDataIrregular(indices, arguments, values)
        >>> fd.dim_domain, fd.dim_codomain
        (1, 2)

        Representation of a functional data object with 2 samples
        representing a function :math:`f : \mathbb{R}^2\longmapsto\mathbb{R}`.

        >>> indices = [0, 2]
        >>> arguments = [[1., 1.], [2., 2.], [3., 3.], [4., 4.], [5., 5.]]
        >>> values = [[1.], [2.], [3.], [4.], [5.]]
        >>> fd = FDataIrregular(indices, arguments, values)
        >>> fd.dim_domain, fd.dim_codomain
        (2, 1)

    """

    def __init__(  # noqa:  WPS211
        self,
        start_indices: ArrayLike,
        points: ArrayLike,
        values: ArrayLike,
        *,
        domain_range: Optional[DomainRangeLike] = None,
        dataset_name: Optional[str] = None,
        sample_names: Optional[LabelTupleLike] = None,
        extrapolation: Optional[ExtrapolationLike] = None,
        interpolation: Optional[Evaluator] = None,
        argument_names: Optional[LabelTupleLike] = None,
        coordinate_names: Optional[LabelTupleLike] = None,
    ):
        """Construct a FDataIrregular object."""
        self.start_indices = np.asarray(start_indices)
        self.points = np.asarray(points)
        if self.points.ndim == 1:
            self.points = self.points.reshape(-1, 1)
        self.values = np.asarray(values)
        if self.values.ndim == 1:
            self.values = self.values.reshape(-1, 1)

        if len(self.points) != len(self.values):
            raise ValueError("Dimension mismatch in points and values")

        if self.start_indices[0] != 0:
            raise ValueError("Array start_indices must start with 0")

        if np.any(np.diff(self.start_indices) < 0):
            raise ValueError("Array start_indices must be non-decreasing")

        if self.start_indices[-1] > len(self.points):
            raise ValueError("Index in start_indices out of bounds")

        # Ensure arguments are in order within each function
        sorted_arguments, sorted_values = self._sort_by_arguments()
        self.points = sorted_arguments
        self.values = sorted_values

        self._sample_range = _get_sample_range_from_data(
            self.start_indices,
            self.points,
        )

        # Default value for sample_range is a list of tuples with
        # the first and last arguments of each curve for each dimension

        if domain_range is None:
            domain_range = _get_domain_range_from_sample_range(
                self._sample_range,
            )

        # Default value for domain_range is a list of tuples with
        # the minimum and maximum value of the arguments for each
        # dimension

        from ..misc.validation import validate_domain_range
        self._domain_range = validate_domain_range(domain_range)

        self.interpolation = interpolation

        super().__init__(
            extrapolation=extrapolation,
            dataset_name=dataset_name,
            argument_names=argument_names,
            coordinate_names=coordinate_names,
            sample_names=sample_names,
        )

    @classmethod
    def _from_dataframe(
        cls,
        dataframe: pandas.DataFrame,
        id_column: str,
        argument_columns: Sequence[str | None],
        coordinate_columns: Sequence[str | None],
        **kwargs: Any,
    ) -> FDataIrregular:
        """Create a FDataIrregular object from a pandas dataframe.

        The pandas dataframe should be in 'long' format: each row
        containing the arguments and values of a given point of the
        dataset, and an identifier which specifies which curve they
        belong to.

        Args:
            dataframe: Pandas dataframe containing the
                irregular functional dataset.
            id_column: Name of the column which contains the information
                about which curve does each each row belong to.
            argument_columns: list of columns where
                the arguments for each dimension of the domain can be found.
            coordinate_columns: list of columns where
                the values for each dimension of the image can be found.
            kwargs: Arguments for the FDataIrregular constructor.

        Returns:
            FDataIrregular: Returns a FDataIrregular object which contains
            the irregular functional data of the dataset.
        """
        # Accept strings but ensure the column names are tuples
        if isinstance(argument_columns, str):
            argument_columns = [argument_columns]

        if isinstance(coordinate_columns, str):
            coordinate_columns = [coordinate_columns]

        # Obtain num functions and num observations from data
        n_measurements = dataframe.shape[0]
        num_functions = dataframe[id_column].nunique()

        # Create data structure of function pointers and coordinates
        start_indices = np.zeros((num_functions, ), dtype=np.uint32)
        points = np.zeros(
            (n_measurements, len(argument_columns)),
        )
        values = np.zeros(
            (n_measurements, len(coordinate_columns)),
        )

        head = 0
        index = 0
        for _, f_values in dataframe.groupby(id_column):
            start_indices[index] = head
            num_values = f_values.shape[0]

            # Insert in order
            f_values = f_values.sort_values(argument_columns)

            new_args = f_values[argument_columns].values
            points[head:head + num_values, :] = new_args

            new_coords = f_values[coordinate_columns].values
            values[head:head + num_values, :] = new_coords

            # Update head and index
            head += num_values
            index += 1

        return cls(
            start_indices,
            points,
            values,
            **kwargs,
        )

    @classmethod
    def from_fdatagrid(
        cls: Type[T],
        f_data: FDataGrid,
        **kwargs,
    ) -> FDataIrregular:
        """Create a FDataIrregular object from a source FDataGrid.

        Args:
            f_data (FDataGrid): FDataGrid object used as source.
            kwargs: Arguments for the FDataIrregular constructor.

        Returns:
            FDataIrregular: FDataIrregular containing the same data
            as the source but with an irregular structure.
        """
        all_points_single_function = _cartesian_product(
            _to_grid_points(f_data.grid_points),
        )
        flat_points = np.tile(
            all_points_single_function, (f_data.n_samples, 1),
        )

        all_values = f_data.data_matrix.reshape(
            (f_data.n_samples, -1, f_data.dim_codomain),
        )
        flat_values = all_values.reshape((-1, f_data.dim_codomain))
        nonnan_all_values = ~np.all(np.isnan(all_values), axis=-1)
        nonnan_flat_values = nonnan_all_values.reshape((-1,))

        values = flat_values[nonnan_flat_values]
        points = flat_points[nonnan_flat_values]

        n_points_per_function = np.sum(nonnan_all_values, axis=-1)
        start_indices = np.concatenate((
            np.zeros(1, np.int32), np.cumsum(n_points_per_function[:-1]),
        ))

        return cls(
            start_indices,
            points,
            values,
            **kwargs,
        )

    def _sort_by_arguments(self) -> Tuple[ArrayLike, ArrayLike]:
        """Sort the arguments lexicographically functionwise.

        Additionally, sort the values accordingly.

        Returns:
            Tuple[ArrayLike, Arraylike]: sorted pair (arguments, values)
        """
        ind = np.repeat(
            range(len(self.start_indices)),
            np.diff(self.start_indices, append=len(self.points)),
        )
        # In order to use lexsort the following manipulations are required:
        # - Transpose the axis, so that the first axis contains the keys.
        # - Flip that axis so that the primary key is last, and they are thus
        #   in last-to-first order.
        sorter = np.lexsort(np.c_[ind, self.points].T[::-1])

        return self.points[sorter], self.values[sorter]

    def round(
        self,
        decimals: int = 0,
        out: Optional[FDataIrregular] = None,
    ) -> FDataIrregular:
        """Evenly round values to the given number of decimals.

        Arguments are not rounded due to possibility of coalescing
        various arguments to the same rounded value.

        .. deprecated:: 0.6
            Use :func:`numpy.round` function instead.

        Args:
            decimals: Number of decimal places to round to.
                If decimals is negative, it specifies the number of
                positions to the left of the decimal point. Defaults to 0.
            out: FDataIrregular where to place the result, if any.

        Returns:
            Returns a FDataIrregular object where all elements
            in its values are rounded.

        """
        # Arguments are not rounded due to possibility of
        # coalescing various arguments to the same rounded value
        rounded_values = self.values.round(decimals=decimals)

        if isinstance(out, FDataIrregular):
            out.values = rounded_values
            return out

        return self.copy(values=rounded_values)

    @property
    def dim_domain(self) -> int:
        return self.points.shape[1]

    @property
    def dim_codomain(self) -> int:
        return self.values.shape[1]

    @property
    def coordinates(self) -> _IrregularCoordinateIterator[T]:
        return _IrregularCoordinateIterator(self)

    @property
    def n_samples(self) -> int:
        return len(self.start_indices)

    @property
    def sample_range(self) -> DomainRange:
        """
        Return the sample range of the function.

        This contains the minimum and maximum values of the grid points in
        each dimension.

        It does not have to be equal to the `domain_range`.
        """
        return self._sample_range

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
            self.to_grid(),  # TODO Create native interpolation for irregular
            eval_points,
            aligned=aligned,
        )

    def derivative(
        self: T,
        order: int = 1,
        method: Optional[Basis] = None,
    ) -> T:
        """Differentiate the FDataIrregular object.

        Args:
            order: Order of the derivative. Defaults to one.
            method (Optional[Basis]): Method used to generate
                the derivatives.

        Returns:
            FDataIrregular with the derivative of the dataset.
        """
        raise NotImplementedError()

    def integrate(
        self: T,
        *,
        domain: DomainRange | None = None,
    ) -> NDArrayFloat:
        """Integrate the FDataIrregular object.

        The integration can only be performed over 1 dimensional domains.

        For a vector valued function the vector of integrals will be
        returned.

        Args:
            domain: tuple with the domain ranges for each dimension of the
                domain. If None, the domain range of the object will be used.

        Returns:
            NumPy array of size (``n_samples``, ``dim_codomain``)
            with the integrated data.

        Examples:
            >>> fdata = FDataIrregular(
            ...     values=[[2, -1], [2, 3], [5, -2], [1, -1], [1, -1]],
            ...     points=[[4], [5], [6], [0], [2]],
            ...     start_indices=[0, 3],
            ... )
            >>> fdata.integrate()
            array([[ 5.,  3.],
                   [ 2., -2.]])
        """
        if self.dim_domain != 1:
            raise NotImplementedError(
                "Integration only implemented for 1D domains.",
            )

        if domain is not None:
            data = self.restrict(domain)
        else:
            data = self

        values_list = np.split(data.values, data.start_indices[1:])
        points_list = np.split(data.points, data.start_indices[1:])
        return np.array([
            scipy.integrate.simpson(y, x=x, axis=0)
            for y, x in zip(values_list, points_list)
        ])

    def check_same_dimensions(self: T, other: T) -> None:
        """Ensure that other FDataIrregular object has compatible dimensions.

        Args:
            other (T): FDataIrregular object to compare dimensions
                with.

        Raises:
            ValueError: Dimension mismatch in coordinates.
            ValueError: Dimension mismatch in arguments.
        """
        if self.dim_codomain != other.dim_codomain:
            raise ValueError("Dimension mismatch in coordinates")
        if self.dim_domain != other.dim_domain:
            raise ValueError("Dimension mismatch in arguments")

    def _get_points_and_values(self: T) -> Tuple[NDArrayFloat, NDArrayFloat]:
        return (self.points, self.values)

    def _get_input_points(self: T) -> GridPoints:
        return self.points  # type: ignore[return-value]

    def _get_common_points_and_values(
        self: T,
    ) -> Tuple[NDArrayFloat, NDArrayFloat]:
        unique_points, counts = (
            np.unique(self.points, axis=0, return_counts=True)
        )
        common_points = unique_points[counts == self.n_samples]

        # Find which points are common to all curves by subtracting each point
        # to each of the common points
        subtraction = self.points[:, np.newaxis, :] - common_points
        is_common_point = np.any(~np.any(subtraction, axis=-1), axis=-1)
        common_points_values = self.values[is_common_point].reshape(
            (self.n_samples, len(common_points), self.dim_codomain),
        )
        return common_points, common_points_values

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
            axis (Optional[int]): Used for compatibility with numpy.
                Must be None or 0.
            out (None): Used for compatibility with numpy.
                Must be None.
            keepdims (bool): Used for compatibility with numpy.
                Must be False.
            skipna (bool): Wether the NaNs are ignored or not.
            min_count: Number of valid (non NaN) data to have in order
                for the a variable to not be NaN when `skipna` is
                `True`.

        Returns:
            FDataIrregular object with only one curve and one value
                representing the sum of all the samples in the original object.
                The points of the new object are the points common to all the
                samples in the original object. Only values present in those
                common points are considered for the sum.
        """
        super().sum(axis=axis, out=out, keepdims=keepdims, skipna=skipna)

        common_points, common_values = self._get_common_points_and_values()

        if len(common_points) == 0:
            raise ValueError("No common points in FDataIrregular object")

        sum_function = np.nansum if skipna else np.sum
        sum_values = sum_function(common_values, axis=0)

        return FDataIrregular(
            start_indices=np.array([0]),
            points=common_points,
            values=sum_values,
            sample_names=(None,),
        )

    def var(self: T, correction: int = 0) -> T:
        """Compute the variance of all the samples.

        Args:
            correction: degrees of freedom adjustment. The divisor used in the
                calculation is `N - correction`, where `N` represents the
                number of elements. Default: `0`.

        Returns:
            FDataIrregular object with only one curve and one value
                representing the pointwise variance of all the samples in the
                original object. The points of the new object are the points
                common to all the samples in the original object.
        """
        # Find all distinct arguments (ordered) and corresponding values
        common_points, common_values = self._get_common_points_and_values()
        var_values = np.var(
            common_values, axis=0, ddof=correction,
        )

        return FDataIrregular(
            start_indices=np.array([0]),
            points=common_points,
            values=var_values,
            sample_names=(None,),
        )

    def cov(
        self: T,
        /,
        correction: int = 0,
    ) -> T:
        """Compute the covariance for a FDataIrregular object.

        Returns:
            FDataIrregular with the covariance function.
        """
        # TODO Implementation to be decided
        raise NotImplementedError()

    def equals(self, other: object) -> bool:
        """Comparison of FDataIrregular objects."""
        if not isinstance(other, FDataIrregular):
            return False

        if not super().equals(other):
            return False

        if not self._eq_elemenwise(other):
            return False

        # Comparison of the domain
        if not np.array_equal(self.domain_range, other.domain_range):
            return False

        if self.interpolation != other.interpolation:
            return False

        return True

    def _eq_elemenwise(self: T, other: T) -> NDArrayBool:
        """Elementwise equality of FDataIrregular."""
        return np.all(
            [
                (self.start_indices == other.start_indices).all(),
                (self.points == other.points).all(),
                (self.values == other.values).all(),
            ],
        )

    def __eq__(self, other: object) -> NDArrayBool:
        return np.array([
            f.equals(o) for f, o in zip(self, other)
        ])

    def _get_op_matrix(  # noqa: WPS212
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
                    (slice(None),)
                    + (np.newaxis,) * (self.values.ndim - 1)
                )

                other_vector = other[other_index]

                # Number of values in each curve
                values_curve = np.diff(
                    self.start_indices, append=len(self.points))

                # Repeat the other value for each curve as many times
                # as values inside the curve
                return np.repeat(other_vector, values_curve).reshape(-1, 1)
            elif other.shape == (
                self.n_samples,
                self.dim_codomain,
            ):
                other_index = (
                    (slice(None),)
                    + (np.newaxis,) * (self.values.ndim - 2)
                    + (slice(None),)
                )

                other_vector = other[other_index]

                # Number of values in each curve
                values_curve = np.diff(
                    self.start_indices, append=len(self.points))

                # Repeat the other value for each curve as many times
                # as values inside the curve
                return np.repeat(other_vector, values_curve, axis=0)

            raise ValueError(
                f"Invalid dimensions in operator between FDataIrregular and "
                f"Numpy array: {other.shape}",
            )

        elif isinstance(other, FDataIrregular):
            # TODO What to do with different argument and value sizes?
            return other.values

        return None

    def __add__(
        self: T,
        other: Union[T, NDArrayFloat, NDArrayInt, float],
    ) -> T:
        values = self._get_op_matrix(other)
        if values is None:
            return NotImplemented

        return self._copy_op(
            other,
            values=self.values + values,
        )

    def __radd__(
        self: T,
        other: Union[T, NDArrayFloat, NDArrayInt, float],
    ) -> T:
        return self.__add__(other)

    def __sub__(
        self: T,
        other: Union[T, NDArrayFloat, NDArrayInt, float],
    ) -> T:
        values = self._get_op_matrix(other)
        if values is None:
            return NotImplemented

        return self._copy_op(
            other,
            values=self.values - values,
        )

    def __rsub__(
        self: T,
        other: Union[T, NDArrayFloat, NDArrayInt, float],
    ) -> T:
        values = self._get_op_matrix(other)
        if values is None:
            return NotImplemented

        return self._copy_op(
            other,
            values=values - self.values,
        )

    def __mul__(
        self: T,
        other: Union[T, NDArrayFloat, NDArrayInt, float],
    ) -> T:
        values = self._get_op_matrix(other)
        if values is None:
            return NotImplemented

        return self._copy_op(
            other,
            values=self.values * values,
        )

    def __rmul__(
        self: T,
        other: Union[T, NDArrayFloat, NDArrayInt, float],
    ) -> T:
        return self.__mul__(other)

    def __truediv__(
        self: T,
        other: Union[T, NDArrayFloat, NDArrayInt, float],
    ) -> T:
        values = self._get_op_matrix(other)
        if values is None:
            return NotImplemented

        return self._copy_op(
            other,
            values=self.values / values,
        )

    def __rtruediv__(
        self: T,
        other: Union[T, NDArrayFloat, NDArrayInt, float],
    ) -> T:
        values = self._get_op_matrix(other)
        if values is None:
            return NotImplemented

        return self._copy_op(
            other,
            values=values / self.values,
        )

    def __neg__(self: T) -> T:
        """Negation of FDataIrregular object."""
        return self.copy(values=-self.values)

    def concatenate(self: T, *others: T, as_coordinates: bool = False) -> T:
        """Join samples from a similar FDataIrregular object.

        Joins samples from another FDataIrregular object if it has the same
        dimensions.

        Args:
            others: Objects to be concatenated.
            as_coordinates (bool): If False concatenates as
                new samples, else, concatenates the other functions as
                new components of the image. Defaults to false.

        Raises:
            NotImplementedError: Not implemented for as_coordinates = True

        Returns:
            T: FDataIrregular object with the samples from the source objects.

        Examples:
            >>> indices = [0, 2]
            >>> arguments = values = np.arange(5.).reshape(-1, 1)
            >>> fd = FDataIrregular(indices, arguments, values)
            >>> arguments_2 = values_2 = np.arange(5, 10).reshape(-1, 1)
            >>> fd_2 = FDataIrregular(indices, arguments_2, values_2)
            >>> fd.concatenate(fd_2)
            FDataIrregular(
                start_indices=array([ 0, 2, 5, 7]),
                points=array([[ 0.],
                    [ 1.],
                    [ 2.],
                    [ 3.],
                    [ 4.],
                    [ 5.],
                    [ 6.],
                    [ 7.],
                    [ 8.],
                    [ 9.]]),
                values=array([[ 0.],
                    [ 1.],
                    [ 2.],
                    [ 3.],
                    [ 4.],
                    [ 5.],
                    [ 6.],
                    [ 7.],
                    [ 8.],
                    [ 9.]]),
                domain_range=((0.0, 9.0),),
                ...)
        """
        # TODO As coordinates
        if as_coordinates:
            raise NotImplementedError(
                "Not implemented for as_coordinates = True",
            )
        # Verify that dimensions are compatible
        assert others, "No objects to concatenate"
        all_objects = (self,) + others
        start_indices_split = []
        total_points = 0
        points_split = []
        values_split = []
        total_sample_names_split = []
        domain_range_split = []
        for x, y in itertools.pairwise(all_objects + (self,)):
            x.check_same_dimensions(y)
            start_indices_split.append(x.start_indices + total_points)
            total_points += len(x.points)
            points_split.append(x.points)
            values_split.append(x.values)
            total_sample_names_split.append(x.sample_names)
            domain_range_split.append(x.domain_range)

        start_indices = np.concatenate(start_indices_split)
        points = np.concatenate(points_split)
        values = np.concatenate(values_split)
        total_sample_names = list(itertools.chain(*total_sample_names_split))
        domain_range_stacked = np.stack(domain_range_split, axis=-1)
        domain_range = np.c_[
            domain_range_stacked[:, 0].min(axis=-1),
            domain_range_stacked[:, 1].max(axis=-1),
        ]

        return self.copy(
            start_indices,
            points,
            values,
            domain_range=domain_range,
            sample_names=total_sample_names,
        )

    def plot(self, *args: Any, **kwargs: Any) -> Figure:
        """Plot the functional data of FDataIrregular with a lines plot.

        Args:
            args: Positional arguments to be passed to the class
                :class:`~skfda.exploratory.visualization.representation.LinearPlotIrregular`.
            kwargs: Keyword arguments to be passed to the class
                :class:`~skfda.exploratory.visualization.representation.LinearPlotIrregular`.

        Returns:
            Figure object in which the graphs are plotted.
        """
        from ..exploratory.visualization.representation import (
            LinearPlotIrregular,
        )

        return LinearPlotIrregular(self, *args, **kwargs).plot()

    def scatter(self, *args: Any, **kwargs: Any) -> Figure:
        """Plot the functional data of FDataIrregular with a scatter plot.

        Args:
            args: Positional arguments to be passed to the class
                :class:`~skfda.exploratory.visualization.representation.ScatterPlotIrregular`.
            kwargs: Keyword arguments to be passed to the class
                :class:`~skfda.exploratory.visualization.representation.ScatterPlotIrregular`.

        Returns:
            Figure object in which the graphs are plotted.
        """
        from ..exploratory.visualization.representation import (
            ScatterPlotIrregular,
        )

        return ScatterPlotIrregular(self, *args, **kwargs).plot()

    def to_basis(
        self,
        basis: Basis,
        *,
        conversion_type: IrregularToBasisConversionType = "function-wise",
        **kwargs: Any,
    ) -> FDataBasis:
        """Return the basis representation of the object.

        Args:
            basis (Basis): basis object in which the functional data are
                going to be represented.
            conversion_type: method to use for the conversion:

                - "function-wise": (default) each curve is converted independently
                    (meaning that only the information of each curve is used
                    for its conversion) with
                    :class:`~skfda.preprocessing.smoothing.BasisSmoother`.
                - "mixed-effects": all curves are converted jointly (this means
                    that the information of all curves is used to convert each
                    one) using the EM algorithm to fit the mixed effects
                    model:
                    :class:`~skfda.representation.conversion.EMMixedEffectsConverter`.
                - "mixed-effects-minimize": all curves are converted jointly
                    using the scipy.optimize.minimize to fit the mixed effects
                    model:
                    :class:`~skfda.representation.conversion.MinimizeMixedEffectsConverter`.
            kwargs: keyword arguments to be passed to FDataBasis.from_data()
                in the case of conversion_type="separately". If conversion_type
                has another value, the keyword arguments are passed to the fit
                method of the
                :class:`~skfda.representation.conversion.MixedEffectsConverter`.

        Raises:
            ValueError: Incorrect domain dimension
            ValueError: Incorrect codomain dimension

        Returns:
            FDataBasis: Basis representation of the funtional data
            object.

        .. jupyter-execute::

            from skfda.datasets import fetch_weather, irregular_sample
            from skfda.representation.basis import FourierBasis
            import matplotlib.pyplot as plt
            fd_temperatures = fetch_weather().data.coordinates[0]
            temp_irregular = irregular_sample(
                fdata=fd_temperatures,
                n_points_per_curve=8,
                random_state=4934755,
            )
            basis = FourierBasis(
                n_basis=5, domain_range=fd_temperatures.domain_range,
            )
            temp_basis_repr = temp_irregular.to_basis(
                basis, conversion_type="mixed-effects",
            )
            fig = plt.figure(figsize=(10, 10))
            for k in range(4):
                axes = plt.subplot(2, 2, k + 1)
                fd_temperatures.plot(axes=axes, alpha=0.05, color="black")
                fd_temperatures[k].plot(
                    axes=axes, color=f"C{k}",
                    label="Original data", linestyle="--",
                )
                temp_basis_repr[k].plot(
                    axes=axes, color=f"C{k}",
                    label="Basis representation",
                )
                temp_irregular[k].scatter(axes=axes, color=f"C{k}")
                plt.legend()
            plt.show()
        """
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

        if conversion_type in ("mixed-effects", "mixed-effects-minimize"):
            from ..representation.conversion import (
                EMMixedEffectsConverter,
                MinimizeMixedEffectsConverter,
            )
            converter_class = (
                EMMixedEffectsConverter if conversion_type == "mixed-effects"
                else MinimizeMixedEffectsConverter
            )
            converter = converter_class(basis)
            return converter.fit_transform(self, **kwargs)

        if conversion_type != "function-wise":
            raise ValueError(f"Invalid conversion type: {conversion_type}")

        from ..preprocessing.smoothing import BasisSmoother
        smoother = BasisSmoother(
            basis=basis,
            **kwargs,
            return_basis=True,
        )

        # Only uses the available values for each curve
        basis_coefficients = [
            smoother.fit_transform(curve).coefficients[0]
            for curve in self
        ]

        return FDataBasis(
            basis,
            basis_coefficients,
            dataset_name=self.dataset_name,
            argument_names=self.argument_names,
            coordinate_names=self.coordinate_names,
            sample_names=self.sample_names,
            extrapolation=self.extrapolation,
        )

    def _to_data_matrix(self) -> tuple[ArrayLike, list[ArrayLike]]:
        """Convert FDataIrregular values to numpy matrix.

        Undefined values in the grid will be represented with np.nan.

        Returns:
            ArrayLike: numpy array with the resulting matrix.
            list: numpy arrays representing grid_points.
        """
        # Find the common grid points
        grid_points = list(map(np.unique, self.points.T))

        unified_matrix = np.full(
            (self.n_samples, *map(len, grid_points), self.dim_codomain), np.nan
        )

        points_pos = tuple(
            np.searchsorted(*arg) for arg in zip(grid_points, self.points.T)
        )

        sample_idx = (
            np.searchsorted(
                self.start_indices, np.arange(len(self.points)), "right"
            )
            - 1
        )

        unified_matrix[(sample_idx,) + points_pos] = self.values

        return unified_matrix, grid_points

    def to_grid(  # noqa: D102
        self: T,
    ) -> FDataGrid:
        """Convert FDataIrregular to FDataGrid.

        Undefined values in the grid will be represented with np.nan.

        Returns:
            FDataGrid: FDataGrid with the irregular functional data.
        """
        data_matrix, grid_points = self._to_data_matrix()

        return FDataGrid(
            data_matrix=data_matrix,
            grid_points=grid_points,
            dataset_name=self.dataset_name,
            argument_names=self.argument_names,
            coordinate_names=self.coordinate_names,
            extrapolation=self.extrapolation,
        )

    def copy(  # noqa: WPS211
        self: T,
        start_indices: Optional[ArrayLike] = None,
        points: Optional[ArrayLike] = None,
        values: Optional[ArrayLike] = None,
        deep: bool = False,  # For Pandas compatibility
        domain_range: Optional[DomainRangeLike] = None,
        dataset_name: Optional[str] = None,
        sample_names: Optional[LabelTupleLike] = None,
        extrapolation: Optional[ExtrapolationLike] = None,
        interpolation: Optional[Evaluator] = None,
        argument_names: Optional[LabelTupleLike] = None,
        coordinate_names: Optional[LabelTupleLike] = None,
    ) -> T:
        """
        Return a copy of the FDataIrregular.

        If an argument is provided the corresponding attribute in the new copy
        is updated.

        """
        if start_indices is None:
            start_indices = self.start_indices

        if points is None:
            points = self.points

        if values is None:
            values = self.values

        if domain_range is None:
            domain_range = self.domain_range

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

        return FDataIrregular(
            start_indices,
            points,
            values,
            domain_range=domain_range,
            dataset_name=dataset_name,
            argument_names=argument_names,
            coordinate_names=coordinate_names,
            sample_names=sample_names,
            extrapolation=extrapolation,
            interpolation=interpolation,
        )

    def restrict(  # noqa: WPS210
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
            T: Restricted function.

        """
        if with_bounds:  # To do
            raise NotImplementedError('Not yet implemented for FDataIrregular')

        from ..misc.validation import validate_domain_range

        npdr = np.broadcast_to(
            validate_domain_range(domain_range),
            (self.dim_domain, 2),
        )

        mask = np.all(
            (npdr[:, 0] <= self.points) & (self.points <= npdr[:, 1]),
            axis=1,
        )

        num_points = _reduceat(np.add, mask, self.start_indices, value_empty=0)
        start_indices = np.r_[[0], num_points[:-1].cumsum()]

        return self.copy(
            start_indices=start_indices,
            points=self.points[mask],
            values=self.values[mask],
            domain_range=npdr,
        )

    def shift(
        self,
        shifts: Union[ArrayLike, float],
        *,
        restrict_domain: bool = False,
        extrapolation: Optional[ExtrapolationLike] = None,
    ) -> FDataIrregular:
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

        Returns:
            Shifted functions.
        """
        raise NotImplementedError()

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
        raise NotImplementedError()

    def __str__(self) -> str:
        """Return str(self)."""
        return (
            f"function indices:    {self.start_indices}\n"
            f"function arguments:    {self.points}\n"
            f"function values:    {self.values}\n"
            f"time range:    {self.domain_range}"
        )

    def __repr__(self) -> str:
        """Return repr(self)."""
        return (
            f"FDataIrregular("  # noqa: WPS221
            f"\nstart_indices={self.start_indices!r},"
            f"\npoints={self.points!r},"
            f"\nvalues={self.values!r},"
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
        required_slices = []
        key = _check_array_key(self.start_indices, key)
        indices = range(self.n_samples)
        required_indices = np.array(indices)[key]
        for i in required_indices:
            next_index = None
            if i + 1 < self.n_samples:
                next_index = self.start_indices[i + 1]
            s = slice(self.start_indices[i], next_index)
            required_slices.append(s)

        arguments = np.concatenate(
            [
                self.points[s]
                for s in required_slices
            ],
        )
        values = np.concatenate(
            [
                self.values[s]
                for s in required_slices
            ],
        )

        chunk_sizes = np.array(
            [
                s.stop - s.start if s.stop is not None
                else len(self.points) - s.start
                for s in required_slices
            ],
        )

        indices = np.concatenate([[0], np.cumsum(chunk_sizes)])[:-1]

        return self.copy(
            start_indices=indices.astype(int),
            points=arguments,
            values=values,
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
                isinstance(i, FDataIrregular)
                and not np.array_equal(
                    i.points,
                    self.points,
                )
            ):
                return NotImplemented

        new_inputs = [
            self._get_op_matrix(input_) for input_ in inputs
        ]

        outputs = kwargs.pop('out', None)
        if outputs:
            new_outputs = [
                o.values if isinstance(o, FDataIrregular)
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

        results = [self.copy(values=r) for r in results]

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
        result.values = np.full(
            (len(indices),) + self.values.shape[1:],
            np.nan,
        )

        positive_mask = indices >= 0
        result.values[positive_mask] = self.values[
            indices[positive_mask]
        ]

        if fill_value is not self.dtype.na_value:
            fill_value_ = fill_value.values[0]
            result.values[~positive_mask] = fill_value_

        return result

    @property
    def dtype(self) -> FDataIrregularDType:
        """The dtype for this extension array, FDataIrregularDType"""
        return FDataIrregularDType(
            start_indices=self.start_indices,
            points=self.points,
            dim_codomain=self.dim_codomain,
            domain_range=self.domain_range,
        )

    @property
    def nbytes(self) -> int:
        """
        The number of bytes needed to store this object in memory.
        """
        array_nbytes = [
            self.start_indices.nbytes,
            self.points.nbytes,
            self.values.nbytes,
        ]
        return sum(array_nbytes)

    def isna(self) -> NDArrayBool:
        """
        Return a 1-D array indicating if each value is missing.

        Returns:
            na_values (NDArrayBool): Positions of NA.
        """
        return np.array([
            np.all(np.isnan(v.values)) for v in self
        ])


class FDataIrregularDType(
    pandas.api.extensions.ExtensionDtype,  # type: ignore[misc]
):
    """DType corresponding to FDataIrregular in Pandas."""

    name = 'FDataIrregular'
    kind = 'O'
    type = FDataIrregular  # noqa: WPS125
    na_value = pandas.NA

    def __init__(
        self,
        start_indices: ArrayLike,
        points: ArrayLike,
        dim_codomain: int,
        domain_range: Optional[DomainRangeLike] = None,
    ) -> None:
        from ..misc.validation import validate_domain_range
        self.start_indices = start_indices
        self.points = points
        self.dim_domain = points.shape[1]

        if domain_range is None:
            sample_range = _get_sample_range_from_data(
                self.start_indices, self.points
            )
            domain_range = _get_domain_range_from_sample_range(sample_range)

        self.domain_range = validate_domain_range(domain_range)
        self.dim_codomain = dim_codomain

    @classmethod
    def construct_array_type(cls) -> Type[FDataIrregular]:  # noqa: D102
        return FDataIrregular

    def _na_repr(self) -> FDataIrregular:

        shape = (
            (len(self.points),)
            + (self.dim_codomain,)
        )

        values = np.full(shape=shape, fill_value=self.na_value)

        return FDataIrregular(
            start_indices=self.start_indices,
            points=self.points,
            values=values,
            domain_range=self.domain_range,
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
        elif not isinstance(other, FDataIrregularDType):
            return False

        return (
            self.start_indices == other.start_indices
            and self.points == other.points
            and self.domain_range == other.domain_range
            and self.dim_codomain == other.dim_codomain
        )

    def __hash__(self) -> int:
        return hash(
            (
                str(self.start_indices),
                str(self.points),
                self.domain_range,
                self.dim_codomain,
            ),
        )


class _IrregularCoordinateIterator(Sequence[T]):
    """Internal class to iterate through the image coordinates."""

    def __init__(self, fdatairregular: T) -> None:
        """Create an iterator through the image coordinates."""
        self._fdatairregular = fdatairregular

    def __getitem__(
        self,
        key: Union[int, slice, NDArrayInt, NDArrayBool],
    ) -> T:
        """Get a specific coordinate."""
        s_key = key
        if isinstance(s_key, int):
            s_key = slice(s_key, s_key + 1)

        coordinate_names = np.array(
            self._fdatairregular.coordinate_names,
        )[s_key]

        coordinate_values = self._fdatairregular.values[..., key]

        return self._fdatairregular.copy(
            values=coordinate_values.reshape(-1, 1),
            coordinate_names=tuple(coordinate_names),
        )

    def __len__(self) -> int:
        """Return the number of coordinates."""
        return self._fdatairregular.dim_codomain
