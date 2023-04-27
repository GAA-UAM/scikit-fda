"""Discretised functional data module.

This module defines a class for representing discretized irregular data,
in which the observations may be made in different grid points in each
data function, and the overall density of the observations may be low

"""
from __future__ import annotations

import numbers
import warnings
from typing import Any, Optional, Sequence, Tuple, Type, TypeVar, Union, cast

import numpy as np
import pandas.api.extensions
from matplotlib.figure import Figure

from .._utils import _check_array_key
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

######################
# Auxiliary functions#
######################


def _get_sample_range_from_data(
    function_indices,
    function_arguments,
    dim_domain,
):
    dim_ranges = []
    for dim in range(dim_domain):
        i = 0
        dim_sample_ranges = []
        for f in function_indices[1:]:
            min_argument = min(
                [function_arguments[j][dim] for j in range(i, f)],
            )
            max_argument = max(
                [function_arguments[j][dim] for j in range(i, f)],
            )
            dim_sample_ranges.append(
                ((min_argument, max_argument)),
            )
            i = f

        min_argument = min(
            [
                function_arguments[i + j][dim]
                for j in range(function_arguments.shape[0] - i)
            ],
        )

        max_argument = max(
            [
                function_arguments[i + j][dim]
                for j in range(function_arguments.shape[0] - i)
            ],
        )

        dim_sample_ranges.append(
            (min_argument, max_argument),
        )
        dim_ranges.append(dim_sample_ranges)

    sample_range = []
    for sample, _ in enumerate(dim_sample_ranges):
        sample_range.append(
            tuple(
                [dim_ranges[d][sample] for d in range(dim_domain)],
            ),
        )

    return sample_range


def _get_domain_range_from_sample_range(
    sample_range,
    dim_domain,
):
    ranges = []
    for dim in range(dim_domain):
        min_argument = min([x[dim][0] for x in sample_range])
        max_argument = max([x[dim][1] for x in sample_range])
        ranges.append((min_argument, max_argument))

    return tuple(ranges)  # domain_range

######################
# FDataIrregular#
######################


class FDataIrregular(FData):  # noqa: WPS214
    r"""Represent discretised functional data of an irregular or sparse nature.

    Class for representing irregular functional data in a compact manner,
    allowing basic operations, representation and conversion to basis format.

    Attributes:
        functional_indices: a unidimensional array which stores the index of
            the functional_values and functional_values arrays where the data
            of each individual curve of the sample begins.
        functional_arguments: an array of every argument of the domain for
            every curve in the sample. Each row contains an observation.
        functional_values: an array of every value of the codomain for
            every curve in the sample. Each row contains an observation.
        domain_range: 2 dimension matrix where each row
            contains the bounds of the interval in which the functional data
            is considered to exist for each one of the axies.
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
        Representation of an irregular functional data object with 2 samples
        representing a function :math:`f : \mathbb{R}\longmapsto\mathbb{R}`,
        with 2 and 3 discretization points respectively.

        >>> indices = [0, 2]
        >>> arguments = [[1], [2], [3], [4], [5]]
        >>> values = [[1], [2], [3], [4], [5]]
        >>> FDataIrregular(indices, arguments, values)
        FDataIrregular(
            function_indices=array([0, 2]),
            function_arguments=array([[1],
                [2],
                [3],
                [4],
                [5]]),
            function_values=array([[1],
                [2],
                [3],
                [4],
                [5]]),
            domain_range=((1.0, 5.0),),
            ...)

        The number of arguments and values must be the same.

        >>> indices = [0,2]
        >>> arguments = np.arange(5).reshape(-1, 1)
        >>> values = np.arange(6).reshape(-1, 1)
        >>> FDataIrregular(indices, arguments, values)
        Traceback (most recent call last):
            ....
        ValueError: Dimension mismatch between function_arguments
        and function_values...

        The indices in function_indices must point to correct rows
        in function_arguments and function_values.

        >>> indices = [0,7]
        >>> arguments = np.arange(5).reshape(-1, 1)
        >>> values = np.arange(5).reshape(-1, 1)
        >>> FDataIrregular(indices, arguments, values)
        Traceback (most recent call last):
            ....
        ValueError: Index in function_indices out of bounds...

        FDataIrregular supports higher dimensional data both in the domain
        and in the codomain (image).

        Representation of a functional data object with 2 samples
        representing a function :math:`f : \mathbb{R}\longmapsto\mathbb{R}^2`.

        >>> indices = [0, 2]
        >>> arguments = [[1], [2], [3], [4], [5]]
        >>> values = [[1, 1], [2, 2], [3, 3], [4, 4], [5, 5]]
        >>> fd = FDataIrregular(indices, arguments, values)
        >>> fd.dim_domain, fd.dim_codomain
        (1, 2)

        Representation of a functional data object with 2 samples
        representing a function :math:`f : \mathbb{R}^2\longmapsto\mathbb{R}`.

        >>> indices = [0, 2]
        >>> arguments = [[1, 1], [2, 2], [3, 3], [4, 4], [5, 5]]
        >>> values = [[1], [2], [3], [4], [5]]
        >>> fd = FDataIrregular(indices, arguments, values)
        >>> fd.dim_domain, fd.dim_codomain
        (2, 1)

    """

    def __init__(  # noqa:  WPS211
        self,
        function_indices: ArrayLike,
        function_arguments: ArrayLike,
        function_values: ArrayLike,
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
        self.function_indices = np.array(function_indices)
        self.function_arguments = np.array(function_arguments)
        if len(self.function_arguments.shape) == 1:
            self.function_arguments = self.function_arguments.reshape(-1, 1)
        self.function_values = np.array(function_values)
        if len(self.function_values.shape) == 1:
            self.function_values = self.function_values.reshape(-1, 1)

        # Set dimensions
        self._dim_domain = self.function_arguments.shape[1]
        self._dim_codomain = self.function_values.shape[1]

        # Set structure to given data
        self.num_functions = self.function_indices.shape[0]

        if self.function_arguments.shape[0] != self.function_values.shape[0]:
            raise ValueError(
                "Dimension mismatch in function_arguments and function_values",
            )

        self.num_observations = self.function_arguments.shape[0]

        if max(self.function_indices) >= self.num_observations:
            raise ValueError("Index in function_indices out of bounds")

        # Ensure arguments are in order within each function
        sorted_arguments, sorted_values = self._sort_by_arguments()
        self.function_arguments = sorted_arguments
        self.function_values = sorted_values

        self._sample_range = _get_sample_range_from_data(
            self.function_indices,
            self.function_arguments,
            self.dim_domain,
        )

        # Default value for sample_range is a list of tuples with
        # the first and last arguments of each curve for each dimension

        if domain_range is None:
            domain_range = _get_domain_range_from_sample_range(
                self._sample_range,
                self.dim_domain,
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
    def from_dataframe(
        cls: Type[T],
        dataframe: pandas.DataFrame,
        id_column: str,
        argument_columns: LabelTupleLike,
        coordinate_columns: LabelTupleLike,
        **kwargs,
    ) -> FDataIrregular:
        """Create a FDataIrregular object from a pandas dataframe.

        The pandas dataframe should be in 'long' format: each row
        containing the arguments and values of a given point of the
        dataset, and an identifier which specifies which curve they
        belong to.

        Args:
            dataframe (pandas.DataFrame): Pandas dataframe containing the
                irregular functional dataset.
            id_column (str): Name of the column which contains the information
                about which curve does each each row belong to.
            argument_columns (LabelTupleLike): list of columns where
                the arguments for each dimension of the domain can be found.
            coordinate_columns (LabelTupleLike): list of columns where
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
        num_observations = dataframe.shape[0]
        num_functions = dataframe[id_column].nunique()

        # Create data structure of function pointers and coordinates
        function_indices = np.zeros((num_functions, ), dtype=np.uint32)
        function_arguments = np.zeros(
            (num_observations, len(argument_columns)),
        )
        function_values = np.zeros(
            (num_observations, len(coordinate_columns)),
        )

        head = 0
        index = 0
        for _, f_values in dataframe.groupby(id_column):
            function_indices[index] = head
            num_values = f_values.shape[0]

            # Insert in order
            f_values = f_values.sort_values(argument_columns)

            new_args = f_values[argument_columns].values
            function_arguments[head:head + num_values, :] = new_args

            new_coords = f_values[coordinate_columns].values
            function_values[head:head + num_values, :] = new_coords

            # Update head and index
            head += num_values
            index += 1

        return cls(
            function_indices,
            function_arguments,
            function_values,
            **kwargs,
        )

    @classmethod
    def from_datagrid(
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
        # Obtain num functions and num observations from data
        num_observations = np.sum(~(np.isnan(f_data.data_matrix).all(axis=-1)))
        num_functions = f_data.data_matrix.shape[0]

        # Create data structure of function pointers and coordinates
        function_indices = np.zeros((num_functions, ), dtype=np.uint32)
        function_arguments = np.zeros(
            (num_observations, f_data.dim_domain),
        )
        function_values = np.zeros(
            (num_observations, f_data.dim_codomain),
        )

        # Find all the combinations of grid points and indices
        from itertools import product
        grid_point_indexes = [
            np.indices(np.array(gp).shape)[0]
            for gp in f_data.grid_points
        ]
        combinations = list(product(*f_data.grid_points))
        index_combinations = list(product(*grid_point_indexes))

        head = 0
        for i in range(num_functions):
            function_indices[i] = head
            num_values = 0

            for g_index, g in enumerate(index_combinations):
                if np.all(np.isnan(f_data.data_matrix[(i,) + g])):
                    continue

                arg = combinations[g_index]
                value = f_data.data_matrix[(i, ) + g]

                function_arguments[head + num_values, :] = arg
                function_values[head + num_values, :] = value

                num_values += 1

            head += num_values

        return cls(
            function_indices,
            function_arguments,
            function_values,
            **kwargs,
        )

    def _sort_by_arguments(self) -> Tuple[ArrayLike, ArrayLike]:
        """Sort the arguments lexicographically functionwise.

        Additionally, sort the values accordingly.

        Returns:
            Tuple[ArrayLike, Arraylike]: sorted pair (arguments, values)
        """
        indices_start_end = np.append(
            self.function_indices,
            self.num_observations,
        )

        slices = list(zip(indices_start_end, indices_start_end[1:]))
        slice_args = [self.function_arguments[slice(*s)] for s in slices]
        slice_values = [self.function_values[slice(*s)] for s in slices]

        # Sort lexicographically, first to last dimension
        sorting_masks = [
            np.lexsort(np.flip(f_args, axis=1).T)
            for f_args in slice_args
        ]

        sorted_args = [
            slice_args[i][mask]
            for i, mask in enumerate(sorting_masks)
        ]

        sorted_values = [
            slice_values[i][mask]
            for i, mask in enumerate(sorting_masks)
        ]

        return np.concatenate(sorted_args), np.concatenate(sorted_values)

    def round(
        self,
        decimals: int = 0,
        out: Optional[FDataIrregular] = None,
    ) -> FDataIrregular:
        """Evenly round function_values to the given number of decimals.

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
            in its function_values are rounded.

        """
        # Arguments are not rounded due to possibility of
        # coalescing various arguments to the same rounded value
        rounded_values = self.function_values.round(decimals=decimals)

        if out is not None and isinstance(out, FDataIrregular):
            out.function_indices = self.function_indices
            out.function_values = rounded_values

            return out

        return self.copy(
            function_values=rounded_values,
        )

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
        return self._dim_domain

    @property
    def dim_codomain(self) -> int:
        return self._dim_codomain

    @property
    def coordinates(self: T) -> _IrregularCoordinateIterator[T]:
        return _IrregularCoordinateIterator(self)

    @property
    def n_samples(self) -> int:
        return self.num_functions

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
        pass

    def integrate(
        self: T,
        domain: Optional[DomainRange] = None,
    ) -> NDArrayFloat:
        """Integrate the FDataIrregular object.

        Args:
            domain (Optional[DomainRange]): tuple with
                the domain ranges for each dimension
                of the domain

        Returns:
            FDataIrregular with the integral.
        """
        pass

    def check_same_dimensions(self: T, other: T) -> None:
        """Ensure that other FDataIrregular object ahs compatible dimensions.

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
            T: FDataIrregular object with only one curve and one value
            representing the sum of all the samples in the original object.
        """
        super().sum(axis=axis, out=out, keepdims=keepdims, skipna=skipna)

        data = (
            np.nansum(self.function_values, axis=0, keepdims=True) if skipna
            else np.sum(self.function_values, axis=0, keepdims=True)
        )

        return FDataIrregular(
            function_indices=np.array([0]),
            function_arguments=np.zeros((1, self.dim_domain)),
            function_values=data,
            sample_names=("sum",),
        )

    def mean(self: T) -> T:
        """Compute the mean pointwise for a sparse dataset.

        Note that, for irregular data, points may be represented in few
        or even an only curve.

        Returns:
            A FDataIrregular object with just one sample representing the
            mean of all curves the across each value.
        """
        # Find all distinct arguments (ordered) and corresponding values
        distinct_args = np.unique(np.matrix.flatten(self.function_arguments))
        values = [
            np.matrix.flatten(self.function_values[
                np.where(self.function_arguments == arg)[0]
            ])
            for arg in distinct_args
        ]

        # Obtain mean of all available values for each argument point
        means = np.array([np.mean(value) for value in values])

        # Create a FDataIrregular object with only 1 curve, the mean curve
        return FDataIrregular(
            function_indices=np.array([0]),
            function_arguments=distinct_args.reshape(-1, 1),
            function_values=means.reshape(-1, 1),
            sample_names=("mean",),
        )

    def var(self: T) -> T:
        """Compute the variance pointwise for a sparse dataset.

        Note that, for irregular data, points may be represented in few
        or even an only curve.

        Returns:
            A FDataIrregular object with just one sample representing the
            variance of all curves the across each value.

        """
        # Find all distinct arguments (ordered) and corresponding values
        distinct_args = np.unique(np.matrix.flatten(self.function_arguments))
        values = [
            np.matrix.flatten(self.function_values[
                np.where(self.function_arguments == arg)[0]
            ])
            for arg in distinct_args
        ]

        # Obtain variance of all available values for each argument point
        variances = np.array([np.var(value) for value in values])

        # Create a FDataIrregular object with only 1 curve, the variance curve
        return FDataIrregular(
            function_indices=np.array([0]),
            function_arguments=distinct_args.reshape(-1, 1),
            function_values=variances.reshape(-1, 1),
            sample_names=("var",),
        )

    def cov(self: T) -> T:
        """Compute the covariance for a FDataIrregular object.

        Returns:
            FDataIrregular with the covariance function.
        """
        # TODO Implementation to be decided
        pass

    def equals(self, other: object) -> bool:
        """Comparison of FDataIrregular objects."""
        other = cast(FDataIrregular, other)

        if not self._eq_elemenwise(other):
            return False

        # Comparison of the domain
        if not np.array_equal(self.domain_range, other.domain_range):
            return False

        # TODO extrapolation when implemented

        if self.interpolation != other.interpolation:
            return False

        return super().equals(other)

    def _eq_elemenwise(self: T, other: T) -> NDArrayBool:
        """Elementwise equality of FDataIrregular."""
        return np.all(
            [
                (self.function_indices == other.function_indices).all(),
                (self.function_arguments == other.function_arguments).all(),
                (self.function_values == other.function_values).all(),
            ],
        )

    def __eq__(self, other: object) -> NDArrayBool:
        return self.equals(other)

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
                    (slice(None),) + (np.newaxis,)
                    * (self.function_values.ndim - 1)
                )

                other_vector = other[other_index]

                # Must expand for the number of values in each curve
                values_after = np.concatenate(
                    (
                        self.function_indices,
                        np.array([self.num_observations]),
                    ),
                )

                values_before = np.concatenate(
                    (
                        np.array([0]),
                        self.function_indices,
                    ),
                )

                values_curve = (values_after - values_before)[1:]

                # Repeat the other value for each curve as many times
                # as values inside the curve
                return np.repeat(other_vector, values_curve).reshape(-1, 1)
            elif other.shape == (
                self.n_samples,
                self.dim_codomain,
            ):
                other_index = (
                    (slice(None),) + (np.newaxis,)
                    * (self.function_values.ndim - 2)
                    + (slice(None),)
                )

                other_vector = other[other_index]

                # Must expand for the number of values in each curve
                values_after = np.concatenate(
                    (
                        self.function_indices,
                        np.array([self.num_observations]),
                    ),
                )

                values_before = np.concatenate(
                    (
                        np.array([0]),
                        self.function_indices,
                    ),
                )

                values_curve = (values_after - values_before)[1:]

                # Repeat the other value for each curve as many times
                # as values inside the curve
                return np.repeat(other_vector, values_curve, axis=0)

            raise ValueError(
                f"Invalid dimensions in operator between FDataIrregular "
                f"and Numpy array: {other.shape}",
            )

        elif isinstance(other, FDataIrregular):
            # TODO What to do with different argument and value sizes?
            return other.function_values

        return None

    def __add__(
        self: T,
        other: Union[T, NDArrayFloat, NDArrayInt, float],
    ) -> T:
        function_values = self._get_op_matrix(other)
        if function_values is None:
            return NotImplemented

        return self._copy_op(
            other,
            function_values=self.function_values + function_values,
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
        function_values = self._get_op_matrix(other)
        if function_values is None:
            return NotImplemented

        return self._copy_op(
            other,
            function_values=self.function_values - function_values,
        )

    def __rsub__(
        self: T,
        other: Union[T, NDArrayFloat, NDArrayInt, float],
    ) -> T:
        function_values = self._get_op_matrix(other)
        if function_values is None:
            return NotImplemented

        return self._copy_op(
            other,
            function_values=function_values - self.function_values,
        )

    def __mul__(
        self: T,
        other: Union[T, NDArrayFloat, NDArrayInt, float],
    ) -> T:
        function_values = self._get_op_matrix(other)
        if function_values is None:
            return NotImplemented

        return self._copy_op(
            other,
            function_values=self.function_values * function_values,
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
        function_values = self._get_op_matrix(other)
        if function_values is None:
            return NotImplemented

        return self._copy_op(
            other,
            function_values=self.function_values / function_values,
        )

    def __rtruediv__(
        self: T,
        other: Union[T, NDArrayFloat, NDArrayInt, float],
    ) -> T:
        function_values = self._get_op_matrix(other)
        if function_values is None:
            return NotImplemented

        return self._copy_op(
            other,
            function_values=function_values / self.function_values,
        )

    def __neg__(self: T) -> T:
        """Negation of FDataIrregular object."""
        return self.copy(function_values=-self.function_values)

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
            >>> arguments = values = np.arange(5).reshape(-1, 1)
            >>> fd = FDataIrregular(indices, arguments, values)
            >>> arguments_2 = values_2 = np.arange(5, 10).reshape(-1, 1)
            >>> fd_2 = FDataIrregular(indices, arguments_2, values_2)
            >>> fd.concatenate(fd_2)
            FDataIrregular(
                function_indices=array([0, 2, 5, 7], dtype=uint32),
                function_arguments=array([[0.],
                    [1.],
                    [2.],
                    [3.],
                    [4.],
                    [5.],
                    [6.],
                    [7.],
                    [8.],
                    [9.]]),
                function_values=array([[0.],
                    [1.],
                    [2.],
                    [3.],
                    [4.],
                    [5.],
                    [6.],
                    [7.],
                    [8.],
                    [9.]]),
                domain_range=((0.0, 9.0),),
                ...)
        """
        # TODO As coordinates
        if as_coordinates:
            raise NotImplementedError(
                "Not implemented for as_coordinates = True",
            )
        # Verify that dimensions are compatible
        assert len(others) > 0, "No objects to concatenate"
        self.check_same_dimensions(others[0])
        if len(others) > 1:
            for x, y in zip(others, others[1:]):
                x.check_same_dimensions(y)

        # Allocate all required memory
        total_functions = self.num_functions + sum(
            [
                o.num_functions
                for o in others
            ],
        )
        total_values = self.num_observations + sum(
            [
                o.num_observations
                for o in others
            ],
        )
        total_sample_names = []
        function_indices = np.zeros((total_functions, ), dtype=np.uint32)
        function_args = np.zeros(
            (total_values, self.dim_domain),
        )
        function_values = np.zeros(
            (total_values, self.dim_codomain),
        )
        index = 0
        head = 0

        # Add samples sequentially
        for f_data in [self] + list(others):
            function_indices[
                index:index + f_data.num_functions
            ] = f_data.function_indices
            function_args[
                head:head + f_data.num_observations
            ] = f_data.function_arguments
            function_values[
                head:head + f_data.num_observations
            ] = f_data.function_values
            # Adjust pointers to the concatenated array
            function_indices[index:index + f_data.num_functions] += head
            index += f_data.num_functions
            head += f_data.num_observations
            total_sample_names = total_sample_names + list(f_data.sample_names)

        # Check domain range
        domain_range = [list(r) for r in self.domain_range]
        for dim in range(self.dim_domain):
            dim_max = np.max(function_args[:, dim])
            dim_min = np.min(function_args[:, dim])

            if dim_max > self.domain_range[dim][1]:
                domain_range[dim][1] = dim_max
            if dim_min < self.domain_range[dim][0]:
                domain_range[dim][0] = dim_min

        return self.copy(
            function_indices,
            function_args,
            function_values,
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

    def to_basis(self, basis: Basis, **kwargs: Any) -> FDataBasis:
        """Return the basis representation of the object.

        Args:
            basis (Basis): basis object in which the functional data are
                going to be represented.
            kwargs: keyword arguments to be passed to
                FDataBasis.from_data().

        Raises:
            ValueError: Incorrect domain dimension
            ValueError: Incorrect codomain dimension

        Returns:
            FDataBasis: Basis representation of the funtional data
            object.
        """
        from ..preprocessing.smoothing import IrregularBasisSmoother

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

        smoother = IrregularBasisSmoother(
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

    def to_matrix(self) -> ArrayLike:
        """Convert FDataIrregular values to numpy matrix.

        Undefined values in the grid will be represented with np.nan.

        Returns:
            ArrayLike: numpy array with the resulting matrix.
        """
        # Find the common grid points
        grid_points = [
            np.unique(self.function_arguments[:, dim])
            for dim in range(self.dim_domain)
        ]

        unified_matrix = np.empty(
            (
                self.n_samples,
                *[len(gp) for gp in grid_points],
                self.dim_codomain,
            ),
        )
        unified_matrix.fill(np.nan)

        # Fill with each function
        next_indices = np.append(
            self.function_indices,
            self.num_observations,
        )

        for i, index in enumerate(self.function_indices):
            for j in range(index, next_indices[i + 1]):
                arg = self.function_arguments[j]
                val = self.function_values[j]
                pos = [
                    np.where(gp == arg[dim])[0][0]
                    for dim, gp in enumerate(grid_points)
                ]
                unified_matrix[(i,) + tuple(pos)] = val

        return unified_matrix, grid_points

    def to_grid(  # noqa: D102
        self: T,
    ) -> FDataGrid:
        """Convert FDataIrregular to FDataGrid.

        Undefined values in the grid will be represented with np.nan.

        Returns:
            FDataGrid: FDataGrid with the irregular functional data.
        """
        data_matrix, grid_points = self.to_matrix()

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
        function_indices: Optional[ArrayLike] = None,
        function_arguments: Optional[ArrayLike] = None,
        function_values: Optional[ArrayLike] = None,
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
        if function_indices is None:
            function_indices = self.function_indices

        if function_arguments is None:
            function_arguments = self.function_arguments

        if function_values is None:
            function_values = self.function_values

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
            function_indices,
            function_arguments,
            function_values,
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
    ) -> T:
        """
        Restrict the functions to a new domain range.

        Args:
            domain_range: New domain range.

        Returns:
            T: Restricted function.

        """
        from ..misc.validation import validate_domain_range

        domain_range = validate_domain_range(domain_range)
        assert all(
            c <= a < b <= d  # noqa: WPS228
            for ((a, b), (c, d)) in zip(domain_range, self.domain_range)
        )

        head = 0
        indices = []
        arguments = []
        values = []
        sample_names = []

        # Eliminate points outside the new range.
        # Must also modify function indices to point to new array
        i = -1
        for i, index in enumerate(self.function_indices[1:]):
            prev_index = self.function_indices[i]
            s = slice(prev_index, index)
            masks = set()
            for dr in domain_range:
                dr_start, dr_end = dr
                select_mask = np.where(
                    (
                        (dr_start <= self.function_arguments[s])
                        & (self.function_arguments[s] <= dr_end)
                    ),
                )

                # Must be union, it is valid if it is in any interval
                masks = masks.union(set(select_mask[0]))

            # TODO Keep functions with no values?
            masks = list(masks)
            if len(masks) > 1:
                indices.append(head)
                arguments.append(self.function_arguments[s][masks, :])
                values.append(self.function_values[s][masks, :])
                sample_names.append(self.sample_names[i])
                head += len(masks)

        # Last index
        i += 1
        prev_index = self.function_indices[i]
        s = slice(prev_index, None)
        masks = set()
        for dr in domain_range:
            dr_start, dr_end = dr
            select_mask = np.where(
                (
                    (dr_start <= self.function_arguments[s])
                    & (self.function_arguments[s] <= dr_end)
                ),
            )

            # Must be union, it is valid if it is in any interval
            masks = masks.union(set(select_mask[0]))

        # TODO Keep functions with no values?
        masks = list(masks)
        if len(masks) > 0:
            indices.append(head)
            arguments.append(self.function_arguments[s][masks, :])
            values.append(self.function_values[s][masks, :])
            sample_names.append(self.sample_names[i])
            head += len(masks)

        function_indices = np.array(indices)
        function_arguments = np.concatenate(arguments)
        function_values = np.concatenate(values)

        return self.copy(
            function_indices=function_indices,
            function_arguments=function_arguments,
            function_values=function_values,
            sample_names=sample_names,
            domain_range=domain_range,
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
        # TODO build based in above
        pass

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
        # TODO Is this possible with this structure?
        pass

    def __str__(self) -> str:
        """Return str(self)."""
        return (
            f"function indices:    {self.function_indices}\n"
            f"function arguments:    {self.function_arguments}\n"
            f"function values:    {self.function_values}\n"
            f"time range:    {self.domain_range}"
        )

    def __repr__(self) -> str:
        """Return repr(self)."""
        return (
            f"FDataIrregular("  # noqa: WPS221
            f"\nfunction_indices={self.function_indices!r},"
            f"\nfunction_arguments={self.function_arguments!r},"
            f"\nfunction_values={self.function_values!r},"
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
        key = _check_array_key(self.function_indices, key)
        indices = range(self.num_functions)
        required_indices = indices[key]
        for i in required_indices:
            next_index = None
            if i + 1 < self.num_functions:
                next_index = self.function_indices[i + 1]
            s = slice(self.function_indices[i], next_index)
            required_slices.append(s)

        arguments = np.concatenate(
            [
                self.function_arguments[s]
                for s in required_slices
            ],
        )
        values = np.concatenate(
            [
                self.function_values[s]
                for s in required_slices
            ],
        )

        chunk_sizes = np.array(
            [
                s.stop - s.start if s.stop is not None
                else self.num_observations - s.start
                for s in required_slices
            ],
        )

        indices = np.cumsum(chunk_sizes) - chunk_sizes[0]

        return self.copy(
            function_indices=indices.astype(int),
            function_arguments=arguments,
            function_values=values,
            sample_names=self.sample_names[key],
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
                    i.function_arguments,
                    self.function_arguments,
                )
            ):
                return NotImplemented

        new_inputs = [
            self._get_op_matrix(input_) for input_ in inputs
        ]

        outputs = kwargs.pop('out', None)
        if outputs:
            new_outputs = [
                o.function_values if isinstance(o, FDataIrregular)
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

        results = [self.copy(function_values=r) for r in results]

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
        result.function_values = np.full(
            (len(indices),) + self.function_values.shape[1:],
            np.nan,
        )

        positive_mask = indices >= 0
        result.function_values[positive_mask] = self.function_values[
            indices[positive_mask]
        ]

        if fill_value is not self.dtype.na_value:
            fill_value_ = fill_value.function_values[0]
            result.function_values[~positive_mask] = fill_value_

        return result

    @property
    def dtype(self) -> FDataIrregularDType:
        """The dtype for this extension array, FDataIrregularDType"""
        return FDataIrregularDType(
            function_indices=self.function_indices,
            function_arguments=self.function_arguments,
            dim_codomain=self.dim_codomain,
            domain_range=self.domain_range,
        )

    @property
    def nbytes(self) -> int:
        """
        The number of bytes needed to store this object in memory.
        """
        array_nbytes = [
            self.function_indices.nbytes,
            self.function_arguments.nbytes,
            self.function_values,
        ]
        return sum(array_nbytes)

    def isna(self) -> NDArrayBool:
        """
        Return a 1-D array indicating if each value is missing.

        Returns:
            na_values (NDArrayBool): Positions of NA.
        """
        return np.all(  # type: ignore[no-any-return]
            np.isnan(self.function_values),
            axis=tuple(range(1, self.function_values.ndim)),
        )


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
        function_indices: ArrayLike,
        function_arguments: ArrayLike,
        dim_codomain: int,
        domain_range: Optional[DomainRangeLike] = None,
    ) -> None:
        from ..misc.validation import validate_domain_range
        self.function_indices = function_indices
        self.function_arguments = function_arguments
        self.dim_domain = function_arguments.shape[1]
        self.num_observations = len(function_arguments)

        if domain_range is None:
            sample_range = _get_sample_range_from_data(
                self.function_indices,
                self.function_arguments,
                self.dim_domain,
            )
            domain_range = _get_domain_range_from_sample_range(
                sample_range,
                self.dim_domain,
            )

        self.domain_range = validate_domain_range(domain_range)
        self.dim_codomain = dim_codomain

    @classmethod
    def construct_array_type(cls) -> Type[FDataIrregular]:  # noqa: D102
        return FDataIrregular

    def _na_repr(self) -> FDataIrregular:

        shape = (
            (self.num_observations,)
            + (self.dim_codomain,)
        )

        function_values = np.full(shape=shape, fill_value=self.na_value)

        return FDataIrregular(
            function_indices=self.function_indices,
            function_arguments=self.function_arguments,
            function_values=function_values,
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
            self.function_indices == other.function_indices
            and self.function_arguments == other.function_arguments
            and self.domain_range == other.domain_range
            and self.dim_codomain == other.dim_codomain
        )

    def __hash__(self) -> int:
        return hash(
            (
                str(self.function_indices),
                str(self.function_arguments),
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

        coordinate_values = self._fdatairregular.function_values[..., key]

        return self._fdatairregular.copy(
            function_values=coordinate_values.reshape(-1, 1),
            coordinate_names=tuple(coordinate_names),
        )

    def __len__(self) -> int:
        """Return the number of coordinates."""
        return self._fdatairregular.dim_codomain
