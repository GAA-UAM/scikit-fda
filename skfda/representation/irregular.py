"""Discretised functional data module.

This module defines a class for representing discretized irregular data,
in which the observations may be made in different grid points in each
data function, and the overall density of the observations may be low

"""
from __future__ import annotations

import copy
import numbers
import warnings
from typing import (
    TYPE_CHECKING,
    Any,
    Optional,
    Sequence,
    Type,
    TypeVar,
    Union,
    List,
    Tuple,
    cast,
)

import numpy as np
import pandas.api.extensions
import scipy.stats.mstats
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
from .grid import FDataGrid, FDataGridDType
from .evaluator import Evaluator
from .extrapolation import ExtrapolationLike
from .interpolation import SplineInterpolation

if TYPE_CHECKING:
    from .basis import Basis, FDataBasis

T = TypeVar("T", bound='FDataIrregular')
        

class FDataIrregular(FData):  # noqa: WPS214
    # TODO Docstring
   
    def __init__(
        self,
        function_indices: ArrayLike,
        function_arguments: ArrayLike,
        function_values: ArrayLike,
        *,
        dim_domain: Optional[int] = 1,
        dim_codomain: Optional[int] = 1,
        domain_range: Optional[DomainRangeLike] = None,
        dataset_name: Optional[str] = None,
        sample_names: Optional[LabelTupleLike] = None,
        extrapolation: Optional[ExtrapolationLike] = None,
        interpolation: Optional[Evaluator] = None,
        argument_names: Optional[LabelTupleLike] = None,
        coordinate_names: Optional[LabelTupleLike] = None
    ):
        """Construct a FDataIrregular object."""
            
        # Set dimensions
        self._dim_domain = dim_domain
        self._dim_codomain = dim_codomain
        
        # Set structure to given data
        self.num_functions = function_indices.shape[0]
        
        assert function_arguments.shape[0] == function_values.shape[0]
        self.num_observations = function_arguments.shape[0]
        
        self.set_function_indices(function_indices)
        self.set_function_arguments(function_arguments)
        self.set_function_values(function_values)
        
        # TODO Fix for higher dimensions
        i = 0
        self._sample_range = list()
        for f in self.function_indices[1:]:
            self._sample_range.append(tuple((self.function_arguments[i][0], 
                                      self.function_arguments[f-1][0])))
            i = f
        self._sample_range.append((self.function_arguments[i][0], 
                                   self.function_arguments[-1][0]))
        
        from ..misc.validation import validate_domain_range
        if domain_range is None:
            domain_range = self.sample_range
            # Default value for domain_range is a list of tuples with
            # the first and last arguments of each curve

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
        **kwargs
        ) -> FDataIrregular:
        
        # Accept strings but ensure the column names are tuples
        _is_str = isinstance(argument_columns, str)
        argument_columns = [argument_columns] if _is_str else \
            argument_columns
        
        _is_str = isinstance(coordinate_columns, str)
        coordinate_columns = [coordinate_columns]  if _is_str else \
            coordinate_columns
        
        # Obtain num functions and num observations from data
        num_observations = dataframe.shape[0]
        num_functions = dataframe[id_column].nunique()
        
        # Create data structure of function pointers and coordinates
        function_indices = np.zeros((num_functions, ), 
                                    dtype=np.uint32)
        function_arguments = np.zeros((num_observations, 
                                       len(argument_columns)))
        function_values = np.zeros((num_observations,
                                    len(coordinate_columns)))
        
        head = 0
        index = 0
        for _, f_values in dataframe.groupby(id_column):
            function_indices[index] = head
            num_values = f_values.shape[0]
            
            # Insert in order
            f_values = f_values.sort_values(argument_columns)
            
            new_args = f_values[argument_columns].values
            function_arguments[head:head+num_values, :] = new_args
            
            new_coords = f_values[coordinate_columns].values
            function_values[head:head+num_values, :] = new_coords
            
            # Update head and index
            head += num_values
            index += 1
        
        return cls(
            function_indices, 
            function_arguments, 
            function_values, 
            **kwargs
            )
        
    @classmethod
    def from_datagrid(
        cls: Type[T],
        f_data: FDataGrid,
        **kwargs
        ) -> FDataIrregular:
        
        # Obtain num functions and num observations from data
        num_observations = np.sum(~np.isnan(f_data.data_matrix))
        num_functions = f_data.data_matrix.shape[0]

        # Create data structure of function pointers and coordinates
        function_indices = np.zeros((num_functions, ), 
                                    dtype=np.uint32)
        function_arguments = np.zeros((num_observations, 
                                       f_data.dim_domain))
        function_values = np.zeros((num_observations, 
                                    f_data.dim_codomain))

        head = 0
        for i in range(num_functions):
            function_indices[i] = head
            num_values = 0

            for j in range(f_data.data_matrix.shape[1]):
                if np.isnan(f_data.data_matrix[i][j]):
                    continue
                            
                arg = [f_data.grid_points[dim][j] for dim 
                       in range(f_data.dim_domain)]
                function_arguments[head+num_values, :] = arg
                             
                value = [f_data.data_matrix[i,j,dim] for dim 
                         in range(f_data.dim_codomain)]
                function_values[head+num_values, :] = value

                num_values += 1
                              
            head += num_values
                   
        return cls(
            function_indices, 
            function_arguments, 
            function_values, 
            **kwargs
            )
        
    def set_function_indices(self, function_indices) -> ArrayLike:
        self.function_indices = function_indices.copy()
    
    def set_function_arguments(self, function_arguments) -> ArrayLike:
        self.function_arguments = function_arguments.copy()
        
    def set_function_values(self, function_values) -> ArrayLike:
        self.function_values = function_values.copy()
        
    def round(
        self,
        decimals: int = 0,
        out: Optional[FDataIrregular] = None,
    ) -> FDataIrregular:
        # Arguments are not rounded due to possibility of
        # coalescing various arguments to the same rounded value
        rounded_values = self.function_values.round(decimals=decimals)
        
        if out is not None and isinstance(out, FDataIrregular):
            out.function_indices = self.function_indices
            out.function_values = rounded_values
            
            return out
        
        return self.copy(
            function_values=rounded_values
            )

    @property
    def sample_points(self) -> GridPoints:
        warnings.warn(
            "Parameter sample_points is deprecated. Use the " \
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

    #TODO Remove CoordinateIterator in an appropiate way
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
        *,
        order: int = 1,
        method: Optional[Basis] = None,
    ) -> T:
        pass

    def integrate(
        self: T,
        *,
        domain: Optional[DomainRange] = None,
    ) -> NDArrayFloat:
        pass

    def _check_same_dimensions(self: T, other: T) -> None:
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
        pass
    
    def mean(self: T) -> T:
        pass
    
    def var(self: T) -> T:
        pass

    def cov(self: T) -> T:
        pass

    def gmean(self: T) -> T:
        return FDataIrregular(
            function_indices=np.array([0]),
            function_arguments=np.array(np.zeros((1, self.dim_domain))),
            function_values=scipy.stats.mstats.gmean(self.function_values, 0),
            sample_names=("geometric mean",),
        )

    def equals(self, other: object) -> bool:
        """Comparison of FDataIrregular objects."""
        if not super().equals(other):
            return False

        other = cast(FDataIrregular, other)

        if not self._eq_elemenwise(other):
            return False

        # Comparison of the domain
        if not np.array_equal(self.domain_range, other.domain_range):
            return False

        # TODO extrapolation when implemented
        
        if self.interpolation != other.interpolation:
            return False
        
        return True

    def _eq_elemenwise(self: T, other: T) -> NDArrayBool:
        """Elementwise equality of FDataIrregular."""
        return np.all(
            [(self.function_indices == other.function_indices).all(),
             (self.function_arguments == other.function_arguments).all(),
             (self.function_values == other.function_values).all()]
        )

    def _get_op_matrix(
        self,
        other: Union[T, NDArrayFloat, NDArrayInt, float],
    ) -> Union[None, float, NDArrayFloat, NDArrayInt]:
        pass

    def __add__(
        self: T,
        other: Union[T, NDArrayFloat, NDArrayInt, float],
    ) -> T:
        pass

    def __radd__(
        self: T,
        other: Union[T, NDArrayFloat, NDArrayInt, float],
    ) -> T:
        pass

    def __sub__(
        self: T,
        other: Union[T, NDArrayFloat, NDArrayInt, float],
    ) -> T:
        pass

    def __rsub__(
        self: T,
        other: Union[T, NDArrayFloat, NDArrayInt, float],
    ) -> T:
        pass

    def __mul__(
        self: T,
        other: Union[T, NDArrayFloat, NDArrayInt, float],
    ) -> T:
        pass

    def __rmul__(
        self: T,
        other: Union[T, NDArrayFloat, NDArrayInt, float],
    ) -> T:
        pass

    def __truediv__(
        self: T,
        other: Union[T, NDArrayFloat, NDArrayInt, float],
    ) -> T:
        pass

    def __rtruediv__(
        self: T,
        other: Union[T, NDArrayFloat, NDArrayInt, float],
    ) -> T:
        pass

    def __neg__(self: T) -> T:
        pass

    def concatenate(self: T, *others: T, as_coordinates: bool = False) -> T:
        pass
    
    def plot(self, *args: Any, **kwargs: Any) -> Figure:
        from ..exploratory.visualization.representation import LinearPlotIrregular

        return LinearPlotIrregular(self, *args, **kwargs).plot()

    def scatter(self, *args: Any, **kwargs: Any) -> Figure:
        from ..exploratory.visualization.representation import ScatterPlotIrregular

        return ScatterPlotIrregular(self, *args, **kwargs).plot()

    def to_basis(self, basis: Basis, **kwargs: Any) -> FDataBasis:
        pass
    
    def to_matrix(self, **kwargs: Any) -> ArrayLike:
        # Convert FDataIrregular to matrix of all points
        # with NaN in undefined values
        
        if self.dim_domain > 1:
            warnings.warn(f"Not implemented for domain dimension > 1, \
                          currently {self.dim_domain}")
        
        # Find the grid points and values for each function
        grid_points = []
        evaluated_points = []
        for index_start, index_end in zip(list(self.function_indices), 
                                          list(self.function_indices[1:])):
            grid_points.append(
                [x[0] for x in self.function_arguments[index_start:index_end]])
            evaluated_points.append(
                self.function_values[index_start:index_end])
            
        # Dont forget to add the last one
        grid_points.append([x[0] for x in self.function_arguments[index_end:]])
        evaluated_points.append(self.function_values[index_end:])
        
        # Aggregate into a complete data matrix
        from functools import reduce
        unified_grid_points = reduce(
            lambda x, y: set(list(y)).union(list(x)),
            grid_points,
            )
        
        unified_grid_points = sorted(unified_grid_points)
        
        # Fill matrix with known values, leave unknown as NA
        num_curves = len(grid_points)
        num_points = len(unified_grid_points)
        
        unified_matrix = np.empty((num_curves, num_points, self.dim_codomain))
        unified_matrix.fill(np.nan)
        
        for curve in range(num_curves):
            for point in range(len(grid_points[curve])):
                for dimension in range(self.dim_codomain):
                    point_index = unified_grid_points.index(grid_points[curve][point])
                    unified_matrix[curve, point_index, dimension] = evaluated_points[curve][point][dimension]

        return unified_matrix, unified_grid_points
        
    def to_grid(  # noqa: D102
        self: T,
    ) -> T:
        
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
        dim_domain: Optional[int] = None,
        dim_codomain: Optional[int] = None,
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
            
        if dim_domain is None:
            dim_domain = self.dim_domain
        
        if dim_codomain is None:
            dim_codomain = self.dim_codomain

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

        return FDataIrregular(
            function_indices,
            function_arguments,
            function_values,
            dim_domain=dim_domain,
            dim_codomain=dim_codomain,
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
        pass

    def shift(
        self,
        shifts: Union[ArrayLike, float],
        *,
        restrict_domain: bool = False,
        extrapolation: Optional[ExtrapolationLike] = None,
    ) -> FDataIrregular:
        pass

    def compose(
        self: T,
        fd: T,
        *,
        eval_points: Optional[GridPointsLike] = None,
    ) -> T:
        pass

    def __str__(self) -> str:
        """Return str(self)."""
        return (
            f"function_indices:    {self.function_indices}\n"
            f"function_arguments:    {self.function_arguments}\n"
            f"function_values:    {self.function_values}\n"
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
            next_index = self.function_indices[i + 1] if i + 1 < \
                self.num_functions else -1
            s = slice(self.function_indices[i], next_index)
            required_slices.append(s)
            
        arguments = np.concatenate([self.function_arguments[s] 
                                    for s in required_slices])
        values = np.concatenate([self.function_values[s] 
                                 for s in required_slices])
        
        chunk_sizes = np.array([s.stop-s.start for s in required_slices])
        indices = np.cumsum(chunk_sizes) - chunk_sizes[0]
        
        return self.copy(
            function_indices=indices.astype(int),
            function_arguments=arguments,
            function_values=values,
            sample_names=self.sample_names[key],
            domain_range=self.sample_names[key],
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
                and not np.array_equal(i.grid_points, self.grid_points)
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
    def dtype(self) -> FDataGridDType:
        # TODO Do this natively?
        """The dtype for this extension array, FDataGridDType"""
        return self.to_grid().dtype

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


# TODO FDataIrregularDType?

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

        coordinate_names = np.array(self._fdatairregular.coordinate_names)[s_key]

        return self._fdatairregular.copy(
            function_values=self._fdatairregular.function_values[..., key],
            coordinate_names=tuple(coordinate_names),
        )

    def __len__(self) -> int:
        """Return the number of coordinates."""
        return self._fdatairregular.dim_codomain
