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

from .._utils import _check_array_key, _int_to_real, _to_grid_points, constants
from ..typing._base import (
    DomainRange,
    DomainRangeLike,
    GridPoints,
    GridPointsLike,
    LabelTupleLike,
)
from ..typing._numpy import ArrayLike, NDArrayBool, NDArrayFloat, NDArrayInt
from ._functional_data import FData
from .grid import FDataGrid
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
        argument_names: Optional[LabelTupleLike] = None,
        coordinate_names: Optional[LabelTupleLike] = None
        ):
        """Construct a FDataIrregular object."""
            
        # Set dimensions
        # TODO Check dimensions against num of arguments and coordinates?
        self._dim_domain = dim_domain
        self._dim_codomain = dim_codomain
        
        # Set structure to given data
        self.num_functions = function_indices.shape[0]
        
        assert function_arguments.shape[0] == function_values.shape[0]
        self.num_observations = function_arguments.shape[0]
        
        self.set_function_indices(function_indices)
        self.set_function_arguments(function_arguments)
        self.set_function_values(function_values)
        
        #TODO Fix for higher dimensions
        i=0
        self._sample_range = list()
        for f in self.function_indices[1:]:
            self._sample_range.append((self.function_arguments[i][0], 
                                      self.function_arguments[f-1][0]))
            i = f
        self._sample_range.append((self.function_arguments[i][0], 
                                      self.function_arguments[-1][0]))
        
        from ..misc.validation import validate_domain_range
        if domain_range is None:
            domain_range = self.sample_range
            # Default value for domain_range is a list of tuples with
            # the first and last element of each list of the grid_points.

        self._domain_range = validate_domain_range(domain_range)

        super().__init__(
            extrapolation=extrapolation,
            dataset_name=dataset_name,
            argument_names=argument_names,
            coordinate_names=coordinate_names,
            sample_names=sample_names,
        )
   
    @classmethod
    def from_dataframe(
        cls,
        *, 
        dataframe: pandas.DataFrame,
        id_column: str,
        argument_columns: LabelTupleLike,
        coordinate_columns: LabelTupleLike,
        **kwargs
        ) -> FDataIrregular:
        
        # Accept strings but ensure the column names are tuples
        _is_str = isinstance(argument_columns, str)
        argument_columns = [argument_columns] if _is_str else argument_columns
        
        _is_str = isinstance(coordinate_columns, str)
        coordinate_columns = [coordinate_columns]  if _is_str else coordinate_columns
        
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
    
    def set_function_indices(self, function_indices):
        self.function_indices = function_indices.copy()
    
    def set_function_arguments(self, function_arguments):
        self.function_arguments = function_arguments.copy()
        
    def set_function_values(self, function_values):
        self.function_values = function_values.copy()
        
    def round(
        self,
        decimals: int = 0,
        out: Optional[FDataIrregular] = None,
    ) -> FDataIrregular:
        rounded_arguments = self.function_arguments.round(decimals=decimals)
        rounded_values = self.function_values.round(decimals=decimals)
        
        if out is not None and isinstance(out, FDataIrregular):
            out.function_indices = self.function_indices
            out.function_arguments = rounded_arguments
            out.function_values = rounded_values
            
            return out
        
        return self.copy(
            function_arguments=rounded_arguments, 
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
    def coordinates(self: T) -> _CoordinateIterator[T]:
        #TODO Does it even make sense to do this? Maybe it requires an specific _IrregularCoordinateIterator over the structure
        pass

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

        #TODO
        pass

    def derivative(
        self: T,
        *,
        order: int = 1,
        method: Optional[Basis] = None,
    ) -> T:
        #TODO
        pass

    def integrate(
        self: T,
        *,
        domain: Optional[DomainRange] = None,
    ) -> NDArrayFloat:
        #TODO
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
        #TODO Implement when attributes are done
        pass
    
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
        values = [np.matrix.flatten(self.function_values[np.where(self.function_arguments == arg)[0]])
                    for arg in distinct_args]
        
        # Obtain mean of all available values for each argument point
        vars = np.array([np.mean(vals) for vals in values])
        
        # Create a FDataGrid object with only 1 curve, the mean curve
        return FDataGrid(
            grid_points=distinct_args,
            data_matrix=np.array([vars]),
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
        values = [np.matrix.flatten(self.function_values[np.where(self.function_arguments == arg)[0]])
                    for arg in distinct_args]
        
        # Obtain variance of all available values for each argument point
        vars = np.array([np.var(vals) for vals in values])
        
        # Create a FDataGrid object with only 1 curve, the variance curve
        return FDataGrid(
            grid_points=distinct_args,
            data_matrix=np.array([vars]),
            sample_names=("variance",),
        )

    def cov(self: T) -> T:
        #TODO Implementation to be decided
        pass

    def gmean(self: T) -> T:
        #TODO Implement when attributes are done
        pass

    def equals(self, other: object) -> bool:
        """Comparison of FDataSparse objects."""
        #TODO Implement when attributes are done
        pass

    def _eq_elemenwise(self: T, other: T) -> NDArrayBool:
        """Elementwise equality of FDataSparse."""
        #TODO Implement when attributes are done
        pass

    def _get_op_matrix(
        self,
        other: Union[T, NDArrayFloat, NDArrayInt, float],
    ) -> Union[None, float, NDArrayFloat, NDArrayInt]:
        
        #TODO Implement when attributes are done
        pass

    def __add__(
        self: T,
        other: Union[T, NDArrayFloat, NDArrayInt, float],
    ) -> T:

        #TODO Implement when attributes are done
        pass

    def __radd__(
        self: T,
        other: Union[T, NDArrayFloat, NDArrayInt, float],
    ) -> T:

        #TODO Implement when attributes are done
        pass

    def __sub__(
        self: T,
        other: Union[T, NDArrayFloat, NDArrayInt, float],
    ) -> T:

        #TODO Implement when attributes are done
        pass

    def __rsub__(
        self: T,
        other: Union[T, NDArrayFloat, NDArrayInt, float],
    ) -> T:

        #TODO Implement when attributes are done
        pass

    def __mul__(
        self: T,
        other: Union[T, NDArrayFloat, NDArrayInt, float],
    ) -> T:

        #TODO Implement when attributes are done
        pass

    def __rmul__(
        self: T,
        other: Union[T, NDArrayFloat, NDArrayInt, float],
    ) -> T:

        #TODO Implement when attributes are done
        pass

    def __truediv__(
        self: T,
        other: Union[T, NDArrayFloat, NDArrayInt, float],
    ) -> T:

        #TODO Implement when attributes are done
        pass

    def __rtruediv__(
        self: T,
        other: Union[T, NDArrayFloat, NDArrayInt, float],
    ) -> T:

        #TODO Implement when attributes are done
        pass

    def __neg__(self: T) -> T:
        """Negation of FData object."""
        #TODO Should be easy to implement, just negating the values
        pass

    def concatenate(self: T, *others: T, as_coordinates: bool = False) -> T:
        #TODO Implement allocing memory only once
        pass
    
    def plot(self, *args: Any, **kwargs: Any) -> Figure:
        from ..exploratory.visualization.representation import LinearPlotIrregular

        return LinearPlotIrregular(self, *args, **kwargs).plot()

    def scatter(self, *args: Any, **kwargs: Any) -> Figure:
        from ..exploratory.visualization.representation import ScatterPlotIrregular

        return ScatterPlotIrregular(self, *args, **kwargs).plot()
    
    def plot_and_scatter(self, *args: Any, **kwargs: Any) -> Figure:
        fig = self.scatter(*args, **kwargs)
        self.plot(fig=fig, *args, **kwargs)

    def to_basis(self, basis: Basis, **kwargs: Any) -> FDataBasis:
        #TODO Use BasisSmoother to return basis?
        pass

    def to_grid(  # noqa: D102
        self: T,
        grid_points: Optional[GridPointsLike] = None,
        *,
        sample_points: Optional[GridPointsLike] = None,
    ) -> T:

        #TODO Return list of data grids? Data grid with holes?
        pass

    def copy(  # noqa: WPS211
        self: T,
        deep: bool = False,  # For Pandas compatibility
        argument_names: Optional[LabelTupleLike] = None,
        coordinate_names: Optional[LabelTupleLike] = None,
        dim_domain: Optional[int] = 1,
        dim_codomain: Optional[int] = 1,
        domain_range: Optional[DomainRangeLike] = None,
        dataset_name: Optional[str] = None,
        sample_names: Optional[LabelTupleLike] = None,
        extrapolation: Optional[ExtrapolationLike] = None,
    ) -> T:
        
        #TODO Should allow to copy directly from FDataIrregular, not from dataframe
        pass

    def restrict(
        self: T,
        domain_range: DomainRangeLike,
    ) -> T:
        
        #TODO Is this possible with this structure
        pass

    def shift(
        self,
        shifts: Union[ArrayLike, float],
        *,
        restrict_domain: bool = False,
        extrapolation: Optional[ExtrapolationLike] = None,
        grid_points: Optional[GridPointsLike] = None,
    ) -> FDataGrid:
        #TODO Is this possible with this structure?
        pass

    def compose(
        self: T,
        fd: T,
        *,
        eval_points: Optional[GridPointsLike] = None,
    ) -> T:
        
        #TODO Is this possible with this structure?
        pass

    def __str__(self) -> str:
        """Return str(self)."""
        #TODO Define str method after all attributes are locked
        pass

    def __repr__(self) -> str:
        """Return repr(self)."""
        return (
            f"FDataIrregular("  # noqa: WPS221
            f"\nfunction_indices={self.function_indices!r},"
            f"\nfunction_arguments={self.function_arguments!r},"
            f"\nfunction_values={self.function_values!r},"
            #f"\ndomain_range={self.domain_range!r},"
            f"\ndataset_name={self.dataset_name!r},"
            f"\nargument_names={self.argument_names!r},"
            f"\ncoordinate_names={self.coordinate_names!r},"
            f"\nextrapolation={self.extrapolation!r},"
            #f"\ninterpolation={self.interpolation!r})"
        ).replace(
            '\n',
            '\n    ',
        )

    def __getitem__(
        self: T,
        key: Union[int, slice, NDArrayInt, NDArrayBool],
    ) -> T:
        """Return self[key]."""
        #TODO Maybe return from the view? Or transform using view functions directly from data structure?

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
        """The dtype for this extension array, FDataGridDType"""
        return FDataGridDType(
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


#TODO Do i need a FDataIrregularDType?

#TODO Do I need a _CoordinateIterator?
