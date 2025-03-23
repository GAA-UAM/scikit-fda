"""Module for functional data manipulation.

Defines the abstract class that should be implemented by the funtional data
objects of the package and contains some commons methods.
"""

from __future__ import annotations

import warnings
from abc import abstractmethod
from math import prod
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Iterable,
    Iterator,
    Literal,
    NoReturn,
    Optional,
    Protocol,
    Sequence,
    TypeVar,
    Union,
    cast,
    overload,
)

import numpy as np
import pandas.api.extensions
from array_api_compat import array_namespace
from matplotlib.figure import Figure
from typing_extensions import Self

from ...typing._base import DomainRange, GridPointsLike, LabelTupleLike
from ...typing._numpy import (
    ArrayLike,
    NDArrayBool,
    NDArrayFloat,
    NDArrayInt,
    NDArrayObject,
)
from .._utils import _evaluate_grid, _to_grid_points
from ._array_api import Array, BoolDType, DType
from ._region import Region
from .evaluator import Evaluator
from .extrapolation import ExtrapolationLike, _parse_extrapolation

if TYPE_CHECKING:
    from ...representation.basis import Basis, FDataBasis
    from ...representation.grid import FDataGrid

T = TypeVar('T', bound='NDFunction')
InputDType = TypeVar('InputDType', bound=DType)
OutputDType = TypeVar('OutputDType', bound=DType)

EvalPointsType = Union[
    ArrayLike,
    GridPointsLike,
    Iterable[GridPointsLike],
]

AcceptedExtrapolation = Union[ExtrapolationLike, None, Literal["default"]]


def _join_evaluation(
    index_matrix: NDArrayBool,
    index_ext: NDArrayBool,
    index_ev: NDArrayBool,
    res_extrapolation: NDArrayFloat,
    res_evaluation: NDArrayFloat,
    n_samples: int,
    dim_codomain: int,
) -> NDArrayFloat:
    """
    Join the points evaluated.

    This method is used internally by :func:`evaluate` to join the result
    of the evaluation and the result of the extrapolation.

    Args:
        index_matrix: Boolean index with the points extrapolated.
        index_ext: Boolean index with the columns that contains
            points extrapolated.
        index_ev: Boolean index with the columns that contains
            points evaluated.
        res_extrapolation: Result of the extrapolation.
        res_evaluation: Result of the evaluation.

    Returns:
        Matrix with the points evaluated with shape
        `n_samples` x `number of points evaluated` x `dim_codomain`.

    """
    res = np.empty((
        n_samples,
        index_matrix.shape[-1],
        dim_codomain,
    ))

    # Case aligned evaluation
    if index_matrix.ndim == 1:
        res[:, index_ev, :] = res_evaluation
        res[:, index_ext, :] = res_extrapolation

    else:

        res[:, index_ev] = res_evaluation
        res[index_matrix] = res_extrapolation[index_matrix[:, index_ext]]

    return res


class NDFunction(Protocol):
    """
    Protocol for an arbitrary-dimensional array of functions.

    Objects of this class define vectors, matrices or tensors containing
    functions of a (multidimensional) array returning a (multidimensional)
    array.
    """

    def __init__(
        self,
        *,
        extrapolation: Optional[ExtrapolationLike] = None,
    ) -> None:

        self.extrapolation = extrapolation  # type: ignore[assignment]

    @property
    def ndim(self) -> int:
        """
        Shape of the array of functions.

        It is a tuple representing the shape of the array containing the
        functions.

        """
        return len(self.shape)

    @property
    @abstractmethod
    def shape(self) -> tuple[int, ...]:
        """
        Shape of the array of functions.

        It is a tuple representing the shape of the array containing the
        functions.

        """
        pass

    @property
    def size(self) -> int:
        """Number of functions in the array."""
        return prod(self.shape)

    @property
    @abstractmethod
    def input_shape(self) -> tuple[int, ...]:
        """
        Shape of the n-dimensional input.

        It is a tuple representing the shape of the arrays that the
        functions will accept.

        """
        pass

    @property
    @abstractmethod
    def output_shape(self) -> tuple[int, ...]:
        """
        Shape of the n-dimensional output.

        It is a tuple representing the shape of the arrays that the
        functions will output.

        """
        pass

    @property
    @abstractmethod
    def coordinates(self) -> Self:
        """
        View of the coordinate functions as an array.

        The coordinate functions are the functions that output each of the
        scalar values conforming the output array.

        """
        pass

    @property
    def extrapolation(self) -> Evaluator | None:
        """Extrapolation used for evaluating values outside the bounds."""
        return (  # type: ignore [no-any-return]
            self._extrapolation  # type: ignore [attr-defined]
        )

    @extrapolation.setter
    def extrapolation(self, value: ExtrapolationLike | None) -> None:
        self._extrapolation = (  # type: ignore [misc, attr-defined]
            _parse_extrapolation(value)
        )

    @property
    @abstractmethod
    def domain(self) -> Region[InputDType]:
        """Domain of the function."""
        pass

    @abstractmethod
    def _evaluate(
        self,
        eval_points: NDArrayFloat,
        *,
        aligned: bool = True,
    ) -> NDArrayFloat:
        """Define the evaluation of the FData.

        Evaluates the samples of an FData object at several points.

        Subclasses must override this method to implement evaluation.

        Args:
            eval_points: List of points where the functions are
                evaluated. If `aligned` is `True`, then a list of
                lists of points must be passed, with one list per sample.
            aligned: Whether the input points are
                the same for each sample, or an array of points per sample is
                passed.

        Returns:
            Numpy 3d array with shape `(n_samples,
            len(eval_points), dim_codomain)` with the result of the
            evaluation. The entry (i,j,k) will contain the value k-th image
            dimension of the i-th sample, at the j-th evaluation point.

        """
        pass

    @overload
    def evaluate(
        self,
        eval_points: ArrayLike,
        *,
        derivative: int = 0,
        extrapolation: AcceptedExtrapolation = "default",
        grid: Literal[False] = False,
        aligned: Literal[True] = True,
    ) -> NDArrayFloat:
        pass

    @overload
    def evaluate(
        self,
        eval_points: Iterable[ArrayLike],
        *,
        derivative: int = 0,
        extrapolation: AcceptedExtrapolation = "default",
        grid: Literal[False] = False,
        aligned: Literal[False],
    ) -> NDArrayFloat:
        pass

    @overload
    def evaluate(
        self,
        eval_points: GridPointsLike,
        *,
        derivative: int = 0,
        extrapolation: AcceptedExtrapolation = "default",
        grid: Literal[True],
        aligned: Literal[True] = True,
    ) -> NDArrayFloat:
        pass

    @overload
    def evaluate(
        self,
        eval_points: Iterable[GridPointsLike],
        *,
        derivative: int = 0,
        extrapolation: AcceptedExtrapolation = "default",
        grid: Literal[True],
        aligned: Literal[False],
    ) -> NDArrayFloat:
        pass

    def evaluate(
        self,
        eval_points: EvalPointsType,
        *,
        derivative: int = 0,
        extrapolation: AcceptedExtrapolation = "default",
        grid: bool = False,
        aligned: bool = True,
    ) -> NDArrayFloat:
        """
        Evaluate the object at a list of values or a grid.

        ..  deprecated:: 0.8
            Use normal calling notation instead.

        Args:
            eval_points: List of points where the functions are
                evaluated. If ``grid`` is ``True``, a list of axes, one per
                :term:`domain` dimension, must be passed instead. If
                ``aligned`` is ``True``, then a list of lists (of points or
                axes, as explained) must be passed, with one list per sample.
            derivative: Deprecated. Order of the derivative to evaluate.
            extrapolation: Controls the
                extrapolation mode for elements outside the domain range. By
                default it is used the mode defined during the instance of the
                object.
            grid: Whether to evaluate the results on a grid
                spanned by the input arrays, or at points specified by the
                input arrays. If true the eval_points should be a list of size
                dim_domain with the corresponding times for each axis. The
                return matrix has shape n_samples x len(t1) x len(t2) x ... x
                len(t_dim_domain) x dim_codomain. If the domain dimension is 1
                the parameter has no efect. Defaults to False.
            aligned: Whether the input points are the same for each sample,
                or an array of points per sample is passed.

        Returns:
            Matrix whose rows are the values of the each
            function at the values specified in eval_points.

        """
        warnings.warn(
            "The method 'evaluate' is deprecated. "
            "Please use the normal calling notation on the functional data "
            "object instead.",
            DeprecationWarning,
            stacklevel=2,
        )

        return self(
            eval_points=eval_points,
            derivative=derivative,
            extrapolation=extrapolation,
            grid=grid,
            aligned=aligned,
        )

    @overload
    def __call__(
        self,
        eval_points: ArrayLike,
        *,
        derivative: int = 0,
        extrapolation: AcceptedExtrapolation = "default",
        grid: Literal[False] = False,
        aligned: bool = True,
    ) -> NDArrayFloat:
        pass

    @overload
    def __call__(
        self,
        eval_points: GridPointsLike,
        *,
        derivative: int = 0,
        extrapolation: AcceptedExtrapolation = "default",
        grid: Literal[True],
        aligned: Literal[True] = True,
    ) -> NDArrayFloat:
        pass

    @overload
    def __call__(
        self,
        eval_points: Iterable[GridPointsLike],
        *,
        derivative: int = 0,
        extrapolation: AcceptedExtrapolation = "default",
        grid: Literal[True],
        aligned: Literal[False],
    ) -> NDArrayFloat:
        pass

    @overload
    def __call__(
        self,
        eval_points: EvalPointsType,
        *,
        derivative: int = 0,
        extrapolation: AcceptedExtrapolation = "default",
        grid: bool = False,
        aligned: bool = True,
    ) -> NDArrayFloat:
        pass

    def __call__(
        self,
        eval_points: EvalPointsType,
        *,
        derivative: int = 0,
        extrapolation: AcceptedExtrapolation = "default",
        grid: bool = False,
        aligned: bool = True,
    ) -> NDArrayFloat:
        """
        Evaluate the :term:`functional object`.

        Evaluate the object or its derivatives at a list of values or a
        grid. This method is a wrapper of :meth:`evaluate`.

        Args:
            eval_points: List of points where the functions are
                evaluated. If a matrix of shape nsample x eval_points is given
                each sample is evaluated at the values in the corresponding row
                in eval_points.
            derivative: Order of the derivative. Defaults to 0.
            extrapolation: Controls the
                extrapolation mode for elements outside the domain range. By
                default it is used the mode defined during the instance of the
                object.
            grid: Whether to evaluate the results on a grid
                spanned by the input arrays, or at points specified by the
                input arrays. If true the eval_points should be a list of size
                dim_domain with the corresponding times for each axis. The
                return matrix has shape n_samples x len(t1) x len(t2) x ... x
                len(t_dim_domain) x dim_codomain. If the domain dimension is 1
                the parameter has no efect. Defaults to False.
            aligned: Whether the input points are the same for each sample,
                or an array of points per sample is passed.

        Returns:
            Matrix whose rows are the values of the each
            function at the values specified in eval_points.

        """
        from ...misc.validation import validate_evaluation_points

        if derivative != 0:
            warnings.warn(
                "Parameter derivative is deprecated. Use the "
                "derivative function instead.",
                DeprecationWarning,
            )
            return self.derivative(order=derivative)(
                eval_points,
                extrapolation=extrapolation,
                grid=grid,
                aligned=aligned,
            )

        if grid:  # Evaluation of a grid performed in auxiliar function

            return _evaluate_grid(
                eval_points,
                evaluate_method=self,
                n_samples=self.n_samples,
                dim_domain=self.dim_domain,
                dim_codomain=self.dim_codomain,
                extrapolation=extrapolation,
                aligned=aligned,
            )

        if extrapolation == "default":
            extrapolation = self.extrapolation
        else:
            # Gets the function to perform extrapolation or None
            extrapolation = _parse_extrapolation(extrapolation)

        eval_points = cast(
            ArrayLike,
            eval_points,
        )

        # Convert to array and check dimensions of eval points
        eval_points = validate_evaluation_points(
            eval_points,
            aligned=aligned,
            n_samples=self.n_samples,
            dim_domain=self.dim_domain,
        )

        if extrapolation is not None:

            index_matrix = ~self.domain.contains(eval_points)

            if index_matrix.any():

                # Partition of eval points
                if aligned:

                    index_ext = index_matrix
                    index_ev = ~index_matrix

                    eval_points_extrapolation = eval_points[index_ext]
                    eval_points_evaluation = eval_points[index_ev]

                else:
                    index_ext = np.logical_or.reduce(index_matrix, axis=0)
                    eval_points_extrapolation = eval_points[:, index_ext]

                    index_ev = np.logical_or.reduce(~index_matrix, axis=0)
                    eval_points_evaluation = eval_points[:, index_ev]

                # Direct evaluation
                res_evaluation = self._evaluate(
                    eval_points_evaluation,
                    aligned=aligned,
                )

                res_extrapolation = extrapolation(
                    self,
                    eval_points_extrapolation,
                    aligned=aligned,
                )

                return _join_evaluation(
                    index_matrix,
                    index_ext,
                    index_ev,
                    res_extrapolation,
                    res_evaluation,
                    n_samples=self.n_samples,
                    dim_codomain=self.dim_codomain,
                )

        return self._evaluate(
            eval_points,
            aligned=aligned,
        )

    @abstractmethod
    def derivative(self, *, order: int = 1) -> Self:
        """Differentiate a FData object.

        Args:
            order: Order of the derivative. Defaults to one.

        Returns:
            Functional object containg the derivative.

        """
        pass

    @abstractmethod
    def integrate(
        self,
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

        """
        pass

    @abstractmethod
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

        """
        assert grid_points is not None
        grid_points = _to_grid_points(grid_points)

        arr_shifts = np.array([shifts] if np.isscalar(shifts) else shifts)

        # Accept unidimensional array when the domain dimension is one or when
        # the shift is the same for each sample
        if arr_shifts.ndim == 1:
            arr_shifts = (
                arr_shifts[np.newaxis, :]  # Same shift for each sample
                if len(arr_shifts) == self.dim_domain
                else arr_shifts[:, np.newaxis]
            )

        if len(arr_shifts) not in {1, self.n_samples}:
            raise ValueError(
                f"The length of the shift vector ({len(arr_shifts)}) must "
                f"have length equal to 1 or to the number of samples "
                f"({self.n_samples})",
            )

        if restrict_domain:
            domain = np.asarray(self.domain_range)

            a = domain[:, 0] - np.min(np.min(arr_shifts, axis=0), 0)
            b = domain[:, 1] - np.max(np.max(arr_shifts, axis=1), 0)

            domain = np.hstack((a, b))
            domain_range = tuple(domain)

        if len(arr_shifts) == 1:
            shifted_grid_points = tuple(
                g + s for g, s in zip(grid_points, arr_shifts[0])
            )
            data_matrix = self(
                shifted_grid_points,
                extrapolation=extrapolation,
                aligned=True,
                grid=True,
            )
        else:
            shifted_grid_points_per_sample = (
                tuple(
                    g + s for g, s in zip(grid_points, shift)
                ) for shift in arr_shifts
            )
            data_matrix = self(
                shifted_grid_points_per_sample,
                extrapolation=extrapolation,
                aligned=False,
                grid=True,
            )

        shifted = self.to_grid().copy(
            data_matrix=data_matrix,
            grid_points=grid_points,
        )

        if restrict_domain:
            shifted = shifted.restrict(domain_range)

        return shifted

    def plot(self, *args: Any, **kwargs: Any) -> Figure:
        """Plot the FDatGrid object.

        Args:
            args: Positional arguments to be passed to the class
                :class:`~skfda.exploratory.visualization.representation.GraphPlot`.
            kwargs: Keyword arguments to be passed to the class
                :class:`~skfda.exploratory.visualization.representation.GraphPlot`.

        Returns:
            Figure object in which the graphs are plotted.

        """
        from ...exploratory.visualization.representation import GraphPlot

        return GraphPlot(self, *args, **kwargs).plot()

    @abstractmethod
    def copy(
        self,
        *,
        deep: bool = False,  # For Pandas compatibility
        dataset_name: Optional[str] = None,
        argument_names: Optional[LabelTupleLike] = None,
        coordinate_names: Optional[LabelTupleLike] = None,
        sample_names: Optional[LabelTupleLike] = None,
        extrapolation: Optional[ExtrapolationLike] = None,
    ) -> Self:
        """Make a copy of the object."""
        pass

    @abstractmethod  # noqa: WPS125
    def sum(  # noqa: WPS125
        self,
        *,
        axis: int | None = None,
        out: None = None,
        keepdims: bool = False,
        skipna: bool = False,
        min_count: int = 0,
    ) -> Self:
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
            A FData object with just one sample representing
            the sum of all the samples in the original object.

        """
        if (
            (axis is not None and axis != 0)
            or out is not None
            or keepdims is not False
        ):
            raise NotImplementedError(
                "Not implemented for that parameter combination",
            )

        return self

    @overload
    def cov(  # noqa: WPS451
        self,
        s_points: NDArrayFloat,
        t_points: NDArrayFloat,
        / ,
        correction: int = 0,
    ) -> NDArrayFloat:
        pass

    @overload
    def cov(  # noqa: WPS451
        self,
        / ,
        correction: int = 0,
    ) -> Callable[[NDArrayFloat, NDArrayFloat], NDArrayFloat]:
        pass

    @abstractmethod
    def cov(  # noqa: WPS320, WPS451
        self,
        s_points: Optional[NDArrayFloat] = None,
        t_points: Optional[NDArrayFloat] = None,
        / ,
        correction: int = 0,
    ) -> Union[
        Callable[[NDArrayFloat, NDArrayFloat], NDArrayFloat],
        NDArrayFloat,
    ]:
        """Compute the covariance of the functional data object.

        Calculates the unbiased sample covariance function of the data.
        This is expected to be only defined for univariate functions.
        The resulting covariance function is defined in the cartesian
        product of the domain of the functions.
        If s_points or t_points are not provided, this method returns
        a callable object representing the covariance function.
        If s_points and t_points are provided, this method returns the
        evaluation of the covariance function at the grid formed by the
        cartesian product of the points in s_points and t_points.

        Args:
            s_points: Points where the covariance function is evaluated.
            t_points: Points where the covariance function is evaluated.
            correction: degrees of freedom adjustment. The divisor used in the
                calculation is `N - correction`, where `N` represents the
                number of elements. Default: `0`.

        Returns:
            Covariance function.

        """
        pass

    def mean(
        self,
        *,
        axis: int | None = None,
        dtype: None = None,
        out: None = None,
        keepdims: bool = False,
        skipna: bool = False,
    ) -> Self:
        """Compute the mean of all the samples.

        Args:
            axis: Used for compatibility with numpy. Must be None or 0.
            dtype: Used for compatibility with numpy. Must be None.
            out: Used for compatibility with numpy. Must be None.
            keepdims: Used for compatibility with numpy. Must be False.
            skipna: Wether the NaNs are ignored or not.

        Returns:
            A FData object with just one sample representing
            the mean of all the samples in the original object.

        """
        if dtype is not None:
            raise NotImplementedError(
                "Not implemented for that parameter combination",
            )

        return (
            self.sum(axis=axis, out=out, keepdims=keepdims, skipna=skipna)
            / self.n_samples
        )

    @abstractmethod
    def to_grid(
        self,
        grid_points: Optional[GridPointsLike] = None,
    ) -> FDataGrid:
        """Return the discrete representation of the object.

        Args:
            grid_points: Points per axis
                where the function is going to be evaluated.

        Returns:
            Discrete representation of the functional data
            object.

        """
        pass

    @abstractmethod
    def to_basis(
        self,
        basis: Basis,
        **kwargs: Any,
    ) -> FDataBasis:
        """Return the basis representation of the object.

        Args:
            basis: basis object in which the functional data are
                going to be represented.
            kwargs: keyword arguments to be passed to
                FDataBasis.from_data().

        Returns:
            Basis representation of the funtional data
            object.

        """
        pass

    @abstractmethod
    def concatenate(self, *others: Self, as_coordinates: bool = False) -> Self:
        """Join samples from a similar FData object.

        Joins samples from another FData object if it has the same
        dimensions and has compatible representations.

        Args:
            others: other FData objects.
            as_coordinates:  If False concatenates as
                new samples, else, concatenates the other functions as
                new components of the image. Defaults to False.

        Returns:
            :class:`FData`: FData object with the samples from the two
            original objects.

        """
        pass

    @abstractmethod
    def compose(
        self,
        fd: Self,
        *,
        eval_points: Optional[NDArrayFloat] = None,
    ) -> Self:
        """Composition of functions.

        Performs the composition of functions.

        Args:
            fd: FData object to make the composition. Should
                have the same number of samples and image dimension equal to
                the domain dimension of the object composed.
            eval_points: Points to perform the evaluation.

        """
        pass

    @abstractmethod
    def __getitem__(
        self,
        key: Union[int, slice, NDArrayInt],
    ) -> Self:
        """Return self[key]."""
        pass

    def equals(self, other: object) -> bool:
        """Whole object equality."""
        return (
            isinstance(other, type(self))  # noqa: WPS222
            and self.extrapolation == other.extrapolation
            and self.dataset_name == other.dataset_name
            and self.argument_names == other.argument_names
            and self.coordinate_names == other.coordinate_names
        )

    @abstractmethod
    def _eq_elemenwise(self, other: Self) -> NDArrayBool:
        """Elementwise equality."""
        pass

    def __eq__(self, other: object) -> NDArrayBool:  # type: ignore[override]
        """Elementwise equality, as with arrays."""
        if not isinstance(other, type(self)) or self.dtype != other.dtype:
            if other is pandas.NA:
                return self.isna()
            if pandas.api.types.is_list_like(other) and not isinstance(
                other, (pandas.Series, pandas.Index, pandas.DataFrame),
            ):
                other = cast(Iterable[object], other)
                return np.concatenate([x == y for x, y in zip(self, other)])

            return NotImplemented

        if len(self) != len(other) and len(self) != 1 and len(other) != 1:
            raise ValueError(
                f"Different lengths: "
                f"len(self)={len(self)} and "
                f"len(other)={len(other)}",
            )

        return self._eq_elemenwise(other)

    def __ne__(self, other: object) -> NDArrayBool:  # type: ignore[override]
        """Return for `self != other` (element-wise in-equality)."""
        result = self.__eq__(other)
        if result is NotImplemented:
            return NotImplemented

        return ~result

    def _copy_op(
        self,
        other: Union[Self, NDArrayFloat, NDArrayInt, float],
        **kwargs: Any,
    ) -> Self:

        base_copy = (
            other if isinstance(other, type(self))
            and self.n_samples == 1 and other.n_samples != 1
            else self
        )

        return base_copy.copy(**kwargs)

    @abstractmethod
    def __add__(self, other: Self) -> Self:
        """Addition for FData object."""
        pass

    @abstractmethod
    def __radd__(self, other: Self) -> Self:
        """Addition for FData object."""
        pass

    @abstractmethod
    def __sub__(self, other: Self) -> Self:
        """Subtraction for FData object."""
        pass

    @abstractmethod
    def __rsub__(self, other: Self) -> Self:
        """Right subtraction for FData object."""
        pass

    @abstractmethod
    def __mul__(
        self,
        other: Union[NDArrayFloat, NDArrayInt, float],
    ) -> Self:
        """Multiplication for FData object."""
        pass

    @abstractmethod
    def __rmul__(
        self,
        other: Union[NDArrayFloat, NDArrayInt, float],
    ) -> Self:
        """Multiplication for FData object."""
        pass

    @abstractmethod
    def __truediv__(
        self,
        other: Union[NDArrayFloat, NDArrayInt, float],
    ) -> Self:
        """Division for FData object."""
        pass

    @abstractmethod
    def __rtruediv__(
        self,
        other: Union[NDArrayFloat, NDArrayInt, float],
    ) -> Self:
        """Right division for FData object."""
        pass

    @abstractmethod
    def __neg__(self) -> Self:
        """Negation of FData object."""
        pass

    def __iter__(self) -> Iterator[Self]:
        """Iterate over the samples."""
        yield from (self[i] for i in range(self.n_samples))

    def __len__(self) -> int:
        """Return the number of samples of the FData object."""
        return self.n_samples

    #####################################################################
    # Numpy methods
    #####################################################################

    def __array__(self, *args: Any, **kwargs: Any) -> NDArrayObject:
        """Return a numpy array with the objects."""
        # This is to prevent numpy to access inner dimensions
        array = np.empty(shape=len(self), dtype=np.object_)

        for i, f in enumerate(self):
            array[i] = f

        return array

    def __array_ufunc__(
        self,
        ufunc: Any,
        method: str,
        *inputs: Any,
        **kwargs: Any,
    ) -> Any:
        """Prevent NumPy from converting to array just to do operations."""
        # Make normal multiplication by scalar use the __mul__ method
        if ufunc == np.multiply and method == "__call__" and len(inputs) == 2:
            if isinstance(inputs[0], np.ndarray):
                inputs = inputs[::-1]

            return inputs[0] * inputs[1]

        return NotImplemented

    #####################################################################
    # Pandas ExtensionArray methods
    #####################################################################
    @property
    def ndim(self) -> int:
        """
        Return number of dimensions of the functional data.

        It is always 1, as each observation is considered a "scalar" object.

        Returns:
            Number of dimensions of the functional data.

        """
        return 1

    @classmethod
    def _from_sequence(
        cls,
        scalars: Union['FData', Sequence['FData']],
        dtype: Any = None,
        copy: bool = False,
    ) -> 'FData':

        scalars_seq: Sequence['FData'] = (
            [scalars] if isinstance(scalars, cls) else scalars
        )

        if copy:
            scalars_seq = [f.copy() for f in scalars_seq]

        if dtype is None:
            first_element = next(s for s in scalars_seq if s is not pandas.NA)
            dtype = first_element.dtype

        scalars_seq = [
            s if s is not pandas.NA else dtype._na_repr()  # noqa: WPS437
            for s in scalars_seq
        ]

        if len(scalars_seq) == 0:
            scalars_seq = [dtype._na_repr()[:0]]  # noqa: WPS437

        return cls._concat_same_type(scalars_seq)

    @classmethod
    def _from_factorized(cls, values: Any, original: Any) -> NoReturn:
        raise NotImplementedError(
            "Factorization does not make sense for functional data",
        )

    @abstractmethod
    def _take_allow_fill(
        self,
        indices: NDArrayInt,
        fill_value: Self,
    ) -> Self:
        pass

    @abstractmethod
    def isna(self) -> NDArrayBool:  # noqa: D102
        pass

    def take(  # noqa: WPS238
        self,
        indices: Union[int, Sequence[int], NDArrayInt],
        allow_fill: bool = False,
        fill_value: Optional[Self] = None,
        axis: int = 0,
    ) -> Self:
        """
        Take elements from an array.

        Parameters:
            indices: Indices to be taken.
            allow_fill: How to handle negative values in `indices`.

                * False: negative values in `indices` indicate positional
                  indices from the right (the default). This is similar to
                  :func:`numpy.take`.
                * True: negative values in `indices` indicate
                  missing values. These values are set to `fill_value`. Any
                  other negative values raise a ``ValueError``.

            fill_value: Fill value to use for NA-indices
                when `allow_fill` is True.
                This may be ``None``, in which case the default NA value for
                the type, ``self.dtype.na_value``, is used.
                For many ExtensionArrays, there will be two representations of
                `fill_value`: a user-facing "boxed" scalar, and a low-level
                physical NA value. `fill_value` should be the user-facing
                version, and the implementation should handle translating that
                to the physical version for processing the take if necessary.
            axis: Parameter for compatibility with numpy. Must be always 0.

        Returns:
            FData

        Raises:
            IndexError: When the indices are out of bounds for the array.
            ValueError: When `indices` contains negative values other than
                ``-1`` and `allow_fill` is True.

        Notes:
            ExtensionArray.take is called by ``Series.__getitem__``, ``.loc``,
            ``iloc``, when `indices` is a sequence of values. Additionally,
            it's called by :meth:`Series.reindex`, or any other method
            that causes realignment, with a `fill_value`.

        See Also:
            numpy.take
            pandas.api.extensions.take
        """
        # The axis parameter must exist, because sklearn tries to use take
        # instead of __getitem__
        if axis != 0:
            raise ValueError(f"Axis must be 0, not {axis}")

        arr_indices = np.atleast_1d(indices)

        if fill_value is None:
            fill_value = self.dtype.na_value

        non_empty_take_msg = "cannot do a non-empty take from an empty axes"

        if allow_fill:
            if (arr_indices < -1).any():
                raise ValueError("Invalid indexes")

            positive_mask = arr_indices >= 0
            if len(self) == 0 and positive_mask.any():
                raise IndexError(non_empty_take_msg)

            sample_names = np.zeros(len(arr_indices), dtype=object)
            result = self._take_allow_fill(arr_indices, fill_value)

            sample_names[positive_mask] = np.array(self.sample_names)[
                arr_indices[positive_mask]
            ]

            if fill_value is not self.dtype.na_value:
                sample_names[~positive_mask] = fill_value.sample_names[0]

            result.sample_names = tuple(sample_names)
        else:
            if len(self) == 0 and len(arr_indices) != 0:
                raise IndexError(non_empty_take_msg)

            result = self[arr_indices]

        return result

    @classmethod
    def _concat_same_type(
        cls,
        to_concat: Sequence[T],
    ) -> T:
        """
        Concatenate multiple array.

        Parameters:
            to_concat: Sequence of FData objects to concat.

        Returns:
            Concatenation of the objects.

        """
        if isinstance(to_concat, cls):
            return to_concat

        return concatenate(to_concat)

    def astype(self, dtype: Any, copy: bool = True) -> Any:
        """Cast to a new dtype."""
        if isinstance(dtype, type(self.dtype)):
            new_obj = self
            if copy:
                new_obj = self.copy()
            return new_obj

        return super().astype(dtype)

    def _reduce(self, name: str, skipna: bool = True, **kwargs: Any) -> Any:
        meth = getattr(self, name, None)
        if meth:
            return meth(skipna=skipna, **kwargs)

        raise TypeError(
            f"'{type(self).__name__}' does not implement "
            f"reduction '{name}'",
        )


def concatenate(functions: Iterable[T], as_coordinates: bool = False) -> T:
    """
    Join samples from an iterable of similar FData objects.

    Joins samples of FData objects if they have the same
    dimensions and sampling points.
    Args:
        objects: Objects to be concatenated.
        as_coordinates:  If False concatenates as
                new samples, else, concatenates the other functions as
                new components of the image. Defaults to False.
    Returns:
        FData object with the samples from the
        original objects.
    Raises:
        ValueError: In case the provided list of FData objects is
        empty.
    Todo:
        By the moment, only unidimensional objects are supported in basis
        representation.
    """
    functions = iter(functions)
    first = next(functions, None)

    if first is None:
        raise ValueError(
            "At least one FData object must be provided to concatenate.",
        )

    return first.concatenate(*functions, as_coordinates=as_coordinates)


F = TypeVar("F", covariant=True)


class _CoordinateSequence(Protocol[F]):
    """
    Sequence of coordinates.

    Note that this represents a sequence of coordinates, not a sequence of
    FData objects.
    """

    def __getitem__(
        self,
        key: Union[int, slice],
    ) -> F:
        pass

    def __len__(self) -> int:
        pass
