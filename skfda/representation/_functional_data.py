"""
Module for functional data manipulation.

Defines the abstract class that should be implemented by the funtional data
objects of the package and contains some commons methods.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Iterable,
    Iterator,
    NoReturn,
    Sequence,
    TypeVar,
    cast,
    overload,
)

import numpy as np
import pandas
from matplotlib.figure import Figure
from typing_extensions import Self, override

from skfda._utils.ndfunction._region import AxisAlignedBox

from .._utils.ndfunction import NDFunction, concatenate as concatenate
from .._utils.ndfunction._array_api import Array, DType, Shape, numpy_namespace
from .._utils.ndfunction._region import Region
from .._utils.ndfunction.extrapolation import AcceptedExtrapolation
from .._utils.ndfunction.typing import GridPointsLike
from .._utils.ndfunction.utils.validation import check_grid_points
from ..typing._base import DomainRange, LabelTuple, LabelTupleLike
from ..typing._numpy import (
    NDArrayBool,
    NDArrayFloat,
    NDArrayInt,
    NDArrayObject,
)
from .extrapolation import ExtrapolationLike

if TYPE_CHECKING:
    from .basis import Basis, FDataBasis
    from .grid import FDataGrid

A = TypeVar('A', bound=Array[Shape, DType])


class FData(  # noqa: WPS214
    ABC,
    NDFunction[A],
    pandas.api.extensions.ExtensionArray,
):
    """
    Defines the structure of a functional data object.

    Attributes:
        n_samples: Number of samples.
        dim_domain: Dimension of the domain.
        dim_codomain: Dimension of the image.
        extrapolation: Default extrapolation mode.
        dataset_name: Name of the dataset.
        argument_names: Tuple containing the names of the different
            arguments.
        coordinate_names: Tuple containing the names of the different
            coordinate functions.

    """

    dataset_name: str | None

    def __init__(
        self,
        *,
        extrapolation: ExtrapolationLike[A] | None = None,
        dataset_name: str | None = None,
        argument_names: LabelTupleLike | None = None,
        coordinate_names: LabelTupleLike | None = None,
        sample_names: LabelTupleLike | None = None,
    ) -> None:

        self.extrapolation = extrapolation  # type: ignore[assignment]
        self.dataset_name = dataset_name

        self.argument_names = argument_names  # type: ignore[assignment]
        self.coordinate_names = coordinate_names  # type: ignore[assignment]
        self.sample_names = sample_names  # type: ignore[assignment]

    @override
    @property
    def array_backend(self) -> Any:
        return numpy_namespace

    @property
    def argument_names(self) -> LabelTuple:
        return self._argument_names

    @argument_names.setter
    def argument_names(
        self,
        names: LabelTupleLike | None,
    ) -> None:
        if names is None:
            names = (None,) * self.dim_domain
        else:
            names = tuple(names)
            if len(names) != self.dim_domain:
                raise ValueError(
                    "There must be a name for each of the "
                    "dimensions of the domain.",
                )

        self._argument_names = names

    @property
    def coordinate_names(self) -> LabelTuple:
        return self._coordinate_names

    @coordinate_names.setter
    def coordinate_names(
        self,
        names: LabelTupleLike | None,
    ) -> None:
        if names is None:
            names = (None,) * self.dim_codomain
        else:
            names = tuple(names)
            if len(names) != self.dim_codomain:
                raise ValueError(
                    "There must be a name for each of the "
                    "dimensions of the codomain.",
                )

        self._coordinate_names = names

    @property
    def sample_names(self) -> LabelTuple:
        return self._sample_names

    @sample_names.setter
    def sample_names(self, names: LabelTupleLike | None) -> None:
        if names is None:
            names = (None,) * self.n_samples
        else:
            names = tuple(names)
            if len(names) != self.n_samples:
                raise ValueError(
                    "There must be a name for each of the samples.",
                )

        self._sample_names = names

    @property
    @abstractmethod
    def n_samples(self) -> int:
        """Return the number of samples.

        Returns:
            Number of samples of the FData object.

        """
        pass

    @override
    @property
    def shape(self) -> tuple[int, ...]:
        return (self.n_samples,)

    @property
    @abstractmethod
    def dim_domain(self) -> int:
        """Return number of dimensions of the :term:`domain`.

        Returns:
            Number of dimensions of the domain.

        """
        pass

    @override
    @property
    def input_shape(self) -> tuple[int, ...]:
        return (self.dim_domain,)

    @property
    @abstractmethod
    def dim_codomain(self) -> int:
        """Return number of dimensions of the :term:`codomain`.

        Returns:
            Number of dimensions of the codomain.

        """
        pass

    @override
    @property
    def output_shape(self) -> tuple[int, ...]:
        return (self.dim_codomain,)

    @property
    @abstractmethod
    def domain_range(self) -> DomainRange:
        """Return the :term:`domain` range of the object

        Returns:
            List of tuples with the ranges for each domain dimension.
        """
        pass

    @override
    @property
    def domain(self) -> Region[A]:
        lower = self.array_backend.asarray([d[0] for d in self.domain_range])
        upper = self.array_backend.asarray([d[1] for d in self.domain_range])

        return AxisAlignedBox(lower, upper)

    @abstractmethod
    def derivative(self, *, order: int = 1) -> Self:
        """
        Differentiate a FData object.

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
        domain: DomainRange | None = None,
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
        shifts: A | float,
        *,
        restrict_domain: bool = False,
        extrapolation: AcceptedExtrapolation[A] = "default",
        grid_points: GridPointsLike[A] | None = None,
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
        grid_points = check_grid_points(grid_points)

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
            shifted_grid_points_per_sample = grid_points + arr_shifts
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
        from ..exploratory.visualization.representation import GraphPlot

        return GraphPlot(self, *args, **kwargs).plot()

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
        /,
        correction: int = 0,
    ) -> NDArrayFloat:
        pass

    @overload
    def cov(  # noqa: WPS451
        self,
        /,
        correction: int = 0,
    ) -> Callable[[NDArrayFloat, NDArrayFloat], NDArrayFloat]:
        pass

    @abstractmethod
    def cov(  # noqa: WPS320, WPS451
        self,
        s_points: NDArrayFloat | None = None,
        t_points: NDArrayFloat | None = None,
        /,
        correction: int = 0,
    ) -> Callable[[NDArrayFloat, NDArrayFloat], NDArrayFloat] | NDArrayFloat:
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
        grid_points: GridPointsLike[A] | None = None,
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

    def _copy_op(
        self,
        other: Self | NDArrayFloat | NDArrayInt | float,
        **kwargs: Any,
    ) -> Self:

        base_copy = (
            other if isinstance(other, type(self))
            and self.n_samples == 1 and other.n_samples != 1
            else self
        )

        return base_copy.copy(**kwargs)

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
        scalars: Self | Sequence[Self],
        dtype: Any = None,
        copy: bool = False,
    ) -> Self:

        scalars_seq: Sequence[Self] = (
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
        indexer: Sequence[int] | NDArrayInt,
        allow_fill: bool = False,
        fill_value: Self | None = None,
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

        arr_indices = np.atleast_1d(indexer)

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
