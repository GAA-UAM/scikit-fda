"""
Definition of generic array of functions over multidimensional arrays.

Defines the protocol class that should be followed by every array
of functions.
"""

from __future__ import annotations

from abc import abstractmethod
from math import prod
from typing import (
    Any,
    Iterable,
    Literal,
    Optional,
    Protocol,
    Sequence,
    TypeAlias,
    TypeVar,
    Union,
    cast,
    overload,
)

from typing_extensions import Self

from ...typing._base import LabelTupleLike
from ...typing._numpy import NDArrayBool, NDArrayFloat, NDArrayInt
from ._array_api import Array, DType, Shape, array_namespace
from ._region import Region
from .evaluator import Evaluator
from .extrapolation import ExtrapolationLike, _parse_extrapolation
from .typing import GridPointsLike
from .utils.validation import check_array_namespace, check_evaluation_points

T = TypeVar('T', bound='NDFunction')
A = TypeVar('A', bound=Array[Shape, DType])

EvalPointsType: TypeAlias = A | GridPointsLike[A] | Sequence[GridPointsLike[A]]

AcceptedExtrapolation: TypeAlias = (
    ExtrapolationLike[A] | None | Literal["default"]
)


# When higher-kinded types are supported in Python, this should be generic on:
# - Array backend (e.g.: NumPy or CuPy arrays, or PyTorch tensors)
# - Function shape (e.g.: for vector of N functions, matrix of N_1 x N_2
#   functions)
# - Input shape (e.g.: for functions of vectors of P elements (functions of
#   several variables) or functions of matrices P_1 x P_2
# - Input dtype (e.g.: float input, complex input)
# - Output shape (e.g.: Q-sized vector-valued functions, matrix-valued
#   functions with size Q_1 x Q_2
# - Output dtype (e.g.: float output, complex output)
#
# Right now, we have to choose, so we choose to preserve the backend
# information, which is (type-wise) probably the most important.
#
# In the future we hope this can be retrofitted when higher-kinded
# types and generic defaults are added.
class NDFunction(Protocol[A]):
    """
    Protocol for an arbitrary-dimensional array of functions.

    Objects of this class define vectors, matrices or tensors containing
    functions of a (multidimensional) array returning a (multidimensional)
    array.
    """

    def __init__(
        self,
        *,
        extrapolation: Optional[ExtrapolationLike[A]] = None,
    ) -> None:

        self.extrapolation = extrapolation  # type: ignore[assignment]

    @property
    @abstractmethod
    def array_backend(self) -> Any:
        """
        Return the array backend used.

        This property returns the array namespace of the arrays
        that the functions use as both input and output.

        """
        pass

    @property
    def ndim(self) -> int:
        """Number of dimensions of the array of functions."""
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
    def input_ndim(self) -> int:
        """Number of dimensions of the n-dimensional input."""
        return len(self.input_shape)

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
    def output_ndim(self) -> int:
        """Number of dimensions of the n-dimensional output."""
        return len(self.output_shape)

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
    def extrapolation(self) -> Evaluator[A] | None:
        """Extrapolation used for evaluating values outside the bounds."""
        return (  # type: ignore [no-any-return]
            self._extrapolation  # type: ignore [attr-defined]
        )

    @extrapolation.setter
    def extrapolation(
        self,
        value: ExtrapolationLike[A] | None,
    ) -> None:
        self._extrapolation = (  # type: ignore [misc, attr-defined]
            _parse_extrapolation(value)
        )

    @property
    @abstractmethod
    def domain(self) -> Region[A]:
        """Domain of the function."""
        pass

    @abstractmethod
    def _evaluate(
        self,
        eval_points: A,
        /,
        *,
        aligned: bool = True,
    ) -> A:
        """
        Evaluate the :term:`functional object`.

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
    def __call__(
        self,
        eval_points: A,
        /,
        *,
        extrapolation: AcceptedExtrapolation[A] = "default",
        grid: Literal[False] = False,
        aligned: bool = True,
    ) -> A:
        pass

    @overload
    def __call__(
        self,
        eval_points: GridPointsLike[A],
        /,
        *,
        extrapolation: AcceptedExtrapolation[A] = "default",
        grid: Literal[True],
        aligned: Literal[True] = True,
    ) -> A:
        pass

    @overload
    def __call__(
        self,
        eval_points: Sequence[GridPointsLike[A]],
        /,
        *,
        extrapolation: AcceptedExtrapolation[A] = "default",
        grid: Literal[True],
        aligned: Literal[False],
    ) -> A:
        pass

    @overload
    def __call__(
        self,
        eval_points: EvalPointsType[A],
        /,
        *,
        extrapolation: AcceptedExtrapolation[A] = "default",
        grid: bool = False,
        aligned: bool = True,
    ) -> A:
        pass

    def __call__(
        self,
        eval_points: EvalPointsType[A],
        /,
        *,
        extrapolation: AcceptedExtrapolation[A] = "default",
        grid: bool = False,
        aligned: bool = True,
    ) -> A:
        """
        Evaluate the :term:`functional object`.

        Evaluate the object at a list of values or a grid.

        Args:
            eval_points: Points where the functions are evaluated.
                An array is expected, whose last dimensions must match the
                :attr:`input_shape <NDFunction.input_shape>` of the
                functions.
                If ``aligned`` is set, then separate points are used for
                each function in the array, and the first dimensions must
                match the :attr:`shape <NDFunction.shape>` of the functions
                accordingly.
                The intermediate dimensions are "batch" dimensions, which
                will be respected in the output.
            extrapolation: Controls the extrapolation mode for elements
                outside the domain range. By default it is used the mode
                defined on the instance of the object.
            grid: Whether to evaluate the results on a grid
                spanned by the input arrays, or at points specified by the
                input arrays. If true the eval_points should be a list or
                object array with the same shape as the
                :attr:`input_shape <NDFunction.input_shape>` of the functions.
                Each element would be an array of grid points for that
                position. The returned evaluations would correspond to those
                of the points in the Cartesian product.
            aligned: Whether the input points are the same for each sample,
                or an array of points per sample is passed.
                If ``aligned`` is set, then separate points are used for
                each function in the array, and the first dimensions of
                ``eval_points`` must match the
                :attr:`shape <NDFunction.shape>` of the functions.

        Returns:
            Array containing the values of each function at the points
            specified in eval_points.
            The first dimensions correspond to
            :attr:`shape <NDFunction.shape>`, while the last dimensions
            correspond to :attr:`output_shape <NDFunction.output_shape>`.
            The intermediate dimensions would be the "batch" dimensions
            used in `eval_points`.

        """
        from ._functions import _evaluate_grid

        if grid:  # Evaluation of a grid performed in auxiliar function

            return _evaluate_grid(
                self,
                eval_points,
                extrapolation=extrapolation,
                aligned=aligned,
            )

        eval_points = cast(A, eval_points)

        extrapolation = (
            self.extrapolation
            if extrapolation == "default"
            # Mypy bug: https://github.com/python/mypy/issues/16465
            else _parse_extrapolation(extrapolation)  # type: ignore [arg-type]
        )

        eval_points = check_array_namespace(
            eval_points,
            namespace=self.array_backend,
            allow_array_like=True,
        )[0]

        eval_points = check_evaluation_points(
            eval_points,
            aligned=aligned,
            shape=self.shape,
            input_shape=self.input_shape,
        )

        res_evaluation = self._evaluate(
            eval_points,
            aligned=aligned,
        )

        if extrapolation is not None:

            xp = array_namespace(eval_points)

            contained_points_idx = self.domain.contains(eval_points)

            if not xp.all(contained_points_idx):

                res_extrapolation = extrapolation(
                    self,
                    eval_points,
                    aligned=aligned,
                )

                return xp.where(
                    contained_points_idx[
                        (...,) + (xp.newaxis,) * self.output_ndim
                    ],
                    res_evaluation,
                    res_extrapolation,
                )

        # Normal evaluation if there are no points to extrapolate.
        return res_evaluation

    @abstractmethod
    def copy(
        self,
        *,
        deep: bool = False,  # For Pandas compatibility
        dataset_name: Optional[str] = None,
        argument_names: Optional[LabelTupleLike] = None,
        coordinate_names: Optional[LabelTupleLike] = None,
        sample_names: Optional[LabelTupleLike] = None,
        extrapolation: Optional[ExtrapolationLike[A]] = None,
    ) -> Self:
        """Make a copy of the object."""
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

    @abstractmethod
    def __eq__(self, other: Self) -> NDArrayBool:  # type: ignore[override]
        """Elementwise equality, as with arrays."""
        return NotImplemented

    def __ne__(self, other: Self) -> NDArrayBool:  # type: ignore[override]
        """Return for `self != other` (element-wise in-equality)."""
        result = self.__eq__(other)
        if result is NotImplemented:
            return NotImplemented

        return ~result

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
        other: A | float,
    ) -> Self:
        """Multiplication for FData object."""
        pass

    @abstractmethod
    def __rmul__(
        self,
        other: A | float,
    ) -> Self:
        """Multiplication for FData object."""
        pass

    @abstractmethod
    def __truediv__(
        self,
        other: A | float,
    ) -> Self:
        """Division for FData object."""
        pass

    @abstractmethod
    def __rtruediv__(
        self,
        other: A | float,
    ) -> Self:
        """Right division for FData object."""
        pass

    @abstractmethod
    def __neg__(self) -> Self:
        """Negation of FData object."""
        pass


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
