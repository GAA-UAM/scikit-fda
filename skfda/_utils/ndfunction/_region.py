from abc import abstractmethod
from typing import Protocol, TypeVar

from array_api_compat import array_namespace
from typing_extensions import override

from ._array_api import Array, DType, Shape

A = TypeVar("A", bound=Array[Shape, DType])


class Region(Protocol[A]):
    """
    Protocol for regions of points.

    This protocol is used to define the points in which extrapolation takes
    place (points outside the region).

    It is also used to define a bounding box for representation and
    integration limits.

    """

    @property
    @abstractmethod
    def bounding_box(self) -> tuple[A, A]:
        """
        A tuple of two arrays representing a boundary box for the data.

        The first array contains the lower values allowed for each dimension.
        The second array contains the higher values allowed.
        """

    @abstractmethod
    def contains(self, points: A) -> A:
        """
        Check that the points in ```points``` are inside the region.

        Args:
            points: Array representing a stacked set of points. Its
                trailing dimensions must match with the dimensions of the
                space.

        Returns:
            A boolean array with the same shape as the leading
            dimensions of ``points``, where ``True`` indicates that
            the corresponding point is contained in the region and
            ``False`` that it is outside the region.

        """


class AxisAlignedBox(Region[A]):
    """
    A box region with edges parallel to the coordinate axes.

    The box is completely determined by the lower and upper values for each
    coordinate.
    """

    def __init__(
        self,
        lower: A,
        upper: A,
    ) -> None:
        # If we want to allow broadcastable arrays we can do so in the future.
        # For now, they have to had the same dimensions as the points.
        assert lower.shape == upper.shape
        self.lower = lower
        self.upper = upper

    @override
    @property
    def bounding_box(self) -> tuple[A, A]:
        return self.lower, self.upper

    @override
    def contains(
        self,
        points: A,
    ) -> A:
        xp = array_namespace(points, self.lower, self.upper)

        if xp.isdtype(points.dtype, "complex floating"):
            lower_check = xp.logical_and(
                xp.real(self.lower) <= xp.real(points),
                xp.imag(self.lower) <= xp.imag(points),
            )
            upper_check = xp.logical_and(
                xp.real(points) <= xp.real(self.upper),
                xp.imag(points) <= xp.imag(self.upper),
            )
        else:
            lower_check = self.lower <= points  # type: ignore[operator]
            upper_check = points <= self.upper  # type: ignore[operator]

        coordinates_in = xp.logical_and(
            lower_check,
            upper_check,
        )

        trailing_axes = tuple(
            -(axis + 1) for axis in range(self.lower.ndim)
        )

        return xp.all(  # type: ignore[no-any-return]
            coordinates_in,
            axis=trailing_axes,
        )
