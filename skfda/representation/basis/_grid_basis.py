from __future__ import annotations

from typing import Any, TypeVar

from ..._utils import _to_grid_points
from ...typing._base import GridPointsLike
from ...typing._numpy import NDArrayFloat
from ..evaluator import Evaluator
from ..interpolation import SplineInterpolation
from ._basis import Basis

T = TypeVar("T", bound="GridBasis")


class GridBasis(Basis):
    """
    Basis associated to a grid of points.

    Used to express functions whose values are known in a grid of points.
    Each basis function is guaranteed to be zero in all points of the grid
    except one, where it is one. The intermediate values are interpolated
    depending on the interpolation object passed as argument.

    This basis is used internally as an alternate representation of a
    FDataGrid object. In certain cases, it is more convenient to work with
    FDataGrid objects as a basis representation in this basis since, then,
    all functional data can be represented as an FDataBasis object.

    Parameters:
        grid_points: Grid of points where the functions are evaluated.
        interpolation: Interpolation object used to interpolate the values
            between grid points. By default, a spline is used

    """

    def __init__(
        self,
        *,
        grid_points: GridPointsLike,
        interpolation: Evaluator | None = None,
    ) -> None:
        """Basis constructor."""
        self.grid_points = _to_grid_points(grid_points)
        self.interpolation = interpolation
        domain_range = tuple((s[0], s[-1]) for s in self.grid_points)
        super().__init__(
            domain_range=domain_range,
            n_basis=len(grid_points[0]),
        )

    @property
    def interpolation(self) -> Evaluator:
        """Define the type of interpolation applied in `evaluate`."""
        return self._interpolation

    @interpolation.setter
    def interpolation(self, new_interpolation: Evaluator | None) -> None:

        if new_interpolation is None:
            new_interpolation = SplineInterpolation()

        self._interpolation = new_interpolation

    def _evaluate(self, eval_points: NDArrayFloat) -> NDArrayFloat:
        return self.interpolation(eval_points)

    def __eq__(self, other: Any) -> bool:
        return (
            super().__eq__(other)
            and self.interpolation == other.interpolation
            and self.grid_points == other.grid_points
        )

    def __ne__(self, other: Any) -> bool:
        return not self.__eq__(other)

    def __repr__(self) -> str:
        return (
            f"{type(self).__name__}("
            f"grid_points={self.grid_points}, "
            f"interpolation={self.interpolation})"
        )

    def __hash__(self) -> int:
        return (
            super().__hash__()
            ^ hash(self.interpolation)
            ^ hash(
                self.grid_points,
            )
        )
