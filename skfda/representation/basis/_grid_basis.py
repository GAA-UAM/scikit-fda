
from __future__ import annotations

from typing import TypeVar

from ...typing._numpy import NDArrayFloat
from ..evaluator import Evaluator
from ..interpolation import SplineInterpolation
from ._basis import Basis
from ...typing._base import GridPointsLike
from ..._utils import _to_grid_points

T = TypeVar("T", bound="GridBasis")


class GridBasis(Basis):
    """
    Basis representing a grid of points.

    Defines a basis whose elements are one in each point of the grid and zero
    in the rest.
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
        domain_range = tuple(
            (s[0], s[-1]) for s in self.grid_points
        )
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
