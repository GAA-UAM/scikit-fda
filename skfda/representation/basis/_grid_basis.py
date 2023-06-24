from __future__ import annotations

from typing import Any, TypeVar

from ..._utils import _to_grid_points
from ...typing._base import GridPointsLike
from ...typing._numpy import NDArrayFloat
from ._basis import Basis

T = TypeVar("T", bound="_GridBasis")


class _GridBasis(Basis):
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

    """

    def __init__(
        self,
        *,
        grid_points: GridPointsLike,
    ) -> None:
        """Basis constructor."""
        self.grid_points = _to_grid_points(grid_points)
        domain_range = tuple((s[0], s[-1]) for s in self.grid_points)
        super().__init__(
            domain_range=domain_range,
            n_basis=len(self.grid_points[0]),
        )

    def _evaluate(self, eval_points: NDArrayFloat) -> NDArrayFloat:
        raise NotImplementedError(
            "Evaluation is not implemented in this basis",
        )

    def __eq__(self, other: Any) -> bool:
        return (
            super().__eq__(other)
            and self.grid_points == other.grid_points
        )

    def __repr__(self) -> str:
        return f"{type(self).__name__}(grid_points={self.grid_points}) "

    def __hash__(self) -> int:
        return hash(
            (
                super(),
                (
                    tuple(grid_point_axis)
                    for grid_point_axis in self.grid_points
                ),
            ),
        )
