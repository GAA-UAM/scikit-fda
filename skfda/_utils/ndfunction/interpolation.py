"""
Module to interpolate functional data objects.

Defines methods to evaluate points in non-measured regions inside
the :term:`domain` range.

"""
from __future__ import annotations

import abc
from collections.abc import Callable, Sequence
from typing import TYPE_CHECKING, Generic, TypeVar

import numpy as np
from scipy.interpolate import (
    PchipInterpolator,
    RegularGridInterpolator,
    make_interp_spline,
)
from typing_extensions import override

from ._array_api import Array, DType, Shape
from .evaluator import Evaluator

if TYPE_CHECKING:
    from ...representation.grid import FDataGrid
    from ._ndfunction import NDFunction

A = TypeVar("A", bound=Array[Shape, DType])


class _BaseInterpolation(Evaluator[A]):
    """ABC for interpolations."""

    @abc.abstractmethod
    def _evaluate_aligned(
        self,
        fdata: FDataGrid,
        eval_points: A,
    ) -> A:
        """Evaluate at aligned points."""

    def _evaluate_unaligned(
        self,
        fdata: FDataGrid,
        eval_points: A,
    ) -> A:
        """Evaluate at unaligned points."""
        xp = fdata.array_backend
        return xp.concat([  # type: ignore[no-any-return]
            self._evaluate_aligned(f, e)
            for f, e in zip(fdata, eval_points, strict=False)
        ])

    @override
    def __call__(
        self,
        function: NDFunction[A],
        /,
        eval_points: A,
        *,
        aligned: bool = True,
    ) -> A:
        from ...representation.grid import FDataGrid

        assert isinstance(function, FDataGrid)

        return (
            self._evaluate_aligned(function, eval_points)
            if aligned
            else self._evaluate_unaligned(function, eval_points)
        )


class _RegularGridInterpolatorWrapper(Generic[A]):

    def __init__(
        self,
        fdatagrid: FDataGrid,
        interpolation_order: int,
    ) -> None:
        self.fdatagrid = fdatagrid
        self.interpolation_order = interpolation_order

        if self.interpolation_order == 0:
            method = "nearest"
        elif self.interpolation_order == 1:
            method = "linear"
        elif self.interpolation_order == 3:  # noqa: PLR2004
            method = "cubic"
        elif self.interpolation_order == 5:  # noqa: PLR2004
            method = "quintic"
        else:
            msg = f"Invalid interpolation order: {self.interpolation_order}."
            raise ValueError(msg)
        self.interpolator = RegularGridInterpolator(
            self.fdatagrid.grid_points,
            np.moveaxis(self.fdatagrid.data_matrix, 0, -2),
            method=method,
            bounds_error=False,
            fill_value=None,
        )

    def __call__(
        self,
        eval_points: A,
    ) -> A:
        xp = self.fdatagrid.array_backend
        return xp.moveaxis(  # type: ignore[no-any-return]
            self.interpolator(eval_points),
            0,
            1,
        )


class SplineInterpolation(_BaseInterpolation[A]):
    """
    Spline interpolation.

    Spline interpolation of discretized functional objects. Implements
    different interpolation methods based in splines, using the sample
    points of the grid as nodes to interpolate.

    See the interpolation example to a detailled explanation.

    Attributes:
        interpolation_order (int, optional): Order of the interpolation, 1
            for linear interpolation, 2 for cuadratic, 3 for cubic and so
            on. In case of curves and surfaces there is available
            interpolation up to degree 5. For higher dimensional objects
            only linear or nearest interpolation is available. Default
            lineal interpolation.
        smoothness_parameter (float, optional): Penalisation to perform
            smoothness interpolation. Option only available for curves and
            surfaces. If 0 the residuals of the interpolation will be 0.
            Defaults 0.
        monotone (boolean, optional): Performs monotone interpolation in
            curves using a PCHIP interpolator. Only valid for curves (domain
            dimension equal to 1) and interpolation order equal to 1 or 3.
            Defaults false.

    """

    def __init__(
        self,
        interpolation_order: int = 1,
        *,
        monotone: bool = False,
    ) -> None:
        self.interpolation_order = interpolation_order
        self.monotone = monotone

    def _get_interpolator_1d(
        self,
        fdatagrid: FDataGrid,
    ) -> Callable[[A], A]:
        if (
            isinstance(self.interpolation_order, Sequence)
            or not 1 <= self.interpolation_order <= 5  # noqa: PLR2004
        ):
            msg = (
                f"Invalid degree of interpolation "
                f"({self.interpolation_order}). It must be an integer greater "
                f"than 0 and lower or equal than 5."
            )
            raise ValueError(msg)

        if self.monotone and self.interpolation_order not in {1, 3}:
            msg = (
                f"Monotone interpolation of degree "
                f"{self.interpolation_order} not supported."
            )
            raise ValueError(msg)

        # Monotone interpolation of degree 1 is performed with linear spline
        monotone = self.monotone and self.interpolation_order > 1

        if monotone:
            return PchipInterpolator(  # type: ignore[no-any-return]
                fdatagrid.grid_points[0],
                fdatagrid.data_matrix,
                axis=1,
            )

        return make_interp_spline(  # type: ignore[no-any-return]
            fdatagrid.grid_points[0],
            fdatagrid.data_matrix,
            k=self.interpolation_order,
            axis=1,
            # Orders 0 and 1 behave well
            check_finite=self.interpolation_order > 1,
        )

    def _get_interpolator_nd(
        self,
        fdatagrid: FDataGrid,
    ) -> Callable[[A], A]:

        return _RegularGridInterpolatorWrapper[A](
            fdatagrid,
            interpolation_order=self.interpolation_order,
        )

    def _get_interpolator(
        self,
        fdatagrid: FDataGrid,
    ) -> Callable[[A], A]:

        if fdatagrid.dim_domain == 1:
            return self._get_interpolator_1d(fdatagrid)

        if self.monotone:
            msg = (
                "Monotone interpolation is only supported with "
                "domain dimension equal to 1."
            )
            raise ValueError(msg)

        return self._get_interpolator_nd(fdatagrid)

    def _evaluate_aligned(
        self,
        fdata: FDataGrid,
        eval_points: A,
    ) -> A:

        interpolator = self._get_interpolator(fdata)
        xp = fdata.array_backend
        return xp.reshape(  # type: ignore[no-any-return]
            interpolator(eval_points),
            (fdata.n_samples, -1, fdata.dim_codomain),
        )

    def __repr__(self) -> str:
        return (
            f"{type(self).__name__}("
            f"interpolation_order={self.interpolation_order}, "
            f"monotone={self.monotone})"
        )

    def __eq__(self, other: object) -> bool:
        return (
            super().__eq__(other)
            and isinstance(other, SplineInterpolation)
            and self.interpolation_order == other.interpolation_order
            and self.monotone == other.monotone
        )
