# -*- coding: utf-8 -*-
"""Linear smoother.

This module contains the abstract base class for all linear smoothers.

"""
from __future__ import annotations

import abc
from typing import Any, Mapping, Optional

import numpy as np

from ..._utils import _to_grid_points
from ..._utils._sklearn_adapter import BaseEstimator, TransformerMixin
from ...representation import FDataGrid
from ...typing._base import GridPoints, GridPointsLike
from ...typing._numpy import NDArrayFloat


class _LinearSmoother(
    BaseEstimator,
    TransformerMixin[FDataGrid, FDataGrid, object],
):
    """Linear smoother.

    Abstract base class for all linear smoothers. The subclasses must override
    ``hat_matrix`` to define the smoothing or 'hat' matrix.

    """
    input_points_: GridPoints
    output_points_: GridPoints

    def __init__(
        self,
        *,
        output_points: Optional[GridPointsLike] = None,
    ):
        self.output_points = output_points

    def hat_matrix(
        self,
        input_points: Optional[GridPointsLike] = None,
        output_points: Optional[GridPointsLike] = None,
    ) -> NDArrayFloat:

        # Use the fitted points if they are not provided
        if input_points is None:
            input_points = self.input_points_
        if output_points is None:
            output_points = self.output_points_

        return self._hat_matrix(
            input_points=self.input_points_,
            output_points=self.output_points_,
        )

    @abc.abstractmethod
    def _hat_matrix(
        self,
        input_points: GridPointsLike,
        output_points: GridPointsLike,
    ) -> NDArrayFloat:
        pass

    def _more_tags(self) -> Mapping[str, Any]:
        return {
            'X_types': [],
        }

    def fit(
        self,
        X: FDataGrid,
        y: object = None,
    ) -> _LinearSmoother:
        """Compute the hat matrix for the desired output points.

        Args:
            X: The data whose points are used to compute the matrix.
            y: Ignored.

        Returns:
            self

        """
        self.input_points_ = X.grid_points
        self.output_points_ = (
            _to_grid_points(self.output_points)
            if self.output_points is not None
            else self.input_points_
        )

        self.hat_matrix_ = self.hat_matrix()

        return self

    def transform(
        self,
        X: FDataGrid,
        y: object = None,
    ) -> FDataGrid:
        """Multiply the hat matrix with the function values to smooth them.

        Args:
            X: The data to smooth.
            y: Ignored

        Returns:
            Functional data smoothed.

        """
        assert all(
            np.array_equal(i, s) for i, s in zip(
                self.input_points_,
                X.grid_points,
            )
        )

        # The matrix is cached
        return X.copy(
            data_matrix=self.hat_matrix_ @ X.data_matrix,
            grid_points=self.output_points_,
        )

    def score(
        self,
        X: FDataGrid,
        y: FDataGrid,
    ) -> float:
        """Return the generalized cross validation (GCV) score.

        Args:
            X: The data to smooth.
            y: The target data. Typically the same as ``X``.

        Returns:
            Generalized cross validation score.

        """
        from .validation import LinearSmootherGeneralizedCVScorer

        return LinearSmootherGeneralizedCVScorer()(self, X, y)
