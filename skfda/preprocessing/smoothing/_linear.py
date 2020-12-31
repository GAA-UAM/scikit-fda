# -*- coding: utf-8 -*-
"""Linear smoother.

This module contains the abstract base class for all linear smoothers.

"""
import abc

from sklearn.base import BaseEstimator, TransformerMixin

import numpy as np

from ... import FDataGrid


def _check_r_to_r(f):
    if f.dim_domain != 1 or f.dim_codomain != 1:
        raise NotImplementedError("Only accepts functions from R to R")


class _LinearSmoother(abc.ABC, BaseEstimator, TransformerMixin):
    """Linear smoother.

    Abstract base class for all linear smoothers. The subclasses must override
    ``hat_matrix`` to define the smoothing or 'hat' matrix.

    """

    def __init__(self, *,
                 output_points=None):
        self.output_points = output_points

    def hat_matrix(self, input_points=None, output_points=None):
        cached_input_points = getattr(self, "input_points_", None)
        cached_output_points = getattr(self, "output_points_", None)

        # Use the fitted points if they are not provided
        if input_points is None:
            input_points = cached_input_points
        if output_points is None:
            output_points = cached_output_points

        if (cached_input_points is not None and
            np.array_equal(input_points, cached_input_points) and
                np.array_equal(output_points, cached_output_points)):
            cached_hat_matrix = getattr(self, "_cached_hat_matrix", None)
            if cached_hat_matrix is None:
                self.cached_hat_matrix = self._hat_matrix(
                    input_points=self.input_points_,
                    output_points=self.output_points_
                )
            return self.cached_hat_matrix

        else:
            # We only cache the matrix for the fit points
            return self._hat_matrix(
                input_points=self.input_points_,
                output_points=self.output_points_
            )

    @abc.abstractmethod
    def _hat_matrix(self, input_points, output_points):
        pass

    def _more_tags(self):
        return {
            'X_types': []
        }

    def fit(self, X: FDataGrid, y=None):
        """Compute the hat matrix for the desired output points.

        Args:
            X (FDataGrid):
                The data whose points are used to compute the matrix.
            y : Ignored
        Returns:
            self (object)

        """
        _check_r_to_r(X)

        self.input_points_ = X.grid_points[0]
        self.output_points_ = (self.output_points
                               if self.output_points is not None
                               else self.input_points_)

        # Force caching the hat matrix
        self.hat_matrix()

        return self

    def transform(self, X: FDataGrid, y=None):
        """Multiplies the hat matrix for the functions values to smooth them.

        Args:
            X (FDataGrid):
                The data to smooth.
            y : Ignored
        Returns:
            FDataGrid: Functional data smoothed.

        """

        assert all(self.input_points_ == X.grid_points[0])

        # The matrix is cached
        return X.copy(data_matrix=self.hat_matrix() @ X.data_matrix,
                      grid_points=self.output_points_)

    def score(self, X, y):
        """Returns the generalized cross validation (GCV) score.

        Args:
            X (FDataGrid):
                The data to smooth.
            y (FDataGrid):
                The target data. Typically the same as ``X``.
        Returns:
            float: Generalized cross validation score.

        """
        from .validation import LinearSmootherGeneralizedCVScorer

        return LinearSmootherGeneralizedCVScorer()(self, X, y)
