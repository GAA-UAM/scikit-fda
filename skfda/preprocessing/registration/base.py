# -*- coding: utf-8 -*-
"""Registration method.
This module contains the abstract base class for all registration methods.
"""

from abc import ABC
from sklearn.base import BaseEstimator, TransformerMixin
from ... import FData

class RegistrationTransformer(ABC, BaseEstimator, TransformerMixin):
    """Base class for the registration methods."""

    def score(self, X: FData, y=None):
        r"""Returns the percentage of total variation removed.

        Computes the squared multiple correlation index of the proportion of
        the total variation due to phase, defined as:

        .. math::
            R^2 = \frac{\text{MSE}_{phase}}{\text{MSE}_{total}},

        where :math:`\text{MSE}_{total}` is the mean squared error and
        :math:`\text{MSE}_{phase}` is the mean squared error due to the phase
        explained by the registration procedure. See
        :class:`~.validation.AmplitudePhaseDecomposition` for a detailed
        explanation.

        Args:
            X (FData): Functional data to be registered
            y (Ignored): Ignored, only for API conventions.

        Returns:
            float.

        See also:
            :class:`~.validation.AmplitudePhaseDecomposition`
            :class:`~.validation.LeastSquares`
            :class:`~.validation.SobolevLeastSquares`
            :class:`~.validation.PairwiseCorrelation`

        """
        from .validation import AmplitudePhaseDecomposition

        return AmplitudePhaseDecomposition()(self, X, y)
