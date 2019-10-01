# -*- coding: utf-8 -*-
"""Registration method.
This module contains the abstract base class for all registration methods.
"""

from abc import ABC
from sklearn.base import BaseEstimator, TransformerMixin
from ... import FData

class RegistrationTransformer(ABC, BaseEstimator, TransformerMixin):

    def score(self, X: FData, y=None):
        r"""Returns the percentage of total variation removed.

        Computes the squared multiple correlation index of the proportion of
        the total variation due to phase, defined as:

        .. math::
            R^2 = \frac{\text{MSE}_{phase}}{\text{MSE}_{total}},

        where :math:`\text{MSE}_{total}` is the mean squared error and
        :math:`\text{MSE}_{phase}` is the mean squared error due to the phase
        explained by the registration procedure. See :func:`mse_decomposition`
        for a detailed explanation.

        Args:
            X (FData): Functional data to be registered
            y : Ignored

        Returns:
            float.

        See also:
            :class:`RegistrationScorer <RegistrationScorer>`
            :func:`mse_r_squared <mse_r_squared>`
            :func:`least_squares <least_squares>`
            :func:`sobolev_least_squares <sobolev_least_squares>`
            :func:`pairwise_correlation <pairwise_correlation>`

        """
        from .validation import RegistrationScorer

        return RegistrationScorer()(self, X, y)
