# -*- coding: utf-8 -*-
"""Registration methods base class.

This module contains the abstract base class for all registration methods.

"""
from __future__ import annotations

from abc import abstractmethod
from typing import Any, TypeVar, overload

from ... import FData
from ..._utils import (
    BaseEstimator,
    InductiveTransformerMixin,
    TransformerMixin,
)

SelfType = TypeVar("SelfType")
Input = TypeVar("Input", bound=FData)
Output = TypeVar("Output", bound=FData)


class RegistrationTransformer(
    BaseEstimator,
    TransformerMixin[Input, Output, None],
):
    """Base class for the registration methods."""

    def fit(
        self: SelfType,
        X: Input,
        y: None = None,
    ) -> SelfType:
        """
        Fit the registration model.

        Args:
            X: Original (unregistered) training data.
            y: Ignored.

        Returns:
            Returns the instance itself.

        """
        return self

    @overload  # type: ignore[misc]
    def fit_transform(
        self,
        X: Input,
        y: None = None,
    ) -> Output:
        pass

    def fit_transform(
        self,
        X: Input,
        y: None = None,
        **fit_params: Any,
    ) -> Output:
        """
        Fit the registration model and return the registered data.

        Args:
            X: Original (unregistered) training data.
            y: Ignored.
            fit_params: Additional fit parameters.

        Returns:
            Registered training data.

        """
        return super().fit_transform(  # type: ignore[call-arg]
            X,
            y,
            **fit_params,
        )

    def score(self, X: Input, y: None = None) -> float:
        r"""
        Return the percentage of total variation removed.

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
            X: Functional data to be registered
            y: Ignored, only for API conventions.

        Returns:
            Registration score.

        See also:
            :class:`~.validation.AmplitudePhaseDecomposition`
            :class:`~.validation.LeastSquares`
            :class:`~.validation.SobolevLeastSquares`
            :class:`~.validation.PairwiseCorrelation`

        """
        from .validation import AmplitudePhaseDecomposition

        return AmplitudePhaseDecomposition()(self, X, y)


class InductiveRegistrationTransformer(
    RegistrationTransformer[Input, Output],
    InductiveTransformerMixin[Input, Output, None],
):

    @abstractmethod
    def transform(
        self: SelfType,
        X: Input,
    ) -> Output:
        """
        Register new data.

        Args:
            X: Original (unregistered) data.

        Returns:
            Registered data.

        """
        pass
