# -*- coding: utf-8 -*-
"""Registration methods base class.

This module contains the abstract base class for all registration methods.

"""
from __future__ import annotations

from abc import abstractmethod
from typing import Any, TypeVar, overload

from ..._utils._sklearn_adapter import (
    BaseEstimator,
    InductiveTransformerMixin,
    TransformerMixin,
)
from ...representation import FData

SelfType = TypeVar("SelfType")
Input = TypeVar("Input", bound=FData)
Output = TypeVar("Output", bound=FData)


class RegistrationTransformer(
    BaseEstimator,
    TransformerMixin[Input, Output, object],
):
    """Base class for the registration methods."""

    def fit(
        self: SelfType,
        X: Input,
        y: object = None,
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
        y: object = None,
    ) -> Output:
        pass

    def fit_transform(  # noqa: WPS612
        self,
        X: Input,
        y: object = None,
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
        return super().fit_transform(
            X,
            y,
            **fit_params,
        )

    def score(self, X: Input, y: object = None) -> float:
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

        return AmplitudePhaseDecomposition()(self, X, X)


class InductiveRegistrationTransformer(
    RegistrationTransformer[Input, Output],
    InductiveTransformerMixin[Input, Output, object],
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
