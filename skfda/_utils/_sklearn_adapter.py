from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Generic, Optional, TypeVar, overload

import sklearn.base

SelfType = TypeVar("SelfType")
Input = TypeVar("Input")
Output = TypeVar("Output")
Target = TypeVar("Target")


class BaseEstimator(
    ABC,
    sklearn.base.BaseEstimator,  # type: ignore[misc]
):
    pass


class TransformerMixin(
    ABC,
    Generic[Input, Output, Target],
    sklearn.base.TransformerMixin,  # type: ignore[misc]
):

    def fit(
        self: SelfType,
        X: Input,
        y: Optional[Target] = None,
    ) -> SelfType:

        return self

    @overload  # type: ignore[misc]
    def fit_transform(
        self,
        X: Input,
        y: Optional[Target] = None,
    ) -> Output:
        pass

    def fit_transform(
        self,
        X: Input,
        y: Optional[Target] = None,
        **fit_params: Any,
    ) -> Output:

        return super().fit_transform(X, y, **fit_params)


class InductiveTransformerMixin(
    TransformerMixin[Input, Output, Target],
):

    @abstractmethod
    def transform(
        self: SelfType,
        X: Input,
    ) -> Output:

        pass
