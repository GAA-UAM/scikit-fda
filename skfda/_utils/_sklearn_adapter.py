from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Generic, Optional, TypeVar, overload

import sklearn.base

from ..representation._typing import NDArrayFloat

SelfType = TypeVar("SelfType")
TransformerNoTarget = TypeVar(
    "TransformerNoTarget",
    bound="TransformerMixin[Any, Any, None]",
)
Input = TypeVar("Input")
Output = TypeVar("Output")
Target = TypeVar("Target", contravariant=True)


class BaseEstimator(
    ABC,
    sklearn.base.BaseEstimator,  # type: ignore[misc]
):
    pass


class TransformerMixin(
    ABC,
    Generic[Input, Output, Target],
    # sklearn.base.TransformerMixin, # Inherit in the future
):

    @overload
    def fit(
        self: TransformerNoTarget,
        X: Input,
    ) -> TransformerNoTarget:
        pass

    @overload
    def fit(
        self: SelfType,
        X: Input,
        y: Target,
    ) -> SelfType:
        pass

    def fit(
        self: SelfType,
        X: Input,
        y: Optional[Target] = None,
    ) -> SelfType:
        return self

    @overload
    def fit_transform(
        self: TransformerNoTarget,
        X: Input,
    ) -> Output:
        pass

    @overload
    def fit_transform(
        self,
        X: Input,
        y: Target,
    ) -> Output:
        pass

    def fit_transform(
        self,
        X: Input,
        y: Optional[Target] = None,
        **fit_params: Any,
    ) -> Output:
        if y is None:
            return self.fit(X, **fit_params).transform(X)  # type: ignore
        else:
            return self.fit(X, y, **fit_params).transform(X)  # type: ignore


class InductiveTransformerMixin(
    TransformerMixin[Input, Output, Target],
):

    @abstractmethod
    def transform(
        self: SelfType,
        X: Input,
    ) -> Output:
        pass


class ClassifierMixin(
    ABC,
    Generic[Input, Target],
    sklearn.base.ClassifierMixin,  # type: ignore[misc]
):
    def score(
        self,
        X: Input,
        y: Target,
        sample_weight: NDArrayFloat | None = None,
    ) -> NDArrayFloat:
        return super().score(X, y, sample_weight=sample_weight)
