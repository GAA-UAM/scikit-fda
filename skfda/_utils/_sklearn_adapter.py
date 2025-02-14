from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any, Generic, TypeVar, overload

import sklearn.base

if TYPE_CHECKING:
    from ..typing._numpy import NDArrayFloat, NDArrayInt

SelfType = TypeVar("SelfType")
TransformerNoTarget = TypeVar(
    "TransformerNoTarget",
    bound="TransformerMixin[Any, Any, None]",
)
Input = TypeVar("Input", contravariant=True)
Output = TypeVar("Output", covariant=True)
Target = TypeVar("Target", contravariant=True)
TargetPrediction = TypeVar("TargetPrediction")


class BaseEstimator(  # noqa: D101
    ABC,
    sklearn.base.BaseEstimator,  # type: ignore[misc]
):
    pass  # noqa: WPS604


class TransformerMixin(  # noqa: D101
    ABC,
    sklearn.base.TransformerMixin,  # type: ignore[misc]
    Generic[Input, Output, Target],
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

    def fit(  # noqa: D102
        self: SelfType,
        X: Input,
        y: Target | None = None,
    ) -> SelfType:
        fit = getattr(super(), "fit", None)
        if fit:
            return super().fit(X, y)
        
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

    def fit_transform(  # noqa: D102
        self,
        X: Input,
        y: Target | None = None,
        **fit_params: Any,
    ) -> Output:
        fit_transform = getattr(super(), "fit_transform", None)
        if fit_transform:
            return fit_transform(X, y, **fit_params)

        if y is None:
            return self.fit(  # type: ignore[no-any-return]
                X,
                **fit_params,
            ).transform(X)

        return self.fit(  # type: ignore[no-any-return]
            X,
            y,
            **fit_params,
        ).transform(X)


class InductiveTransformerMixin(  # noqa: D101
    TransformerMixin[Input, Output, Target],
):

    @abstractmethod
    def transform(  # noqa: D102
        self: SelfType,
        X: Input,
    ) -> Output:
        return super().transform(X)


class OutlierMixin(  # noqa: D101
    ABC,
    sklearn.base.OutlierMixin,  # type: ignore[misc]
    Generic[Input],
):

    def fit_predict(  # noqa: D102
        self,
        X: Input,
        y: object = None,
    ) -> NDArrayInt:
        fit_predict = getattr(super(), "fit_predict", None)
        if fit_predict:
            return fit_predict(X, y)
        
        return self.fit(X, y).predict(X)  # type: ignore[no-any-return]


class ClassifierMixin(  # noqa: D101
    ABC,
    sklearn.base.ClassifierMixin,  # type: ignore[misc]
    Generic[Input, TargetPrediction],
):
    def fit(  # noqa: D102
        self: SelfType,
        X: Input,
        y: TargetPrediction,
    ) -> SelfType:
        fit = getattr(super(), "fit", None)
        if fit:
            return super().fit(X, y)
        
        return self

    @abstractmethod
    def predict(  # noqa: D102
        self: SelfType,
        X: Input,
    ) -> TargetPrediction:
        return super().predict(X)

    def score(  # noqa: D102
        self,
        X: Input,
        y: Target,
        sample_weight: NDArrayFloat | None = None,
    ) -> float:
        return super().score(  # type: ignore[no-any-return]
            X,
            y,
            sample_weight=sample_weight,
        )


class ClusterMixin(  # noqa: D101
    ABC,
    sklearn.base.ClusterMixin,  # type: ignore[misc]
    Generic[Input],
):
    def fit_predict(  # noqa: D102
        self,
        X: Input,
        y: object = None,
    ) -> NDArrayInt:
        return super().fit_predict(X, y)  # type: ignore[no-any-return]


class RegressorMixin(  # noqa: D101
    ABC,
    sklearn.base.RegressorMixin,  # type: ignore[misc]
    Generic[Input, TargetPrediction],
):
    def fit(  # noqa: D102
        self: SelfType,
        X: Input,
        y: TargetPrediction,
    ) -> SelfType:
        fit = getattr(super(), "fit", None)
        if fit:
            return super().fit(X, y)
        
        return self

    @abstractmethod
    def predict(  # noqa: D102
        self: SelfType,
        X: Input,
    ) -> TargetPrediction:
        return super().predict(X)

    def score(  # noqa: D102
        self,
        X: Input,
        y: TargetPrediction,
        sample_weight: NDArrayFloat | None = None,
    ) -> float:
        from ..misc.scoring import r2_score
        y_pred = self.predict(X)
        return r2_score(
            y,
            y_pred,
            sample_weight=sample_weight,
        )
