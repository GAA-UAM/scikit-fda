from __future__ import annotations

from typing import Any, Literal, TypeVar, Union, overload

from scipy.sparse import csr_matrix
from sklearn.neighbors import KNeighborsTransformer as _KNeighborsTransformer

from skfda._utils._sklearn_adapter import InductiveTransformerMixin

from ..._utils._neighbors_base import AlgorithmType, KNeighborsMixin
from ...misc.metrics import l2_distance
from ...representation import FData
from ...typing._metric import Metric
from ...typing._numpy import NDArrayFloat

InputBound = Union[NDArrayFloat, FData]
Input = TypeVar("Input", contravariant=True, bound=InputBound)
Target = TypeVar("Target")
SelfType = TypeVar("SelfType", bound="KNeighborsTransformer[Any, Any]")


class KNeighborsTransformer(
    KNeighborsMixin[Input, Any],
    InductiveTransformerMixin[Input, NDArrayFloat, Any],
):

    n_neighbors: int

    @overload
    def __init__(
        self: KNeighborsTransformer[NDArrayFloat],
        *,
        mode: Literal["connectivity", "distance"] = "distance",
        n_neighbors: int = 5,
        algorithm: AlgorithmType = 'auto',
        leaf_size: int = 30,
        metric: Literal["precomputed"],
        n_jobs: int | None = None,
    ) -> None:
        pass

    @overload
    def __init__(
        self: KNeighborsTransformer[InputBound],
        *,
        mode: Literal["connectivity", "distance"] = "distance",
        n_neighbors: int = 5,
        algorithm: AlgorithmType = 'auto',
        leaf_size: int = 30,
        n_jobs: int | None = None,
    ) -> None:
        pass

    @overload
    def __init__(
        self,
        *,
        mode: Literal["connectivity", "distance"] = "distance",
        n_neighbors: int = 5,
        algorithm: AlgorithmType = 'auto',
        leaf_size: int = 30,
        metric: Metric[Input] = l2_distance,
        n_jobs: int | None = None,
    ) -> None:
        pass

    # Not useless, it restricts parameters
    def __init__(  # noqa: WPS612
        self,
        *,
        mode: Literal["connectivity", "distance"] = "distance",
        n_neighbors: int = 5,
        algorithm: AlgorithmType = 'auto',
        leaf_size: int = 30,
        metric: Literal["precomputed"] | Metric[Input] = l2_distance,
        n_jobs: int | None = None,
    ) -> None:
        self.mode = mode
        super().__init__(
            n_neighbors=n_neighbors,
            algorithm=algorithm,
            leaf_size=leaf_size,
            metric=metric,
            n_jobs=n_jobs,
        )

    def _init_estimator(self) -> _KNeighborsTransformer:

        return _KNeighborsTransformer(
            mode=self.mode,
            n_neighbors=self.n_neighbors,
            algorithm=self.algorithm,
            leaf_size=self.leaf_size,
            metric="precomputed",
            n_jobs=self.n_jobs,
        )

    def _fit(
        self: SelfType,
        X: Input,
        y: Target,
        fit_with_zeros: bool = True,
    ) -> SelfType:
        ret = super()._fit(X, y)

        self.n_features_in_ = 1 if isinstance(X, FData) else X.shape[1]

        return ret

    def transform(
        self,
        X: Input,
    ) -> csr_matrix:
        self._check_is_fitted()
        add_one = self.mode == "distance"
        return self.kneighbors_graph(
            X,
            mode=self.mode,
            n_neighbors=self.n_neighbors + add_one,
        )
