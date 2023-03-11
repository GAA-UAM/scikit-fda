from __future__ import annotations

import operator
from dataclasses import dataclass
from typing import (
    Any,
    Callable,
    Dict,
    Generic,
    Tuple,
    TypeVar,
    Union,
    overload,
)

import numpy as np
import sklearn.utils.validation
from sklearn.feature_selection import (
    mutual_info_classif,
    mutual_info_regression,
)
from typing_extensions import Final, Literal

from ...._utils._sklearn_adapter import (
    BaseEstimator,
    InductiveTransformerMixin,
)
from ....representation.grid import FDataGrid
from ....typing._base import RandomStateLike
from ....typing._numpy import NDArrayFloat, NDArrayInt, NDArrayReal
from ._base import _compute_dependence, _DependenceMeasure

_Criterion = Callable[[NDArrayFloat, NDArrayFloat], NDArrayFloat]
_CriterionLike = Union[
    _Criterion,
    Literal["difference", "quotient"],
]

SelfType = TypeVar(
    "SelfType",
    bound="MinimumRedundancyMaximumRelevance[Any]",
)

dtype_X_T = TypeVar("dtype_X_T", bound=np.float_, covariant=True)
dtype_y_T = TypeVar(
    "dtype_y_T",
    bound=Union[np.int_, np.float_],
    covariant=True,
)


@dataclass
class Method(Generic[dtype_y_T]):
    """Predefined mRMR method."""

    relevance_dependence_measure: _DependenceMeasure[
        NDArrayFloat,
        np.typing.NDArray[dtype_y_T],
    ]
    redundancy_dependence_measure: _DependenceMeasure[
        NDArrayFloat,
        NDArrayFloat,
    ]
    criterion: _Criterion


def mutual_information(
    x: NDArrayFloat,
    y: NDArrayReal,
    *,
    n_neighbors: int | None = None,
    random_state: RandomStateLike = None,
) -> NDArrayFloat:
    """Compute mutual information."""
    y = y.ravel()

    method = (
        mutual_info_regression if issubclass(y.dtype.type, np.floating)
        else mutual_info_classif
    )

    extra_args: Dict[str, Any] = {}
    if n_neighbors is not None:
        extra_args['n_neighbors'] = n_neighbors

    return method(  # type: ignore[no-any-return]
        x,
        y,
        random_state=random_state,
        **extra_args,
    )


MID: Final = Method(
    relevance_dependence_measure=mutual_information,
    redundancy_dependence_measure=mutual_information,
    criterion=operator.sub,
)


MIQ: Final = Method(
    relevance_dependence_measure=mutual_information,
    redundancy_dependence_measure=mutual_information,
    criterion=operator.truediv,
)


MethodName = Literal["MID", "MIQ"]


def _parse_method(name: MethodName) -> Method[Union[np.int_, np.float_]]:
    if name == "MID":
        return MID
    elif name == "MIQ":
        return MIQ


def _mrmr(
    X: np.typing.NDArray[dtype_X_T],
    Y: np.typing.NDArray[dtype_y_T],
    n_features_to_select: int = 1,
    relevance_dependence_measure: _DependenceMeasure[
        np.typing.NDArray[dtype_X_T],
        np.typing.NDArray[dtype_y_T],
    ] = mutual_information,
    redundancy_dependence_measure: _DependenceMeasure[
        np.typing.NDArray[dtype_X_T],
        np.typing.NDArray[dtype_X_T],
    ] = mutual_information,
    criterion: _Criterion = operator.truediv,
) -> Tuple[NDArrayInt, NDArrayFloat, NDArrayFloat]:
    indexes = list(range(X.shape[1]))

    selected_features = []
    scores = []
    selected_relevances = []

    relevances = _compute_dependence(
        X[..., np.newaxis],
        Y,
        dependence_measure=relevance_dependence_measure,
    )
    redundancies = np.zeros((X.shape[1], X.shape[1]))

    max_index = int(np.argmax(relevances))
    selected_features.append(indexes[max_index])
    scores.append(relevances[max_index])
    selected_relevances.append(relevances[max_index])

    indexes.remove(max_index)

    # TODO: Vectorize
    for i in range(1, n_features_to_select):

        # Calculate redundancies of the last selected variable
        last_selected = selected_features[i - 1]

        for j in range(X.shape[1]):
            if not redundancies[last_selected, j]:
                redundancies[last_selected, j] = redundancy_dependence_measure(
                    X[:, last_selected, np.newaxis],
                    X[:, j, np.newaxis],
                )
                redundancies[j, last_selected] = redundancies[last_selected, j]

        W = np.mean(
            redundancies[np.ix_(selected_features[:i], indexes)],
            axis=0,
        )

        coef = criterion(relevances[indexes], W)

        max_index = int(np.argmax(coef))
        selected_features.append(indexes[max_index])
        scores.append(coef[max_index])
        selected_relevances.append(relevances[max_index])

        indexes.remove(indexes[max_index])

    return (
        np.asarray(selected_features),
        np.asarray(scores),
        np.asarray(relevances),
    )


class MinimumRedundancyMaximumRelevance(
    BaseEstimator,
    InductiveTransformerMixin[
        FDataGrid,
        NDArrayFloat,
        Union[NDArrayInt, NDArrayFloat],
    ],
    Generic[dtype_y_T],
):
    r"""
    Minimum redundancy maximum relevance (mRMR) method.

    This is a greedy version of mRMR that selects the variables iteratively.
    This method considers the relevance of a variable as well as its redundancy
    with respect of the already selected ones.    

    It uses a dependence measure between random variables to compute the
    dependence between the candidate variable and the target (for the
    relevance) and another to compute the dependence between two variables
    (for the redundancy).
    It combines both measurements using a criterion such as the difference or
    the quotient, and then selects the variable that maximizes that quantity.
    For example, using the quotient criterion and the same dependence function
    :math:`D` for relevance and redundancy, the variable selected at the
    :math:`i`-th step would be :math:`X(t_i)` with

    .. math::
        t_i = \underset {t}{\operatorname {arg\,max}} \frac{D(X(t), y)}
        {\frac{1}{i-1}\sum_{j < i} D(X(t), X(t_j))}.

    For further discussion of the applicability of this method to functional
    data see :footcite:`berrendero++_2016_mrmr`.

    Parameters:
        n_features_to_select: Number of features to select.
        method: Predefined method to use (MID or MIQ).
        dependence_measure: Dependence measure to use both for relevance and
            for redundancy.
        relevance_dependence_measure: Dependence measure used to compute
            relevance.
        redundancy_dependence_measure: Dependence measure used to compute
            redundancy.
        criterion: Criterion to combine relevance and redundancy. It must be
            a Python callable with two inputs. As common choices include the
            difference and the quotient, both can be especified as strings.

    Examples:
        >>> from skfda.preprocessing.dim_reduction import variable_selection
        >>> from skfda.datasets import make_gaussian_process
        >>> import skfda
        >>> import numpy as np
        >>> import dcor

        We create trajectories from two classes, one with zero mean and the
        other with a peak-like mean. Both have Brownian covariance.

        >>> n_samples = 1000
        >>> n_features = 100
        >>>
        >>> def mean_1(t):
        ...     return (
        ...         np.abs(t - 0.25)
        ...         - 2 * np.abs(t - 0.5)
        ...         + np.abs(t - 0.75)
        ...     )
        >>>
        >>> X_0 = make_gaussian_process(
        ...     n_samples=n_samples // 2,
        ...     n_features=n_features,
        ...     random_state=0,
        ... )
        >>> X_1 = make_gaussian_process(
        ...     n_samples=n_samples // 2,
        ...     n_features=n_features,
        ...     mean=mean_1,
        ...     random_state=1,
        ... )
        >>> X = skfda.concatenate((X_0, X_1))
        >>>
        >>> y = np.zeros(n_samples, dtype=np.int_)
        >>> y [n_samples // 2:] = 1

        Select the relevant points to distinguish the two classes. You
        may specify a method such as MIQ (the default) or MID.

        >>> mrmr = variable_selection.MinimumRedundancyMaximumRelevance(
        ...     n_features_to_select=3,
        ...     method="MID",
        ... )
        >>> _ = mrmr.fit(X, y)
        >>> point_mask = mrmr.get_support()
        >>> points = X.grid_points[0][point_mask]

        Apply the learned dimensionality reduction

        >>> X_dimred = mrmr.transform(X)
        >>> len(X.grid_points[0])
        100
        >>> X_dimred.shape
        (1000, 3)

        It is also possible to specify the measure of dependence used (or
        even different ones for relevance and redundancy) as well as the
        function to combine relevance and redundancy (usually the division
        or subtraction operations).

        >>> mrmr = variable_selection.MinimumRedundancyMaximumRelevance(
        ...     n_features_to_select=3,
        ...     dependence_measure=dcor.u_distance_correlation_sqr,
        ...     criterion="quotient",
        ... )
        >>> _ = mrmr.fit(X, y)

        As a toy example illustrating the customizability of this method,
        consider the following:

        >>> mrmr = variable_selection.MinimumRedundancyMaximumRelevance(
        ...     n_features_to_select=3,
        ...     relevance_dependence_measure=dcor.u_distance_covariance_sqr,
        ...     redundancy_dependence_measure=dcor.u_distance_correlation_sqr,
        ...     criterion=lambda rel, red: 0.5 * rel / red,
        ... )
        >>> _ = mrmr.fit(X, y)

    References:
        .. footbibliography::

    """

    @overload
    def __init__(
        self,
        *,
        n_features_to_select: int = 1,
    ) -> None:
        pass

    @overload
    def __init__(
        self,
        *,
        n_features_to_select: int = 1,
        method: Method[dtype_y_T] | MethodName,
    ) -> None:
        pass

    @overload
    def __init__(
        self,
        *,
        n_features_to_select: int = 1,
        dependence_measure: _DependenceMeasure[
            np.typing.NDArray[np.float_],
            np.typing.NDArray[np.float_ | dtype_y_T],
        ],
        criterion: _CriterionLike,
    ) -> None:
        pass

    @overload
    def __init__(
        self,
        *,
        n_features_to_select: int = 1,
        relevance_dependence_measure: _DependenceMeasure[
            np.typing.NDArray[np.float_],
            np.typing.NDArray[dtype_y_T],
        ],
        redundancy_dependence_measure: _DependenceMeasure[
            np.typing.NDArray[np.float_],
            np.typing.NDArray[np.float_],
        ],
        criterion: _CriterionLike,
    ) -> None:
        pass

    def __init__(
        self,
        *,
        n_features_to_select: int = 1,
        method: Method[dtype_y_T] | MethodName | None = None,
        dependence_measure: _DependenceMeasure[
            np.typing.NDArray[np.float_],
            np.typing.NDArray[np.float_ | dtype_y_T],
        ] | None = None,
        relevance_dependence_measure: _DependenceMeasure[
            np.typing.NDArray[np.float_],
            np.typing.NDArray[dtype_y_T],
        ] | None = None,
        redundancy_dependence_measure: _DependenceMeasure[
            np.typing.NDArray[np.float_],
            np.typing.NDArray[np.float_],
        ] | None = None,
        criterion: _CriterionLike | None = None,
    ) -> None:
        self.n_features_to_select = n_features_to_select
        self.method = method
        self.dependence_measure = dependence_measure
        self.relevance_dependence_measure = relevance_dependence_measure
        self.redundancy_dependence_measure = redundancy_dependence_measure
        self.criterion = criterion

    def _validate_parameters(self) -> None:
        method = MIQ if all(
            p is None for p in (
                self.method,
                self.dependence_measure,
                self.relevance_dependence_measure,
                self.redundancy_dependence_measure,
                self.criterion,
            )
        ) else self.method

        if method:
            if (
                self.dependence_measure
                or self.relevance_dependence_measure
                or self.redundancy_dependence_measure
                or self.criterion
            ):
                raise ValueError(
                    "The 'method' parameter and the parameters "
                    "'dependency_measure', 'relevance_dependence_measure' "
                    "'redundancy_dependence_measure' and 'criterion' are "
                    "incompatible",
                )

            method = (
                _parse_method(method)
                if isinstance(method, str) else method
            )

            self.relevance_dependence_measure_: _DependenceMeasure[
                np.typing.NDArray[np.float_],
                np.typing.NDArray[dtype_y_T],
            ] = (
                method.relevance_dependence_measure
            )
            self.redundancy_dependence_measure_ = (
                method.redundancy_dependence_measure
            )
            self.criterion_ = method.criterion

        else:
            if self.criterion is None:
                raise ValueError(
                    "You must specify a criterion parameter",
                )

            if self.criterion == "difference":
                self.criterion = operator.sub
            elif self.criterion == "quotient":
                self.criterion_ = operator.truediv
            else:
                self.criterion_ = self.criterion

            if self.dependence_measure:
                if (
                    self.relevance_dependence_measure
                    or self.redundancy_dependence_measure
                ):
                    raise ValueError(
                        "The 'dependency_measure' parameter and the "
                        "parameters 'relevance_dependence_measure' "
                        "and 'redundancy_dependence_measure' "
                        "are incompatible",
                    )

                self.relevance_dependence_measure_ = (
                    self.dependence_measure
                )
                self.redundancy_dependence_measure_ = (
                    self.dependence_measure
                )
            else:
                if not self.relevance_dependence_measure:
                    raise ValueError(
                        "Missing parameter 'relevance_dependence_measure'",
                    )
                if not self.redundancy_dependence_measure:
                    raise ValueError(
                        "Missing parameter 'redundancy_dependence_measure'",
                    )
                self.relevance_dependence_measure_ = (
                    self.relevance_dependence_measure
                )
                self.redundancy_dependence_measure_ = (
                    self.redundancy_dependence_measure
                )

    def fit(  # type: ignore[override] # noqa: D102
        self: SelfType,
        X: FDataGrid,
        y: np.typing.NDArray[dtype_y_T],
    ) -> SelfType:

        self._validate_parameters()

        X_array = X.data_matrix[..., 0]

        X_array, y = sklearn.utils.validation.check_X_y(X_array, y)

        self.features_shape_ = X_array.shape[1:]

        self.results_ = _mrmr(
            X=X_array,
            Y=y,
            n_features_to_select=self.n_features_to_select,
            relevance_dependence_measure=self.relevance_dependence_measure_,
            redundancy_dependence_measure=self.redundancy_dependence_measure_,
            criterion=self.criterion_,
        )[0]

        return self

    def transform(
        self,
        X: FDataGrid,
        y: NDArrayInt | NDArrayFloat | None = None,
    ) -> NDArrayFloat:

        X_array = X.data_matrix[..., 0]

        sklearn.utils.validation.check_is_fitted(self)

        X_array = sklearn.utils.validation.check_array(X_array)

        if X_array.shape[1:] != self.features_shape_:
            raise ValueError(
                "The trajectories have a different number of "
                "points than the ones fitted",
            )

        return X_array[:, self.results_]

    def get_support(self, indices: bool = False) -> NDArrayInt:
        indexes_unraveled = self.results_
        if indices:
            return indexes_unraveled

        mask = np.zeros(self.features_shape_[0], dtype=bool)
        mask[self.results_] = True
        return mask
