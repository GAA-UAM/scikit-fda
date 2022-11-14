"""Depth-based models for supervised classification."""
from __future__ import annotations

from collections import defaultdict
from contextlib import suppress
from itertools import combinations
from typing import (
    DefaultDict,
    Dict,
    Mapping,
    Optional,
    Sequence,
    Tuple,
    TypeVar,
    Union,
)

import numpy as np
import pandas as pd
from scipy.interpolate import lagrange
from sklearn.base import clone
from sklearn.metrics import accuracy_score
from sklearn.pipeline import FeatureUnion, make_pipeline
from sklearn.utils.validation import check_is_fitted as sklearn_check_is_fitted

from ..._utils import _classifier_get_classes
from ..._utils._sklearn_adapter import BaseEstimator, ClassifierMixin
from ...exploratory.depth import Depth, ModifiedBandDepth
from ...preprocessing.feature_construction._per_class_transformer import (
    PerClassTransformer,
)
from ...representation import FData
from ...typing._numpy import NDArrayFloat, NDArrayInt, NDArrayStr

Input = TypeVar("Input", bound=FData)
Target = TypeVar("Target", bound=Union[NDArrayInt, NDArrayStr])


def _classifier_get_depth_methods(
    classes: NDArrayInt | NDArrayStr,
    X: Input,
    y_ind: NDArrayInt,
    depth_methods: Sequence[Depth[Input]],
) -> Sequence[Depth[Input]]:
    return [
        clone(depth_method).fit(X[y_ind == cur_class])
        for cur_class in range(len(classes))
        for depth_method in depth_methods
    ]


def _classifier_fit_depth_methods(
    X: Input,
    y: NDArrayInt | NDArrayStr,
    depth_methods: Sequence[Depth[Input]],
) -> Tuple[NDArrayStr | NDArrayInt, Sequence[Depth[Input]]]:
    classes, y_ind = _classifier_get_classes(y)

    class_depth_methods_ = _classifier_get_depth_methods(
        classes, X, y_ind, depth_methods,
    )

    return classes, class_depth_methods_


class DDClassifier(
    BaseEstimator,
    ClassifierMixin[Input, Target],
):
    """Depth-versus-depth (DD) classifer for functional data.

    Transforms the data into a DD-plot and then classifies using a polynomial
    of a chosen degree. The polynomial passes through zero and maximizes the
    accuracy of the classification on the train dataset.

    If a point is below the polynomial in the DD-plot, it is classified to
    the first class. Otherwise, the point is classified to the second class.

    Parameters:
        degree: degree of the polynomial used to classify in the DD-plot
        depth_method:
            The depth class to use when calculating the depth of a test
            sample in a class. See the documentation of the depths module
            for a list of available depths. By default it is ModifiedBandDepth.

    Examples:
        Firstly, we will import and split the Berkeley Growth Study dataset

        >>> from skfda.datasets import fetch_growth
        >>> from sklearn.model_selection import train_test_split
        >>> dataset = fetch_growth()
        >>> fd = dataset['data']
        >>> y = dataset['target']
        >>> X_train, X_test, y_train, y_test = train_test_split(
        ...     fd, y, test_size=0.25, stratify=y, random_state=0)

        We will fit a DD-classifier

        >>> from skfda.ml.classification import DDClassifier
        >>> clf = DDClassifier(degree=2)
        >>> clf.fit(X_train, y_train)
        DDClassifier(...)

        We can predict the class of new samples

        >>> clf.predict(X_test)
        array([1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 1,
               1, 0, 0, 0, 0, 1, 1, 1, 1, 0, 1, 1])

        Finally, we calculate the mean accuracy for the test data

        >>> clf.score(X_test, y_test)
        0.875

    See also:
        :class:`~skfda.ml.classification.DDGClassifier`
        :class:`~skfda.ml.classification.MaximumDepthClassifier`
        :class:`~skfda.preprocessing.dim_reduction.feature_extraction._ddg_transformer`

    References:
        Li, J., Cuesta-Albertos, J. A., and Liu, R. Y. (2012). DD-classifier:
        Nonparametric classification procedure based on DD-plot. Journal of
        the American Statistical Association, 107(498):737-753.
    """

    def __init__(
        self,
        degree: int,
        depth_method: Optional[Depth[Input]] = None,
    ) -> None:
        self.depth_method = depth_method
        self.degree = degree

    def fit(self, X: Input, y: Target) -> DDClassifier[Input, Target]:
        """Fit the model using X as training data and y as target values.

        Args:
            X: FDataGrid with the training data.
            y: Target values of shape = (n_samples).

        Returns:
            self
        """
        if self.depth_method is None:
            self.depth_method = ModifiedBandDepth()

        classes, class_depth_methods = _classifier_fit_depth_methods(
            X, y, [self.depth_method],
        )

        self.classes_ = classes
        self.class_depth_methods_ = class_depth_methods

        if (len(self.classes_) != 2):
            raise ValueError("DDClassifier only accepts two classes.")

        dd_coordinates = [
            depth_method.transform(X)
            for depth_method in self.class_depth_methods_
        ]

        polynomial_elements = combinations(
            range(len(dd_coordinates[0])),  # noqa: WPS518
            self.degree,
        )

        accuracy = -1  # initialise accuracy

        for elements in polynomial_elements:
            x_coord = np.append(dd_coordinates[0][list(elements)], 0)
            y_coord = np.append(dd_coordinates[1][list(elements)], 0)

            poly = lagrange(
                x_coord, y_coord,
            )

            predicted_values = np.polyval(poly, dd_coordinates[0])

            y_pred = self.classes_[(
                dd_coordinates[1] > predicted_values
            ).astype(int)
            ]

            new_accuracy = accuracy_score(y, y_pred)

            if (new_accuracy > accuracy):
                accuracy = new_accuracy
                self.poly_ = poly

        return self

    def predict(self, X: Input) -> Target:
        """Predict the class labels for the provided data.

        Args:
            X: FDataGrid with the test samples.

        Returns:
            Array of shape (n_samples) with class labels
                for each data sample.
        """
        sklearn_check_is_fitted(self)

        dd_coordinates = [
            depth_method.transform(X)
            for depth_method in self.class_depth_methods_
        ]

        predicted_values = np.polyval(self.poly_, dd_coordinates[0])

        return self.classes_[(  # type: ignore[no-any-return]
            dd_coordinates[1] > predicted_values
        ).astype(int)
        ]


class DDGClassifier(
    BaseEstimator,
    ClassifierMixin[Input, Target],
):
    r"""Generalized depth-versus-depth (DD) classifer for functional data.

    This classifier builds an interface around the DDGTransfomer.

    The transformer takes a list of k depths and performs the following map:

    .. math::
        \mathcal{X} &\rightarrow \mathbb{R}^G \\
        x &\rightarrow \textbf{d} = (D_1^1(x), D_1^2(x),...,D_g^k(x))

    Where :math:`D_i^j(x)` is the depth of the point :math:`x` with respect to
    the data in the :math:`i`-th group using the :math:`j`-th depth of the
    provided list.

    Note that :math:`\mathcal{X}` is possibly multivariate, that is,
    :math:`\mathcal{X} = \mathcal{X}_1 \times ... \times \mathcal{X}_p`.

    In the G dimensional space the classification is performed using a
    multivariate classifer.

    Parameters:
        depth_method:
            The depth class or sequence of depths to use when calculating
            the depth of a test sample in a class. See the documentation of
            the depths module for a list of available depths. By default it
            is ModifiedBandDepth.
        multivariate_classifier:
            The multivariate classifier to use in the DDG-plot.

    Examples:
        Firstly, we will import and split the Berkeley Growth Study dataset

        >>> from skfda.datasets import fetch_growth
        >>> from sklearn.model_selection import train_test_split

        >>> dataset = fetch_growth()
        >>> fd = dataset['data']
        >>> y = dataset['target']
        >>> X_train, X_test, y_train, y_test = train_test_split(
        ...     fd, y, test_size=0.25, stratify=y, random_state=0)

        >>> from skfda.exploratory.depth import (
        ...     ModifiedBandDepth,
        ...     IntegratedDepth,
        ... )
        >>> from sklearn.neighbors import KNeighborsClassifier

        We will fit a DDG-classifier using KNN

        >>> from skfda.ml.classification import DDGClassifier
        >>> clf = DDGClassifier(
        ...     depth_method=ModifiedBandDepth(),
        ...     multivariate_classifier=KNeighborsClassifier(),
        ... )
        >>> clf.fit(X_train, y_train)
        DDGClassifier(...)

        We can predict the class of new samples

        >>> clf.predict(X_test)
        array([1, 1, 1, 0, 1, 1, 1, 1, 0, 0, 0, 1,
               1, 0, 0, 0, 0, 1, 1, 1, 1, 0, 1, 1])

        Finally, we calculate the mean accuracy for the test data

        >>> clf.score(X_test, y_test)
        0.875

        It is also possible to use several depth functions to increase the
        number of features available to the classifier

        >>> clf = DDGClassifier(
        ...     depth_method=[
        ...         ("mbd", ModifiedBandDepth()),
        ...         ("id", IntegratedDepth()),
        ...     ],
        ...     multivariate_classifier=KNeighborsClassifier(),
        ... )
        >>> clf.fit(X_train, y_train)
        DDGClassifier(...)
        >>> clf.predict(X_test)
        array([1, 1, 1, 0, 1, 1, 1, 1, 0, 0, 0, 1,
               1, 0, 0, 0, 0, 1, 1, 1, 1, 0, 1, 1])
        >>> clf.score(X_test, y_test)
        0.875

    See also:
        :class:`~skfda.ml.classification.DDClassifier`
        :class:`~skfda.ml.classification.MaximumDepthClassifier`
        :class:`~skfda.preprocessing.dim_reduction.feature_extraction._ddg_transformer`

    References:
        Li, J., Cuesta-Albertos, J. A., and Liu, R. Y. (2012). DD-classifier:
        Nonparametric classification procedure based on DD-plot. Journal of
        the American Statistical Association, 107(498):737-753.

        Cuesta-Albertos, J.A., Febrero-Bande, M. and Oviedo de la Fuente, M.
        (2017) The DDG-classifier in the functional setting. TEST, 26. 119-142.
    """

    def __init__(  # noqa: WPS234
        self,
        *,
        multivariate_classifier: ClassifierMixin[
            NDArrayFloat,
            NDArrayInt,
        ] | None = None,
        depth_method: Depth[Input] | Sequence[
            Tuple[str, Depth[Input]]
        ] | None = None,
    ) -> None:
        self.multivariate_classifier = multivariate_classifier
        self.depth_method = depth_method

    def get_params(self, deep: bool = True) -> Mapping[str, object]:
        params = BaseEstimator.get_params(self, deep=deep)
        if deep and isinstance(self.depth_method, Sequence):
            for name, depth in self.depth_method:
                depth_params = depth.get_params(deep=deep)

                for key, value in depth_params.items():
                    params[f"depth_method__{name}__{key}"] = value

        return params  # type: ignore[no-any-return]

    # Copied from scikit-learn's _BaseComposition with minor modifications
    def _set_params(
        self,
        attr: str,
        **params: object,
    ) -> DDGClassifier[Input, Target]:
        # Ensure strict ordering of parameter setting:
        # 1. All steps
        if attr in params:
            setattr(self, attr, params.pop(attr))
        # 2. Replace items with estimators in params
        items = getattr(self, attr)
        if isinstance(items, list) and items:
            # Get item names used to identify valid names in params
            # `zip` raises a TypeError when `items` does not contains
            # elements of length 2
            with suppress(TypeError):
                item_names, _ = zip(*items)
                item_params: DefaultDict[
                    str,
                    Dict[str, object],
                ] = defaultdict(dict)
                for name in list(params.keys()):
                    if name.startswith(f"{attr}__"):
                        key, delim, sub_key = name.partition("__")
                        if "__" not in sub_key and sub_key in item_names:
                            self._replace_estimator(
                                attr,
                                sub_key,
                                params.pop(name),
                            )
                        else:
                            key, delim, sub_key = sub_key.partition("__")
                            item_params[key][sub_key] = params.pop(name)

                for name, estimator in items:
                    estimator.set_params(**item_params[name])

        # 3. Step parameters and other initialisation arguments
        super().set_params(**params)
        return self

    # Copied from scikit-learn's _BaseComposition
    def _replace_estimator(
        self,
        attr: str,
        name: str,
        new_val: object,
    ) -> None:
        # assumes `name` is a valid estimator name
        new_estimators = list(getattr(self, attr))
        for i, (estimator_name, _) in enumerate(new_estimators):
            if estimator_name == name:
                new_estimators[i] = (name, new_val)
                break
        setattr(self, attr, new_estimators)

    def set_params(
        self,
        **params: object,
    ) -> DDGClassifier[Input, Target]:

        return self._set_params("depth_method", **params)

    def fit(self, X: Input, y: Target) -> DDGClassifier[Input, Target]:
        """Fit the model using X as training data and y as target values.

        Args:
            X: FDataGrid with the training data.
            y: Target values of shape = (n_samples).

        Returns:
            self
        """
        depth_method = (
            ModifiedBandDepth()
            if self.depth_method is None
            else self.depth_method
        )

        if isinstance(depth_method, Sequence):
            transformer = FeatureUnion([
                (name, PerClassTransformer(depth))
                for name, depth in depth_method
            ])
        else:
            transformer = PerClassTransformer(depth_method)

        self._pipeline = make_pipeline(
            transformer,
            clone(self.multivariate_classifier),
        )

        self._pipeline.fit(X, y)
        self.classes_ = _classifier_get_classes(y)[0]

        return self

    def predict(self, X: Input) -> Target:
        """Predict the class labels for the provided data.

        Args:
            X: FDataGrid with the test samples.

        Returns:
            Array of shape (n_samples) with class labels
                for each data sample.
        """
        return self._pipeline.predict(X)  # type: ignore[no-any-return]


class _ArgMaxClassifier(
    BaseEstimator,
    ClassifierMixin[NDArrayFloat, Target],
):
    r"""Arg max classifier for multivariate data.

    Test samples are classified to the class that corresponds to the
    index of the highest coordinate.

    Examples:
        >>> import numpy as np
        >>> X = np.array([[1,5], [3,2], [4,1]])
        >>> y = np.array([1, 0, 0])

        We will fit an ArgMax classifier

        >>> from skfda.ml.classification._depth_classifiers import \
        ... _ArgMaxClassifier
        >>> clf = _ArgMaxClassifier()
        >>> clf.fit(X, y)
        _ArgMaxClassifier(...)

        We can predict the class of new samples

        >>> clf.predict(X) # Predict labels for test samples
        array([1, 0, 0])
    """

    def fit(self, X: NDArrayFloat, y: Target) -> _ArgMaxClassifier[Target]:
        """Fit the model using X as training data and y as target values.

        Args:
            X: Array with the training data.
            y: Target values of shape = (n_samples).

        Returns:
            self
        """
        classes, _ = _classifier_get_classes(y)
        self.classes_ = classes
        return self

    def predict(self, X: Union[NDArrayFloat, pd.DataFrame]) -> Target:
        """Predict the class labels for the provided data.

        Args:
            X: Array with the test samples or a pandas DataFrame.

        Returns:
            Array of shape (n_samples) with class labels
                for each data sample.
        """
        if isinstance(X, pd.DataFrame):
            X = X.to_numpy()
        return self.classes_[  # type: ignore[no-any-return]
            np.argmax(X, axis=1)
        ]


class MaximumDepthClassifier(DDGClassifier[Input, Target]):
    """Maximum depth classifier for functional data.

    Test samples are classified to the class where they are deeper.

    Parameters:
        depth_method:
            The depth class to use when calculating the depth of a test
            sample in a class. See the documentation of the depths module
            for a list of available depths. By default it is ModifiedBandDepth.
    Examples:
        Firstly, we will import and split the Berkeley Growth Study dataset

        >>> from skfda.datasets import fetch_growth
        >>> from sklearn.model_selection import train_test_split
        >>> dataset = fetch_growth()
        >>> fd = dataset['data']
        >>> y = dataset['target']
        >>> X_train, X_test, y_train, y_test = train_test_split(
        ...     fd, y, test_size=0.25, stratify=y, random_state=0)

        We will fit a Maximum depth classifier

        >>> from skfda.ml.classification import MaximumDepthClassifier
        >>> clf = MaximumDepthClassifier()
        >>> clf.fit(X_train, y_train)
        MaximumDepthClassifier(...)

        We can predict the class of new samples

        >>> clf.predict(X_test) # Predict labels for test samples
        array([1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 1,
               1, 0, 0, 0, 0, 1, 1, 1, 1, 0, 1, 1])

        Finally, we calculate the mean accuracy for the test data

        >>> clf.score(X_test, y_test)
        0.875

    See also:
        :class:`~skfda.ml.classification.DDClassifier`
        :class:`~skfda.ml.classification.DDGClassifier`

    References:
        Ghosh, A. K. and Chaudhuri, P. (2005b). On maximum depth and
        related classifiers. Scandinavian Journal of Statistics, 32, 327–350.
    """

    def __init__(self, depth_method: Depth[Input] | None = None) -> None:
        super().__init__(
            multivariate_classifier=_ArgMaxClassifier(),
            depth_method=depth_method,
        )
