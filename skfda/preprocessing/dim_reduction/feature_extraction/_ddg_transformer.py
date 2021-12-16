"""Feature extraction transformers for dimensionality reduction."""
from __future__ import annotations

from typing import Generic, Sequence, TypeVar, Union

import numpy as np
from numpy import ndarray
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted as sklearn_check_is_fitted

from ...._utils import _classifier_fit_depth_methods
from ....exploratory.depth import Depth, ModifiedBandDepth
from ....representation.grid import FData

T = TypeVar("T", bound=FData)


class DDGTransformer(
    BaseEstimator,  # type: ignore
    TransformerMixin,  # type: ignore
    Generic[T],
):
    r"""Generalized depth-versus-depth (DD) transformer for functional data.

    This transformer takes a list of k depths and performs the following map:

    .. math::
        \mathcal{X} &\rightarrow \mathbb{R}^G \\
        x &\rightarrow \textbf{d} = (D_1^1(x), D_1^2(x),...,D_g^k(x))

    Where :math:`D_i^j(x)` is the depth of the point :math:`x` with respect to
    the data in the :math:`i`-th group using the :math:`j`-th depth of the
    provided list.

    Note that :math:`\mathcal{X}` is possibly multivariate, that is,
    :math:`\mathcal{X} = \mathcal{X}_1 \times ... \times \mathcal{X}_p`.

    Parameters:
        depth_method:
            The depth class or sequence of depths to use when calculating
            the depth of a test sample in a class. See the documentation of
            the depths module for a list of available depths. By default it
            is ModifiedBandDepth.

    Examples:
        Firstly, we will import and split the Berkeley Growth Study dataset

        >>> from skfda.datasets import fetch_growth
        >>> from sklearn.model_selection import train_test_split
        >>> dataset = fetch_growth()
        >>> fd = dataset['data']
        >>> y = dataset['target']
        >>> X_train, X_test, y_train, y_test = train_test_split(
        ...     fd, y, test_size=0.25, stratify=y, random_state=0)

        >>> from skfda.preprocessing.dim_reduction.feature_extraction import \
        ... DDGTransformer
        >>> from sklearn.pipeline import make_pipeline
        >>> from sklearn.neighbors import KNeighborsClassifier

        We classify by first transforming our data using the defined map
        and then using KNN

        >>> pipe = make_pipeline(DDGTransformer(), KNeighborsClassifier())
        >>> pipe.fit(X_train, y_train)
        Pipeline(steps=[('ddgtransformer',
                         DDGTransformer(depth_method=[ModifiedBandDepth()])),
                        ('kneighborsclassifier', KNeighborsClassifier())])

        We can predict the class of new samples

        >>> pipe.predict(X_test)
        array([1, 1, 1, 0, 1, 1, 1, 1, 0, 0, 0, 1,
               1, 0, 0, 0, 0, 1, 1, 1, 1, 0, 1, 1])

        Finally, we calculate the mean accuracy for the test data

        >>> pipe.score(X_test, y_test)
        0.875

    References:
        Cuesta-Albertos, J. A., Febrero-Bande, M. and Oviedo de la Fuente, M.
        (2017). The DDG-classifier in the functional setting.
        TEST, 26. 119-142.
    """

    def __init__(  # noqa: WPS234
        self,
        depth_method: Union[Depth[T], Sequence[Depth[T]], None] = None,
    ) -> None:
        self.depth_method = depth_method

    def fit(self, X: T, y: ndarray) -> DDGTransformer[T]:
        """Fit the model using X as training data and y as target values.

        Args:
            X: FDataGrid with the training data.
            y: Target values of shape = (n_samples).

        Returns:
            self
        """
        if self.depth_method is None:
            self.depth_method = ModifiedBandDepth()

        if isinstance(self.depth_method, Depth):
            self.depth_method = [self.depth_method]

        classes, class_depth_methods = _classifier_fit_depth_methods(
            X, y, self.depth_method,
        )

        self._classes = classes
        self.class_depth_methods_ = class_depth_methods

        return self

    def transform(self, X: T) -> ndarray:
        """Transform the provided data using the defined map.

        Args:
            X: FDataGrid with the test samples.

        Returns:
            Array of shape (n_samples, G).
        """
        sklearn_check_is_fitted(self)

        return np.transpose([
            depth_method.transform(X)
            for depth_method in self.class_depth_methods_
        ])
