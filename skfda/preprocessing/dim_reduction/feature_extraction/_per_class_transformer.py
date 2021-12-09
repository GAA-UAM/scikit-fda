"""Feature extraction transformers for dimensionality reduction."""
from __future__ import annotations

import warnings
from typing import TypeVar, Union

import numpy as np
from pandas import DataFrame
from sklearn.utils.validation import check_is_fitted as sklearn_check_is_fitted

from ...._utils import TransformerMixin, _fit_feature_transformer
from ....representation._typing import NDArrayInt
from ....representation.basis import FDataBasis
from ....representation.grid import FData, FDataGrid

T = TypeVar("T", bound=FData)
Input = TypeVar("Input")
Output = TypeVar("Output")
Target = TypeVar("Target", bound=NDArrayInt)


class PerClassTransformer(TransformerMixin[Input, Output, Target]):
    r"""Per class feature transformer for functional data.

    This class takes a transformer and performs the following map:

    .. math::
        \mathcal{X} &\rightarrow \mathbb{R}^G \\
        x &\rightarrow \textbf{t} = (T_1(x), T_2(x),...,T_k(x))

    Where :math:`T_i(x)` is the transformation  :math:`x` with respect to
    the data in the :math:`i`-th group.

    Note that :math:`\mathcal{X}` is possibly multivariate, that is,
    :math:`\mathcal{X} = \mathcal{X}_1 \times ... \times \mathcal{X}_p`.

    Parameters:
        transformer:
            The transformer that we want to apply to the given data.
            It should use target data while fitting.
            This is checked by looking at the 'stateless' and 'requires_y' tags
        array_output:
            Indicates if the transformed data is requested to be a NumPy array
            output. By default the value is False.
    Examples:
        Firstly, we will import the Berkeley Growth Study dataset:

        >>> from skfda.datasets import fetch_growth
        >>> X, y = fetch_growth(return_X_y=True, as_frame=True)
        >>> X = X.iloc[:, 0].values
        >>> y = y.values.codes

        >>> from skfda.preprocessing.dim_reduction.feature_extraction import (
        ...     PerClassTransformer,
        ... )

        Then we will need to select a fda transformer, and so we will
        use RecursiveMaximaHunting. We need to fit the data and transform it:

        >>> from skfda.preprocessing.dim_reduction.variable_selection import (
        ...     RecursiveMaximaHunting,
        ... )
        >>> t = PerClassTransformer(
        ...     RecursiveMaximaHunting(),
        ...     array_output=True,
        ... )
        >>> x_transformed = t.fit_transform(X, y)

        ``x_transformed`` will be a vector with the transformed data.
        We will split the generated data and fit a KNN classifier.

        >>> from sklearn.model_selection import train_test_split
        >>> from sklearn.neighbors import KNeighborsClassifier
        >>> X_train, X_test, y_train, y_test = train_test_split(
        ...     x_transformed,
        ...     y,
        ...     test_size=0.25,
        ...     stratify=y,
        ...     random_state=0,
        ... )
        >>> neigh = KNeighborsClassifier()
        >>> neigh = neigh.fit(X_train, y_train)

        Finally we can predict and check the score:
        >>> neigh.predict(X_test)
            array([0, 0, 1, 0, 1, 1, 1, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0,
            1, 1, 1, 1, 1, 1, 1], dtype=int8)

        >>> round(neigh.score(X_test, y_test), 3)
            0.958
    """

    def __init__(
        self,
        transformer: TransformerMixin[Input, Output, Target],
        *,
        array_output: bool = False,
    ) -> None:
        self.transformer = transformer
        self.array_output = array_output

    def _validate_transformer(
        self,
    ) -> None:
        """
        Check that the transformer passed is valid.

        Check that it is scikit-learn-like and that
        uses target data in fit.

        Args:
            None

        Returns:
            None
        """
        if not (
            hasattr(self.transformer, "fit")  # noqa: WPS421
            and hasattr(self.transformer, "transform")  # noqa: WPS421
            and hasattr(self.transformer, "fit_transform")  # noqa: WPS421
        ):
            raise TypeError(
                "Transformer should implement fit and "
                "transform. " + str(self.transformer)
                + " (type " + str(type(self.transformer)) + ")"
                " doesn't",
            )

        tags = self.transformer._get_tags()  # noqa: WPS437

        if tags['stateless'] or not tags['requires_y']:
            warnings.warn(
                f"Parameter ``transformer`` with type"  # noqa: WPS237
                f" {type(self.transformer)} should use class information."
                f" It should have the ``requires_y`` tag set to ``True`` and"
                f" the ``stateless`` tag set to ``False``",
            )

    def fit(
        self,
        X: T,
        y: np.ndarray,
    ) -> PerClassTransformer:
        """
        Fit the model on each class.

        It uses X as training data and y as target values.

        Args:
            X: FDataGrid with the training data.
            y: Target values of shape = (n_samples).

        Returns:
            self
        """
        self._validate_transformer()
        classes, class_feature_transformers = _fit_feature_transformer(
            X,
            y,
            self.transformer,
        )

        self._classes = classes
        self._class_feature_transformers_ = class_feature_transformers

        return self

    def transform(self, X: T) -> Union[DataFrame, np.ndarray]:
        """
        Transform the provided data using the already fitted transformer.

        Args:
            X: FDataGrid with the test samples.

        Returns:
            Eiter array of shape (n_samples, G) or a Data Frame
            including the transformed data.
        """
        sklearn_check_is_fitted(self)

        transformed_data = np.empty((len(X), 0))
        for feature_transformer in self._class_feature_transformers_:
            elem = feature_transformer.transform(X)
            data = np.array(elem)
            transformed_data = np.hstack((transformed_data, data))

        if self.array_output:
            for i in transformed_data:
                if isinstance(i, (FDataGrid, FDataBasis)):
                    raise TypeError(
                        "There are transformed instances of FDataGrid or "
                        "FDataBasis that can't be concatenated on a NumPy "
                        "array.",
                    )
            return np.array(transformed_data)

        return DataFrame(
            {'Transformed data': transformed_data},
        )

    def fit_transform(
        self,
        X: T,
        y: np.ndarray,
    ) -> Union[DataFrame, np.ndarray]:
        """
        Fits and transforms the provided data.

        It uses the transformer specified when initializing the class.

        Args:
            X: FDataGrid with the samples.
            y: Target values of shape = (n_samples)

        Returns:
            Eiter array of shape (n_samples, G) or a Data Frame \
            including the transformed data.
        """
        return self.fit(X, y).transform(X)
