"""Feature extraction transformers for dimensionality reduction."""
from __future__ import annotations

import warnings
from typing import TypeVar, Union

from numpy import ndarray
from pandas import DataFrame
from sklearn.base import TransformerMixin
from sklearn.utils.validation import check_is_fitted as sklearn_check_is_fitted

from ...._utils import _fit_feature_transformer
from ....representation.basis import FDataBasis
from ....representation.grid import FData, FDataGrid

T = TypeVar("T", bound=FData)


class PerClassTransformer(TransformerMixin):
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
        transformer: TransformerMixin
            The transformer that we want to apply to the given data.
            It should use target data while fitting.
            This is checked by looking at the 'stateless' and 'requires_y' tags
        array_output: bool
            indicates if the transformed data is requested to be a NumPy array
            output. By default the value is False.
    Examples:
        Firstly, we will import and split the Berkeley Growth Study dataset

        >>> from skfda.datasets import fetch_growth
        >>> from sklearn.model_selection import train_test_split
        >>> X, y = fetch_growth(return_X_y=True, as_frame=True)
        >>> X = X.iloc[:, 0].values
        >>> y = y.values.codes
        >>> X_train, X_test, y_train, y_test = train_test_split(
        ...    X, y, test_size=0.25, stratify=y, random_state=0)

        >>> from skfda.preprocessing.dim_reduction.feature_extraction
        ... import PerClassTransformer

        Then we will need to select a fda transformer, and so we will
        use RecursiveMaximaHunting

        >>> from skfda.preprocessing.dim_reduction.variable_selection
        ... import RecursiveMaximaHunting
        >>> t = PerClassTransformer(RecursiveMaximaHunting(),
        ... array_output=True)

        Finally we need to fit the data and transform it

        >>> t.fit(X_train, y_train)
        >>> x_transformed = t.transform(X_test)

        x_transformed will be a vector with the transformed data
    """

    def __init__(
        self,
        transformer: TransformerMixin,
        *,
        array_output=False,
    ) -> None:
        self.transformer = transformer
        self.array_output = array_output

    def _validate_transformer(
        self,
    ) -> None:
        """
        Check that the transformer passed is\
        scikit-learn-like and that uses target data in fit.

        Args:
            None

        Returns:
            None
        """
        if not (hasattr(self.transformer, "fit")
                and hasattr(self.transformer, "transform")
                and hasattr(self.transformer, "fit_transform")
                ):
            raise TypeError(
                "Transformer should implement fit and "
                "transform. " + str(self.transformer)
                + " (type " + str(type(self.transformer)) + ")"
                " doesn't",
            )

        tags = self.transformer._get_tags()

        if tags['stateless'] and not tags['requires_y']:
            warnings.warn(
                "Transformer should use target data in fit."
                + str(self.transformer)
                + " (type " + str(type(self.transformer)) + ")"
                " doesn't",
            )

    def fit(
        self,
        X: T,
        y: ndarray,
    ) -> PerClassTransformer:
        """
        Fit the model on each class using X as\
        training data and y as target values.

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

    def transform(self, X: T) -> Union[DataFrame, ndarray]:
        """
        Transform the provided data using the already fitted transformer.

        Args:
            X: FDataGrid with the test samples.

        Returns:
            Eiter array of shape (n_samples, G) or a Data Frame \
            including the transformed data.
        """
        sklearn_check_is_fitted(self)
        transformed_data = [
            feature_transformer.transform(X)
            for feature_transformer in self._class_feature_transformers_
        ]

        if self.array_output:
            for i in transformed_data:
                if isinstance(i, FDataGrid or FDataBasis):
                    raise TypeError(
                        "There are transformed instances of FDataGrid or "
                        "FDataBasis that can't be concatenated on a NumPy "
                        "array.",
                    )
            return transformed_data

        for j in transformed_data:
            if not isinstance(j, FDataGrid or FDataBasis):
                raise TypeError(
                    "Transformed instance is not of type FDataGrid or"
                    " FDataBasis. It is " + type(j),
                )

        return DataFrame(
            {'Transformed data': transformed_data},
        )

    def fit_transform(self, X: T, y: ndarray) -> Union[DataFrame, ndarray]:
        """
        Fits and transforms the provided data\
        using the transformer specified when initializing the class.

        Args:
            X: FDataGrid with the samples.
            y: Target values of shape = (n_samples)

        Returns:
            Eiter array of shape (n_samples, G) or a Data Frame \
            including the transformed data.
        """
        return self.fit(X, y).transform(X)
