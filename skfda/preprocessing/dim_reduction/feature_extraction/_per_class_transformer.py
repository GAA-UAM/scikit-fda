"""Feature extraction transformers for dimensionality reduction."""
from __future__ import annotations

import warnings
from typing import Any, Mapping, TypeVar, Union

import numpy as np
from pandas import DataFrame
from sklearn.utils.validation import check_is_fitted as sklearn_check_is_fitted

from ...._utils import TransformerMixin, _fit_feature_transformer
from ....representation import FData
from ....representation._typing import NDArrayFloat, NDArrayInt
from ....representation.basis import FDataBasis
from ....representation.grid import FDataGrid

Input = TypeVar("Input", bound=Union[FData, NDArrayFloat])
Output = TypeVar("Output", bound=Union[DataFrame, NDArrayFloat])
Target = TypeVar("Target", bound=NDArrayInt)

TransformerOutput = Union[FData, NDArrayFloat]


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
        >>> t1 = PerClassTransformer(
        ...     RecursiveMaximaHunting(),
        ...     array_output=True,
        ... )
        >>> x_transformed1 = t1.fit_transform(X, y)

        ``x_transformed1`` will be a vector with the transformed data.
        We will split the generated data and fit a KNN classifier.

        >>> from sklearn.model_selection import train_test_split
        >>> from sklearn.neighbors import KNeighborsClassifier
        >>> X_train1, X_test1, y_train1, y_test1 = train_test_split(
        ...     x_transformed1,
        ...     y,
        ...     test_size=0.25,
        ...     stratify=y,
        ...     random_state=0,
        ... )
        >>> neigh1 = KNeighborsClassifier()
        >>> neigh1 = neigh1.fit(X_train1, y_train1)

        Finally we can predict and check the score:
        >>> neigh1.predict(X_test1)
            array([0, 0, 1, 0, 1, 1, 1, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0,
            1, 1, 1, 1, 1, 1, 1], dtype=int8)

        >>> round(neigh1.score(X_test1, y_test1), 3)
            0.958



        We can also use a transformer that returns a FData object
        when predicting.
        In our example we are going to use the Nadaraya Watson Smoother.

        >>> from skfda.preprocessing.smoothing.kernel_smoothers import (
        ...     NadarayaWatsonSmoother,
        ... )
        >>> t2 = PerClassTransformer(
        ...     NadarayaWatsonSmoother(),
        ... )
        >>> x_transformed2 = t2.fit_transform(X, y)

        ``x_transformed2`` will be a DataFrame with the transformed data.
        Each row on the frame contains a FDataGrid describing a transformed
        curve.
        We need to convert the DataFrame into a FDataGrid with all the
        samples, so we can train a classifier. We also need to duplicate
        the outputs as we have the double amount of data curves:

        >>> for i, curve_grid in enumerate(x_transformed2.iloc[:,0].values):
        ...     if i == 0:
        ...         X_transformed_grid = curve_grid
        ...     else:
        ...         X_transformed_grid = X_transformed_grid.concatenate(
        ...                                 curve_grid,
        ...                              )

        >>> y = np.concatenate((y,y))


        ``X_transformed_grid`` contains a FDataGrid with all the transformed
        curves. Now we are able to use it to fit a KNN classifier.
        Again we split the data into train and test.
        >>> X_train2, X_test2, y_train2, y_test2 = train_test_split(
        ...     X_transformed_grid,
        ...     y,
        ...     test_size=0.25,
        ...     stratify=y,
        ...     random_state=0,
        ... )

        This time we need a functional data classifier.
        We fit the classifier and predict.
        >>> from skfda.ml.classification import KNeighborsClassifier
        >>> neigh2 = KNeighborsClassifier()
        >>> neigh2 = neigh2.fit(X_train2, y_train2)
        >>> neigh2.predict(X_test2)
            array([1, 0, 0, 0, 1, 0, 1, 1, 0, 0, 1, 1, 1, 0, 1, 1, 0, 1, 1, 1,
            0, 1, 1, 1, 0, 1, 1, 0, 0, 1, 1, 0, 1, 0, 0, 0, 0, 1, 0, 1, 0, 0,
            1, 1, 1, 0, 0], dtype=int8)

        >>> round(neigh2.score(X_test2, y_test2), 3)
            0.957

    """

    def __init__(
        self,
        transformer: TransformerMixin[Input, TransformerOutput, Target],
        *,
        array_output: bool = False,
    ) -> None:
        self.transformer = transformer
        self.array_output = array_output

    def _more_tags(self) -> Mapping[str, Any]:
        parent_tags = super()._more_tags()
        transformer_tags = self.transformer._get_tags()  # noqa: WPS437

        return {
            **parent_tags,
            'allow_nan': transformer_tags['allow_nan'],
            'non_deterministic': transformer_tags['non_deterministic'],
            'pairwise': transformer_tags['pairwise'],
            'requires_positive_X': transformer_tags['requires_positive_X'],
            'requires_y': True,
            'X_types': transformer_tags['X_types'],
        }

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

        if tags['stateless']:
            warnings.warn(
                f"Parameter 'transformer' with type "
                f"{type(self.transformer)} should use the data for "
                f" fitting."
                f"It should have the 'stateless' tag set to 'False'",
            )

        if tags['requires_y']:
            warnings.warn(
                f"Parameter 'transformer' with type "  # noqa: WPS237
                f"{type(self.transformer)} should not use the class label."
                f"It should have the 'requires_y' tag set to 'False'",
            )

    def fit(  # type: ignore[override]
        self,
        X: Input,
        y: Target,
    ) -> PerClassTransformer[Input, Output, Target]:
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

    def transform(self, X: Input) -> Output:
        """
        Transform the provided data using the already fitted transformer.

        Args:
            X: FDataGrid with the test samples.

        Returns:
            Eiter array of shape (n_samples, G) or a Data Frame
            including the transformed data.
        """
        sklearn_check_is_fitted(self)

        transformed_data = [
            feature_transformer.transform(X)
            for feature_transformer in self._class_feature_transformers_
        ]

        if self.array_output:
            for data in transformed_data:
                if isinstance(data, (FDataGrid, FDataBasis)):
                    raise TypeError(
                        "There are transformed instances of FDataGrid or "
                        "FDataBasis that can't be concatenated on a NumPy "
                        "array.",
                    )
            return np.hstack(transformed_data)

        return DataFrame(
            {'Transformed data': transformed_data},
        )

    def fit_transform(  # type: ignore[override]
        self,
        X: Input,
        y: Target,
    ) -> Output:
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
