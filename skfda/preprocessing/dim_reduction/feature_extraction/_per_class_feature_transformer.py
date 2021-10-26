"""Feature extraction transformers for dimensionality reduction."""
from __future__ import annotations
import numpy as np
from typing import TypeVar
from sklearn.base import TransformerMixin
from sklearn.utils.validation import check_is_fitted as sklearn_check_is_fitted
from ....representation.grid import FData
from ...._utils import _classifier_fit_feature_transformer

T = TypeVar("T", bound=FData)

class PerClassFeatureTransformer(TransformerMixin):

    def __init__(
        self,
        transformer: TransformerMixin
    ) -> None:
        self.transformer= transformer
        self._validate_transformer()
    
    def _validate_transformer(
        self
    ) -> None:
        """
        Checks that the transformer passed is scikit-learn-like and that uses target data in fit

        Args:
            None

        Returns:
            None
        """
        if not (hasattr(self.transformer, "fit") or hasattr(self.transformer, "fit_transform")) or not hasattr(
                self.transformer, "transform"
            ):
                raise TypeError(
                    "Transformer should implement fit and "
                    "transform. '%s' (type %s) doesn't" % (self.transformer, type(self.transformer))
                )
        
        tags = self.transformer._get_tags()
        
        if not(tags['stateless'] and tags['requires_y']):
                raise TypeError(
                    "Transformer should use target data in fit."
                    " '%s' (type %s) doesn't" % (self.transformer, type(self.transformer))
                )
        
        
    def fit(
        self,
        X: T,
        y: np.ndarray
    ) -> PerClassFeatureTransformer:
        """
        Fit the model on each class using X as training data and y as target values.

        Args:
            X: FDataGrid with the training data.
            y: Target values of shape = (n_samples).

        Returns:
            self
        """
        classes, class_feature_transformers = _classifier_fit_feature_transformer(
            X, y, self.transformer
        )
        
        self._classes = classes
        self._class_feature_transformers_ = class_feature_transformers

        return self


    def transform(self, X: T) -> np.ndarray:
        """
        Transform the provided data using the already fitted transformer.

        Args:
            X: FDataGrid with the test samples.

        Returns:
            Array of shape (n_samples, G).
        """
        sklearn_check_is_fitted(self)
        
        return [
            feature_transformer.transform(X) 
            for feature_transformer in self._class_feature_transformers_
        ]


    def fit_transform(self, X: T, y: np.ndarray) -> np.ndarray:
        """
        Fits and transforms the provided data 
        using the transformer specified when initializing the class.

        Args:
            X: FDataGrid with the samples.
            y: Target values of shape = (n_samples)

        Returns:
            Array of shape (n_samples, G).
        """
        return self.fit(X, y).transform(X)
