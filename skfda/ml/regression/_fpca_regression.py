from __future__ import annotations

from typing import TypeVar

from sklearn.utils.validation import check_is_fitted

from ..._utils._sklearn_adapter import BaseEstimator, RegressorMixin
from ...misc.regularization import L2Regularization
from ...preprocessing.dim_reduction import FPCA
from ...representation import FData
from ...representation.basis import Basis, CustomBasis, FDataBasis
from ...typing._numpy import NDArrayFloat
from ._linear_regression import LinearRegression

FPCARegressionSelf = TypeVar("FPCARegressionSelf", bound="FPCARegression")


class FPCARegression(
    BaseEstimator,
    RegressorMixin,
):
    r"""Regression using Functional Principal Components Analysis.

    It performs Functional Principal Components Analysis to reduce the
    dimension of the functional data, and then uses a linear regression model
    to relate the transformed data to a scalar value.

    Args:
        n_components: Number of principal components to keep. Defaults to 5.
        fit\_intercept: If True, the linear model is calculated with an
            intercept.  Defaults to ``True``.
        pca_regularization: Regularization parameter for the principal
            component extraction. If None then no regularization is applied.
            Defaults to ``None``.
        regression_regularization: Regularization parameter for the linear
            regression. If None then no regularization is applied.
            Defaults to ``None``.
        components_basis: Basis used for the principal components. If None
            then the basis of the input data is used. Defaults to None.
            It is only used if the input data is a FDataBasis object.

    Attributes:
        n\_components\_: Number of principal components used.
        components\_: Principal components.
        coef\_: Coefficients of the linear regression model.
        explained\_variance\_: Amount of variance explained by
            each of the selected components.
        explained\_variance\_ratio\_: Percentage of variance
            explained by each of the selected components.

    Examples:
        Using the Berkeley Growth Study dataset, we can fit the model.

        >>> import skfda
        >>> dataset = skfda.datasets.fetch_growth()
        >>> fd = dataset["data"]
        >>> y = dataset["target"]
        >>> reg = skfda.ml.regression.FPCARegression(n_components=2)
        >>> reg.fit(fd, y)
        FPCARegression(n_components=2)

        Then, we can predict the target values and calculate the
        score.

        >>> score = reg.score(fd, y)
        >>> reg.predict(fd) # doctest:+ELLIPSIS
        array([...])

    """

    def __init__(
        self,
        n_components: int = 5,
        fit_intercept: bool = True,
        pca_regularization: L2Regularization | None = None,
        regression_regularization: L2Regularization | None = None,
        components_basis: Basis | None = None,
    ) -> None:
        self.n_components = n_components
        self.fit_intercept = fit_intercept
        self.pca_regularization = pca_regularization
        self.regression_regularization = regression_regularization
        self.components_basis = components_basis

    def fit(
        self,
        X: FData,
        y: NDArrayFloat,
    ) -> FPCARegressionSelf:
        """Fit the model according to the given training data.

        Args:
            X: Functional data.
            y: Target values.

        Returns:
            self

        """
        self._fpca = FPCA(
            n_components=self.n_components,
            centering=True,
            regularization=self.pca_regularization,
            components_basis=self.components_basis,
        )
        self._linear_model = LinearRegression(
            fit_intercept=self.fit_intercept,
            regularization=self.regression_regularization,
        )
        transformed_coefficients = self._fpca.fit_transform(X)

        # The linear model is fitted with the observations expressed in the
        # basis of the principal components.
        self.fpca_basis = CustomBasis(
            fdata=self._fpca.components_,
        )

        X_transformed = FDataBasis(
            basis=self.fpca_basis,
            coefficients=transformed_coefficients,
        )
        self._linear_model.fit(X_transformed, y)

        self.n_components_ = self.n_components
        self.components_ = self._fpca.components_
        self.coef_ = self._linear_model.coef_
        self.explained_variance_ = self._fpca.explained_variance_
        self.explained_variance_ratio_ = self._fpca.explained_variance_ratio_

        return self

    def predict(
        self,
        X: FData,
    ) -> NDArrayFloat:
        """Predict using the linear model.

        Args:
            X: Functional data.

        Returns:
            Target values.

        """
        check_is_fitted(self)

        X_transformed = FDataBasis(
            basis=self.fpca_basis,
            coefficients=self._fpca.transform(X),
        )

        return self._linear_model.predict(X_transformed)
