"""Feature extraction union for dimensionality reduction."""
from __future__ import annotations

from typing import Any, Mapping, Sequence, Tuple, Union

import pandas as pd
from sklearn.pipeline import FeatureUnion

from ..._utils._sklearn_adapter import TransformerMixin
from ...representation import FData
from ...typing._numpy import NDArrayAny


class FDAFeatureUnion(FeatureUnion):  # type: ignore[misc]
    """Concatenates results of multiple functional transformer objects.

    This estimator applies a list of transformer objects in parallel to the
    input data, then concatenates the results (They can be either FDataGrid
    and FDataBasis objects or multivariate data itself).This is useful to
    combine several feature extraction mechanisms into a single transformer.
    Parameters of the transformers may be set using its name and the parameter
    name separated by a '__'. A transformer may be replaced entirely by
    setting the parameter with its name to another transformer,
    or removed by setting to 'drop'.

    Parameters:
        transformer_list: list of tuple
            List of tuple containing `(str, transformer)`. The first element
            of the tuple is name affected to the transformer while the
            second element is a scikit-learn transformer instance.
            The transformer instance can also be `"drop"` for it to be
            ignored.
        n_jobs: int
            Number of jobs to run in parallel.
            ``None`` means 1 unless in a :obj:`joblib.parallel_backend`
            context.
            ``-1`` means using all processors.
            The default value is None
        transformer_weights: dict
            Multiplicative weights for features per transformer.
            Keys are transformer names, values the weights.
            Raises ValueError if key not present in ``transformer_list``.
        verbose: bool
            If True, the time elapsed while fitting each transformer will be
            printed as it is completed. By default the value is False
        array_output: bool
            indicates if the transformed data is requested to be a NumPy array
            output. By default the value is False.

    Examples:
        Firstly we will import the Berkeley Growth Study data set:

        >>> from skfda.datasets import fetch_growth
        >>> X,y = fetch_growth(return_X_y=True)

        Then we need to import the transformers we want to use. In our case we
        will use the Recursive Maxima Hunting method to select important
        features.
        We will concatenate to the results of the previous method the original
        curves with an Evaluation Transfomer.

        >>> from skfda.preprocessing.feature_construction import (
        ...     FDAFeatureUnion,
        ... )
        >>> from skfda.preprocessing.dim_reduction.variable_selection import (
        ...     RecursiveMaximaHunting,
        ... )
        >>> from skfda.preprocessing.feature_construction import (
        ...     EvaluationTransformer,
        ... )
        >>> import numpy as np

        Finally we apply fit and transform.

        >>> union = FDAFeatureUnion(
        ...     [
        ...        ("rmh", RecursiveMaximaHunting()),
        ...        ("eval", EvaluationTransformer()),
        ...     ],
        ...     array_output=True,
        ... )
        >>> np.around(union.fit_transform(X,y), decimals=2)
        array([[ 195.1,  141.1,  163.8, ...,  193.8,  194.3,  195.1],
               [ 178.7,  133. ,  148.1, ...,  176.1,  177.4,  178.7],
               [ 171.5,  126.5,  143.6, ...,  170.9,  171.2,  171.5],
                ...,
               [ 166.8,  132.8,  152.2, ...,  166. ,  166.3,  166.8],
               [ 168.6,  139.4,  161.6, ...,  168.3,  168.4,  168.6],
               [ 169.2,  138.1,  161.7, ...,  168.6,  168.9,  169.2]])
    """

    def __init__(
        self,
        transformer_list: Sequence[
            Tuple[str, TransformerMixin[Any, Any, Any]],
        ],
        *,
        n_jobs: int = 1,
        transformer_weights: Mapping[str, float] | None = None,
        verbose: bool = False,
        array_output: bool = False,
    ) -> None:
        self.array_output = array_output
        super().__init__(
            transformer_list,
            n_jobs=n_jobs,
            transformer_weights=transformer_weights,
            verbose=verbose,
        )

    def _hstack(self, Xs: NDArrayAny) -> Union[pd.DataFrame, NDArrayAny]:

        if self.array_output:
            for i in Xs:
                if isinstance(i, FData):
                    raise TypeError(
                        "There are transformed instances of FDataGrid or "
                        "FDataBasis that can't be concatenated on a NumPy "
                        "array.",
                    )
            return super()._hstack(Xs)

        return pd.concat(
            [
                pd.DataFrame({0: data})
                for data in Xs
            ],
            axis=1,
        )
