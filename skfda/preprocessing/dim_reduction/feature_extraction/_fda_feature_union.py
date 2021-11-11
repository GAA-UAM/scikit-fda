"""Feature extraction union for dimensionality reduction."""
from __future__ import annotations

from typing import Any, Union

from numpy import ndarray
from pandas import DataFrame
from sklearn.pipeline import FeatureUnion

from ....representation.basis import FDataBasis
from ....representation.grid import FDataGrid


class FdaFeatureUnion(FeatureUnion):
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
    Firstly we will import the Berkeley Growth Study data set
    >>> from skfda.datasets import fetch_growth
    >>> X = fetch_growth(return_X_y=True)[0]

    Then we need to import the transformers we want to use
    >>> from skfda.preprocessing.dim_reduction.feature_extraction import (
    ...     FPCA,
    ...     FdaFeatureUnion,
    ... )
    >>> from skfda.representation import EvaluationTransformer

    Finally we apply fit and transform
    >>> union = FdaFeatureUnion(
    ...     [
    ...        ("eval", EvaluationTransformer()),
    ...        ("fpca", FPCA()),
    ...     ],
    ...     array_output=True,
    ... )
    >>> transformed_data = union.fit_transform(X)
    >>> transformed_data
        [[ 81.3       ,  84.2       ,  86.4       , ..., 105.84283261,
            -34.60733887, -14.97276458],
           [ 76.2       ,  80.4       ,  83.2       , ..., -11.42260839,
            -17.01293819,  24.77047871],
           [ 76.8       ,  79.8       ,  82.6       , ..., -33.81180503,
            -23.312921  ,   7.67421522],
           ...,
           [ 68.6       ,  73.6       ,  78.6       , ..., -19.49404628,
             12.76825883,   0.70188222],
           [ 79.9       ,  82.6       ,  84.8       , ...,  19.28399897,
             31.49601648,   6.54012077],
           [ 76.1       ,  78.4       ,  82.3       , ...,  17.71973789,
             27.7332045 ,  -1.70532625]]

    We can also concatenate the result with the
    original data on a Pandas DataFrame.
    >>> from pandas.core.frame import DataFrame
    >>> DataFrame({
    ...     "Data": [transformed_data, X.data_matrix]
    ... })
        Data
        0  [[81.3, 84.2, 86.4, 88.9, 91.4, 101.1, 109.5, ...
        1  [[[81.3], [84.2], [86.4], [88.9], [91.4], [101...

    """

    def __init__(
        self,
        transformer_list,
        *,
        n_jobs=None,
        transformer_weights=None,
        verbose=False,
        array_output=False,
    ) -> None:
        self.array_output = array_output
        super().__init__(
            transformer_list,
            n_jobs=n_jobs,
            transformer_weights=transformer_weights,
            verbose=verbose,
        )

    def _hstack(self, Xs) -> Union[DataFrame, ndarray, Any]:

        if self.array_output:
            for i in Xs:
                if isinstance(i, (FDataGrid, FDataBasis)):
                    raise TypeError(
                        "There are transformed instances of FDataGrid or "
                        "FDataBasis that can't be concatenated on a NumPy "
                        "array.",
                    )
            return super()._hstack(Xs)

        return DataFrame({'Transformed data': Xs})
