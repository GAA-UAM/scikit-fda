"""Feature extraction union for dimensionality reduction."""
from __future__ import annotations

from typing import Union

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
    >>> X,y = fetch_growth(return_X_y=True)

    Then we need to import the transformers we want to use. In our case we
    will use FPCA and Minimum Redundancy Maximum Relevance.
    Evaluation Transformer returns the original curve, and as it is helpful,
    we will concatenate it to the already metioned transformers.
    >>> from skfda.preprocessing.dim_reduction.feature_extraction import (
    ...     FPCA,
    ...     FdaFeatureUnion,
    ... )
    >>> from skfda.preprocessing.dim_reduction.variable_selection import (
    ...     MinimumRedundancyMaximumRelevance,
    ... )
    >>> from skfda.representation import EvaluationTransformer

    Finally we apply fit and transform.
    >>> union = FdaFeatureUnion(
    ...     [
    ...        ("mrmr", MinimumRedundancyMaximumRelevance()),
    ...        ("fpca", FPCA()),
    ...        ("eval", EvaluationTransformer()),
    ...     ],
    ...     array_output=True,
    ... )
    >>> union.fit_transform(X,y)
      [[194.3       , 105.84, -34.61, ..., 193.8       ,
        194.3       , 195.1       ],
       [177.4       , -11.42, -17.01, ..., 176.1       ,
        177.4       , 178.7       ],
       [171.2       , -33.81, -23.31 , ..., 170.9       ,
        171.2       , 171.5       ],
       ...,
       [166.3       , -19.49  12.77, ..., 166.        ,
        166.3       , 166.8       ],
       [168.4       ,  19.28,  31.5, ..., 168.3       ,
        168.4       , 168.6       ],
       [168.9       ,  17.72,  27.73 , ..., 168.6       ,
        168.9       , 169.2       ]]
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

    def _hstack(self, Xs) -> Union[DataFrame, ndarray]:

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
