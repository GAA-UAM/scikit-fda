"""Feature extraction union for dimensionality reduction."""
from __future__ import annotations

from typing import Union

from numpy import ndarray
from pandas import DataFrame
from sklearn.pipeline import FeatureUnion

from ....representation import FData


class FDAFeatureUnion(FeatureUnion):  # type: ignore
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
    will use Generalized depth-versus-depth transformer.
    Evaluation Transformer returns the original curve, and as it is helpful,
    we will concatenate it to the already metioned transformer.
    >>> from skfda.preprocessing.dim_reduction.feature_extraction import (
    ...     FDAFeatureUnion,
    ... )
    >>> from skfda.preprocessing.dim_reduction.feature_extraction import (
    ...     DDGTransformer,
    ... )
    >>> from skfda.exploratory.depth import ModifiedBandDepth
    >>> from skfda.representation import EvaluationTransformer
    >>> import numpy as np

    Finally we apply fit and transform.
    >>> union = FDAFeatureUnion(
    ...     [
    ...        (
    ...             'ddgtransformer',
    ...             DDGTransformer(depth_method=[ModifiedBandDepth()]),
    ...        ),
    ...        ("eval", EvaluationTransformer()),
    ...     ],
    ...     array_output=True,
    ... )
    >>> np.around(union.fit_transform(X,y), decimals=2)
      array([[ 2.100e-01,  9.000e-02,  8.130e+01, ...,  1.938e+02,  1.943e+02,
               1.951e+02],
             [ 4.600e-01,  3.800e-01,  7.620e+01, ...,  1.761e+02,  1.774e+02,
               1.787e+02],
             [ 2.000e-01,  3.300e-01,  7.680e+01, ...,  1.709e+02,  1.712e+02,
               1.715e+02],
             ...,
             [ 3.900e-01,  5.100e-01,  6.860e+01, ...,  1.660e+02,  1.663e+02,
               1.668e+02],
             [ 2.600e-01,  2.700e-01,  7.990e+01, ...,  1.683e+02,  1.684e+02,
               1.686e+02],
             [ 3.300e-01,  3.200e-01,  7.610e+01, ...,  1.686e+02,  1.689e+02,
               1.692e+02]])
    """

    def __init__(
        self,
        transformer_list: list,  # type: ignore
        *,
        n_jobs: int = 1,
        transformer_weights: dict = None,  # type: ignore
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

    def _hstack(self, Xs: ndarray) -> Union[DataFrame, ndarray]:

        if self.array_output:
            for i in Xs:
                if isinstance(i, FData):
                    raise TypeError(
                        "There are transformed instances of FDataGrid or "
                        "FDataBasis that can't be concatenated on a NumPy "
                        "array.",
                    )
            return super()._hstack(Xs)

        return DataFrame({'Transformed data': Xs})
