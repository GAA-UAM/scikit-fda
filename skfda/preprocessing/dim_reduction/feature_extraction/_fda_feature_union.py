"""Feature extraction union for dimensionality reduction."""
from __future__ import annotations
from typing import Any
from numpy import ndarray
from pandas import DataFrame
from sklearn.pipeline import FeatureUnion
from ....representation.grid import FDataGrid
from ....representation.basis import FDataBasis

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
        transformer_list:
            List of tuple containing `(str, transformer)`. The first element
            of the tuple is name affected to the transformer while the
            second element is a scikit-learn transformer instance.
            The transformer instance can also be `"drop"` for it to be
            ignored.
        n_jobs:
            Number of jobs to run in parallel.
            ``None`` means 1 unless in a :obj:`joblib.parallel_backend` context.
            ``-1`` means using all processors.
            The default value is None
        transformer_weights:
            Multiplicative weights for features per transformer.
            Keys are transformer names, values the weights.
            Raises ValueError if key not present in ``transformer_list``.
        verbose:
            If True, the time elapsed while fitting each transformer will be
            printed as it is completed.
        np_array_output: 
            indicates if the transformed data is requested to be a NumPy array
            output. By default the value is False.
    
    Examples:
    Firstly we will import the Berkeley Growth Study data set
    >>> from skfda.datasets import fetch_growth
    >>> X, y= fetch_growth(return_X_y=True, as_frame=True)
    >>> X = X.iloc[:, 0].values
    
    Then we need to import the transformers we want to use
    >>> from skfda.preprocessing.dim_reduction.feature_extraction import FPCA
    >>> from skfda.representation import EvaluationTransformer
    
    Finally we import the union and apply fit and transform
    >>> from skfda.preprocessing.dim_reduction.feature_extraction._fda_feature_union
    ... import FdaFeatureUnion
    >>> union = FdaFeatureUnion([
    ...    ("Eval", EvaluationTransformer()),
    ...    ("fpca", FPCA()), ], np_array_output=True)   
    >>> union.fit_transform(X)
    """
    def __init__(
        self,
        transformer_list,
        *,
        n_jobs=None,
        transformer_weights=None,
        verbose=False,
        np_array_output=False
    ) -> None :
        self.np_array_output = np_array_output
        super().__init__(transformer_list, n_jobs=n_jobs, transformer_weights = transformer_weights, verbose=verbose)
        


    def _hstack(self, Xs) -> (ndarray | DataFrame | Any):

        if (self.np_array_output):
            for i in Xs:
                if(isinstance(i, FDataGrid) or isinstance(i, FDataBasis)):
                    raise TypeError(
                    "There are transformed instances of FDataGrid or FDataBasis"
                    " that can't be concatenated on a NumPy array."
                )
            return super()._hstack(Xs)

        first_grid = True
        first_basis = True
        for j in Xs:
            if isinstance(j, FDataGrid):
                if first_grid:
                    curves = j
                    first_grid = False
                else:
                    curves = curves.concatenate(j)
            elif isinstance(j, FDataBasis):
                if first_basis:
                    target = j
                    first_basis = False
                else:
                    target = target.concatenate(j)
            else: 
                raise TypeError(
                    "Transformed instance is not of type FDataGrid or FDataBasis."
                    "It is %s" %(type(j))
                )

        feature_name = curves.dataset_name.lower() + " transformed"
        target_name  = "transformed target"
        if first_grid: # There are only FDataBasis
            return DataFrame({
                target_name:target
            })
        elif first_basis: # There are only FDataGrids
            return DataFrame({
                feature_name:curves
            })
        else:
            return DataFrame({
                feature_name : curves,
                target_name: target,
            })
