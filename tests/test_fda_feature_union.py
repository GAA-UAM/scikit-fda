"""Test to check the Fda Feature Union module"""
from pandas.core.frame import DataFrame
from skfda.preprocessing.dim_reduction.feature_extraction._fda_feature_union import FdaFeatureUnion
from skfda.preprocessing.dim_reduction.feature_extraction import FPCA
from skfda.preprocessing.smoothing.kernel_smoothers import NadarayaWatsonSmoother
from skfda.representation import EvaluationTransformer
from skfda.misc.operators import SRSF
from skfda.datasets import fetch_growth
import unittest


class TestFdaFeatureUnion(unittest.TestCase):
    def setUp(self) -> None:
        X, y= fetch_growth(return_X_y=True, as_frame=True)
        self.X = X.iloc[:, 0].values
    
    def test_incompatible_array_output(self):
       
        u = FdaFeatureUnion([("EvaluationT", EvaluationTransformer()), ("fpca", FPCA()), ], np_array_output=False)
        self.assertRaises(TypeError, u.fit_transform, self.X)
    
    def test_incompatible_FDataGrid_output(self):
       
        u = FdaFeatureUnion([("EvaluationT", EvaluationTransformer()), ("srsf",SRSF()), ], np_array_output=True)
        self.assertRaises(TypeError, u.fit_transform, self.X)
        
    def test_correct_transformation_concat(self):
        u = FdaFeatureUnion([("srsf1",SRSF()), ("smooth",NadarayaWatsonSmoother())])
        created_frame = u.fit_transform(self.X)

        t1 = SRSF().fit_transform(self.X)
        t2 = NadarayaWatsonSmoother().fit_transform(self.X)
        t = t1.concatenate(t2)

        true_frame = DataFrame({
            t.dataset_name.lower() + " transformed": t
        })

        self.assertEqual(True, true_frame.equals(created_frame))
       
    

if __name__ == '__main__':
    unittest.main()
