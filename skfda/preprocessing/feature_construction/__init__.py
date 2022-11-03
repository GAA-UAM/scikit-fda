"""Feature extraction."""
from typing import TYPE_CHECKING

import lazy_loader as lazy

__getattr__, __dir__, __all__ = lazy.attach(
    __name__,
    submod_attrs={
        "_coefficients_transformer": ["CoefficientsTransformer"],
        "_evaluation_trasformer": ["EvaluationTransformer"],
        "_fda_feature_union": ["FDAFeatureUnion"],
        "_function_transformers": [
            "LocalAveragesTransformer",
            "NumberCrossingsTransformer",
            "OccupationMeasureTransformer",
        ],
        "_functions": [
            "local_averages",
            "number_crossings",
            "occupation_measure",
            "unconditional_central_moment",
            "unconditional_expected_value",
            "unconditional_moment",
        ],
        "_per_class_transformer": ["PerClassTransformer"],
    },
)

if TYPE_CHECKING:
    from ._coefficients_transformer import (
        CoefficientsTransformer as CoefficientsTransformer,
    )
    from ._evaluation_trasformer import (
        EvaluationTransformer as EvaluationTransformer,
    )
    from ._fda_feature_union import FDAFeatureUnion as FDAFeatureUnion
    from ._function_transformers import (
        LocalAveragesTransformer as LocalAveragesTransformer,
        NumberCrossingsTransformer as NumberCrossingsTransformer,
        OccupationMeasureTransformer as OccupationMeasureTransformer,
    )
    from ._functions import (
        local_averages as local_averages,
        number_crossings as number_crossings,
        occupation_measure as occupation_measure,
        unconditional_central_moment as unconditional_central_moment,
        unconditional_expected_value as unconditional_expected_value,
        unconditional_moment as unconditional_moment,
    )
    from ._per_class_transformer import (
        PerClassTransformer as PerClassTransformer,
    )
