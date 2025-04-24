"""Outlier detection methods."""
from typing import TYPE_CHECKING

import lazy_loader as lazy

__getattr__, __dir__, __all__ = lazy.attach(
    __name__,
    submod_attrs={
        '_boxplot': ["BoxplotOutlierDetector"],  # noqa: Q000
        "_directional_outlyingness": [
            "MSPlotOutlierDetector",
            "directional_outlyingness_stats",
        ],
        "_outliergram": ["OutliergramOutlierDetector"],
        "neighbors_outlier": ["LocalOutlierFactor"],
    },
)

if TYPE_CHECKING:
    from ._boxplot import BoxplotOutlierDetector as BoxplotOutlierDetector  # noqa: I001
    from ._directional_outlyingness import (
        MSPlotOutlierDetector as MSPlotOutlierDetector,
        directional_outlyingness_stats as directional_outlyingness_stats,
    )
    from ._outliergram import (
        OutliergramOutlierDetector as OutliergramOutlierDetector  # noqa: COM812
    )
    from .neighbors_outlier import LocalOutlierFactor as LocalOutlierFactor
