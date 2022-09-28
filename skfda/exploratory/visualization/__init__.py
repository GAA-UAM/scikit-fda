"""Initialization module of visualization folder."""

from typing import TYPE_CHECKING

import lazy_loader as lazy

__getattr__, __dir__, __all__ = lazy.attach(
    __name__,
    submodules=[
        "clustering",
        "representation",
    ],
    submod_attrs={
        "_baseplot": ["BasePlot"],
        "_boxplot": ["Boxplot", "SurfaceBoxplot"],
        "_ddplot": ["DDPlot"],
        "_magnitude_shape_plot": ["MagnitudeShapePlot"],
        "_multiple_display": ["MultipleDisplay"],
        "_outliergram": ["Outliergram"],
        "_parametric_plot": ["ParametricPlot"],
        "fpca": ["FPCAPlot"],
    },
)

if TYPE_CHECKING:
    from ._baseplot import BasePlot as BasePlot
    from ._boxplot import Boxplot as Boxplot, SurfaceBoxplot as SurfaceBoxplot
    from ._ddplot import DDPlot as DDPlot
    from ._magnitude_shape_plot import MagnitudeShapePlot as MagnitudeShapePlot
    from ._multiple_display import MultipleDisplay as MultipleDisplay
    from ._outliergram import Outliergram as Outliergram
    from ._parametric_plot import ParametricPlot as ParametricPlot
    from .fpca import FPCAPlot as FPCAPlot
