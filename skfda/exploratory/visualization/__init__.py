"""Initialization module of visualization folder."""

from . import clustering, representation
from ._baseplot import BasePlot
from ._boxplot import Boxplot, SurfaceBoxplot
from ._ddplot import DDPlot
from ._magnitude_shape_plot import MagnitudeShapePlot
from ._multiple_display import MultipleDisplay
from ._outliergram import Outliergram
from ._parametric_plot import ParametricPlot
from .fpca import FPCAPlot
