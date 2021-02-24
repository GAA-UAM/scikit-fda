"""Initialization module of visualization folder."""

from . import clustering, representation
from ._boxplot import Boxplot, SurfaceBoxplot
from ._ddplot import DDPlot
from ._magnitude_shape_plot import MagnitudeShapePlot
from ._phase_plane_plot import PhasePlanePlot
from .fpca import plot_fpca_perturbation_graphs
