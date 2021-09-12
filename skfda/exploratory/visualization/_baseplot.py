"""BasePlot Module.

This module contains the abstract class of which inherit all
the visualization modules, containing the basic functionality
common to all of them.
"""

from abc import ABC, abstractmethod
from typing import Optional, Sequence, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from ._utils import _figure_to_svg, _get_figure_and_axes, _set_figure_layout


class BasePlot(ABC):
    """
    BasePlot class.

    Attributes:
        artists: List of Artist objects corresponding
            to every instance of our plot. They will be used to modify
            the visualization with interactivity and widgets.
        fig: Figure over with the graphs are plotted.
        axes: Sequence of axes where the graphs are plotted.
    """

    @abstractmethod
    def __init__(
        self,
        chart: Union[Figure, Axes, None] = None,
        *,
        fig: Optional[Figure] = None,
        axes: Union[Axes, Sequence[Axes], None] = None,
        n_rows: Optional[int] = None,
        n_cols: Optional[int] = None,
    ) -> None:
        self.artists: Optional[np.ndarray] = None
        self.chart = chart
        self.fig = fig
        self.axes = axes
        self.n_rows = n_rows
        self.n_cols = n_cols

    def _plot(
        self,
        fig: Figure,
        axes: Sequence[Axes],
    ) -> None:
        pass

    def plot(
        self,
    ) -> Figure:
        """
        Abstract method used to plot the object and its data.

        Returns:
            Figure: figure object in which the displays and
                widgets will be plotted.
        """
        fig = getattr(self, "fig_", None)
        axes = getattr(self, "axes_", None)

        if fig is None:
            fig, axes = self._set_figure_and_axes(
                self.chart,
                fig=self.fig,
                axes=self.axes,
            )

        self._plot(fig, axes)
        return fig

    @property
    def dim(self) -> int:
        """Get the number of dimensions for this plot."""
        return 2

    @property
    def n_subplots(self) -> int:
        """Get the number of subplots that this plot uses."""
        return 1

    @property
    def n_samples(self) -> Optional[int]:
        """Get the number of instances that will be used for interactivity."""
        return None

    def _set_figure_and_axes(
        self,
        chart: Union[Figure, Axes, None] = None,
        *,
        fig: Optional[Figure] = None,
        axes: Union[Axes, Sequence[Axes], None] = None,
    ) -> Tuple[Figure, Sequence[Axes]]:
        fig, axes = _get_figure_and_axes(chart, fig, axes)
        fig, axes = _set_figure_layout(
            fig=fig,
            axes=axes,
            dim=self.dim,
            n_axes=self.n_subplots,
            n_rows=self.n_rows,
            n_cols=self.n_cols,
        )

        self.fig_ = fig
        self.axes_ = axes

        return fig, axes

    def _repr_svg_(self) -> str:
        """Automatically represents the object as an svg when calling it."""
        self.fig = self.plot()
        plt.close(self.fig)
        return _figure_to_svg(self.fig)
