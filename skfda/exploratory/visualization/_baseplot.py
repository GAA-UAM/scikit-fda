"""BasePlot Module.

This module contains the abstract class of which inherit all
the visualization modules, containing the basic functionality
common to all of them.
"""

from abc import ABC, abstractmethod
from typing import Optional, Sequence, Union

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
    ) -> None:
        self.artists: np.ndarray

    @abstractmethod
    def plot(
        self,
    ) -> Figure:
        """
        Abstract method used to plot the object and its data.

        Returns:
            Figure: figure object in which the displays and
                widgets will be plotted.
        """
        pass

    @property
    def n_subplots(self) -> int:
        """Get the number of subplots that this plot uses."""
        return 1

    @abstractmethod
    def n_samples(self) -> int:
        """Get the number of instances that will be used for interactivity."""
        pass

    def _set_figure_and_axes(
        self,
        chart: Union[Figure, Axes, None] = None,
        *,
        fig: Optional[Figure] = None,
        axes: Union[Axes, Sequence[Axes], None] = None,
    ) -> None:
        fig, axes = _get_figure_and_axes(chart, fig, axes)
        fig, axes = _set_figure_layout(fig, axes)

        self.fig = fig
        self.axes = axes

    def _repr_svg_(self) -> str:
        """Automatically represents the object as an svg when calling it."""
        self.fig = self.plot()
        plt.close(self.fig)
        return _figure_to_svg(self.fig)
