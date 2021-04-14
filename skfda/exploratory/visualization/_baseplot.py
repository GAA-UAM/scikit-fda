"""BasePlot Module.

This module contains the abstract class of which inherit all
the visualization modules, containing the basic functionality
common to all of them.
"""

from abc import ABC, abstractmethod
from typing import List, Optional, Sequence, Union

import matplotlib.pyplot as plt
from matplotlib.artist import Artist
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from ._utils import _figure_to_svg


class BasePlot(ABC):
    """
    BasePlot class.

    Attributes:
        id_function: list of Artist objects corresponding
            to every instance of our plot. They will be used to modify
            the visualization with interactivity and widgets.
    """

    @abstractmethod
    def __init__(
        self,
    ) -> None:
        self.id_function: List[Artist] = []

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

    @abstractmethod
    def num_instances(self) -> int:
        """Get the number of instances that will be used for interactivity."""
        pass

    @abstractmethod
    def set_figure_and_axes(
        self,
        chart: Union[Figure, Axes, None] = None,
        fig: Optional[Figure] = None,
        axes: Union[Axes, Sequence[Axes], None] = None,
    ) -> None:
        """
        Initialize the axes and fig of the plot.

        Args:
            chart: figure over with the graphs are plotted or axis over
                where the graphs are plotted. If None and ax is also
                None, the figure is initialized.
            fig: figure over with the graphs are plotted in case ax is not
                specified. If None and ax is also None, the figure is
                initialized.
            axes: axis where the graphs are plotted. If None, see param fig.
        """
        pass

    def clear_ax(self) -> None:
        """
        Reset the basic attributes of the BasePlot.

        Clear the old axes of the BasePlot and reset the
        id_function list.
        """
        for ax in self.axes:
            ax.clear()
        if len(self.id_function) != 0:
            self.id_function = []

    def _repr_svg_(self) -> str:
        """Automatically represents the object as an svg when calling it."""
        self.fig = self.plot()
        plt.close(self.fig)
        return _figure_to_svg(self.fig)
