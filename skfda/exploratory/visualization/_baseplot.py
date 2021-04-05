from abc import ABC, abstractmethod
from typing import List, Optional, Sequence, TypeVar, Union

import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from ._utils import _figure_to_svg

S = TypeVar('S', Figure, Axes, List[Axes])


class BasePlot(ABC):
    @abstractmethod
    def __init__(
        self,
    ) -> None:
        self.id_function = []

    @abstractmethod
    def plot(
        self,
    ) -> Figure:
        pass

    @abstractmethod
    def num_instances(self) -> int:
        pass

    @abstractmethod
    def set_figure_and_axes(
        self,
        chart: Union[Figure, Axes, None] = None,
        fig: Optional[Figure] = None,
        axes: Union[Axes, Sequence[Axes], None] = None,
    ) -> None:
        pass

    def clear_ax(self) -> None:
        for ax in self.axes:
            ax.clear()
        if len(self.id_function) != 0:
            self.id_function = []

    def _repr_svg_(self):
        self.fig = self.plot()
        plt.close(self.fig)
        return _figure_to_svg(self.fig)

