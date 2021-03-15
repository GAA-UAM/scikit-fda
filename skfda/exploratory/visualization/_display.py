from abc import ABC, abstractmethod
from typing import List, Optional, Sequence, TypeVar, Union

from matplotlib.axes import Axes
from matplotlib.figure import Figure

from ._utils import _get_figure_and_axes

S = TypeVar('S', Figure, Axes, List[Axes])


class Display(ABC):
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
