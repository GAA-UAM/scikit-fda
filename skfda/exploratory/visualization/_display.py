from abc import ABC, abstractmethod
from typing import List, Optional, TypeVar

from matplotlib.axes import Axes
from matplotlib.figure import Figure

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
        chart: Optional[S] = None,
        *,
        fig: Optional[Figure] = None,
        interactivity_mode: bool = True,
        **kwargs,
    ) -> Figure:
        pass

    @abstractmethod
    def num_instances(self) -> int:
        pass
