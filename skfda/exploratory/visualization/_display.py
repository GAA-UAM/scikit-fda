from abc import ABC, abstractmethod
from typing import List, Optional, TypeVar

import numpy as np
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from ._utils import (
    _get_figure_and_axes,
    _set_figure_layout,
    _set_figure_layout_for_fdata,
)

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
