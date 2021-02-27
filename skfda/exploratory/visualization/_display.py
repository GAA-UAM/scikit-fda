from abc import ABC, abstractmethod
from typing import List, Optional, Sequence, TypeVar, Union

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.artist import Artist
from matplotlib.axes import Axes
from matplotlib.backend_bases import Event
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
        self.point_clicked: Artist = None

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

    def init_axes(
        self,
        chart: Union[Figure, Axes, None] = None,
        fig: Optional[Figure] = None,
        axes: Union[Axes, Sequence[Axes], None] = None,
        n_rows: Optional[int] = None,
        n_cols: Optional[int] = None,
    ) -> Figure
    :
        if self.extra_displays is None:
            fig, self.axes = _get_figure_and_axes(chart, fig, axes)
            fig, self.axes = _set_figure_layout_for_fdata(
                fdata=self.fdata,
                fig=fig,
                axes=self.axes,
                n_rows=n_rows,
                n_cols=n_cols,
            )
        else:
            if fig is None:
                fig = plt.figure(figsize=(9, 3))

            fig, self.axes = _get_figure_and_axes(chart, fig, axes)

            if isinstance(self.extra_displays, Display):
                fig, self.axes = _set_figure_layout(
                    fig=fig, axes=self.axes, n_axes=2,
                )
            else:
                fig, self.axes = _set_figure_layout(
                    fig=fig, axes=self.axes, n_axes=len(self.extra_displays),
                )
  
        return fig

    def display_extra(self):
        if isinstance(self.extra_displays, Display):
            self.extra_displays.plot(axes=self.axes[1])
            if (len(self.id_function) != len(self.extra_displays.id_function)):
                raise ValueError(
                    "Length of the first data set ",
                    "is not equal to the second dataset.",
                )
        elif self.extra_displays is not None:
            for ax, display in zip(self.axes[1:], self.extra_displays):
                display.plot(axes=ax)
                if (len(self.id_function) != len(display.id_function)):
                    raise ValueError(
                        "Length of the first data set ",
                        "is not equal to other dataset.",
                    )

    def pick(self, event: Event) -> None:
        if self.point_clicked is None:
            self.point_clicked = event.artist
            self.reduce_points_intensity()
        elif self.point_clicked == event.artist:
            self.restore_points_intensity()
            self.point_clicked = None
        else:
            self.change_points_intensity(event.artist)
            self.point_clicked = event.artist

    def reduce_points_intensity(self) -> None:
        for i in range(len(self.id_function)):
            if not (
                np.ma.getdata(self.id_function[i].get_offsets())[0][0]
                == np.ma.getdata(self.point_clicked.get_offsets())[0][0]
            ) or not (
                np.ma.getdata(self.id_function[i].get_offsets())[0][1]
                == np.ma.getdata(self.point_clicked.get_offsets())[0][1]
            ):
                self.id_function[i].set_alpha(0.5)
                if isinstance(self.extra_displays, Display):
                    self.extra_displays.id_function[i].set_alpha(0.1)
                elif self.extra_displays is not None:
                    for display in self.extra_displays:
                        display.id_function[i].set_alpha(0.1)

    def restore_points_intensity(self) -> None:
        for i in range(len(self.id_function)):
            self.id_function[i].set_alpha(1)
            if isinstance(self.extra_displays, Display):
                self.extra_displays.id_function[i].set_alpha(1)
            elif self.extra_displays is not None:
                for display in self.extra_displays:
                    display.id_function[i].set_alpha(1)

    def change_points_intensity(self, new_point: Artist) -> None:
        for i in range(len(self.id_function)):
            if (
                np.ma.getdata(self.id_function[i].get_offsets())[0][0]
                == np.ma.getdata(new_point.get_offsets())[0][0]
            ) and (
                np.ma.getdata(self.id_function[i].get_offsets())[0][1]
                == np.ma.getdata(new_point.get_offsets())[0][1]
            ):
                self.id_function[i].set_alpha(1)
                if isinstance(self.extra_displays, Display):
                    self.extra_displays.id_function[i].set_alpha(1)
                elif self.extra_displays is not None:
                    for display in self.extra_displays:
                        display.id_function[i].set_alpha(1)

            else:
                self.id_function[i].set_alpha(0.5)
                if isinstance(self.extra_displays, Display):
                    self.extra_displays.id_function[i].set_alpha(0.1)
                elif self.extra_displays is not None:
                    for display in self.extra_displays:
                        display.id_function[i].set_alpha(0.1)
