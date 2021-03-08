import collections
from typing import Any, Optional, Sequence, Union

import ipywidgets as widgets
import numpy as np
from IPython.display import display
from matplotlib.artist import Artist
from matplotlib.axes import Axes
from matplotlib.backend_bases import Event
from matplotlib.figure import Figure

from ._display import Display
from ._utils import _get_figure_and_axes, _set_figure_layout


class MultipleDisplay:
    def __init__(
        self,
        displays: Union[Display, Sequence[Display]],
    ):
        if isinstance(displays, Display):
            self.displays = [displays]
        else:
            self.displays = displays
        self.point_clicked: Artist = None
        self.num_graphs = len(self.displays)
        self.length_data = self.displays[0].num_instances()
        self.sliders = []
        self.criteria = []
        self.ind = 0
        self.clicked = False

    def plot(
        self,
        chart: Union[Figure, Axes, None] = None,
        *,
        fig: Optional[Figure] = None,
        axes: Optional[Sequence[Axes]] = None,
        **kwargs: Any,
    ):
        fig, axes = self.init_axes(chart=chart, fig=fig, axes=axes)
        self.fig = fig
        self.axes = axes

        if self.num_graphs > 1:
            for d in self.displays[1:]:
                if d.num_instances() != self.length_data:
                    raise ValueError(
                        "Length of some data sets are not equal ",
                    )

        for disp, ax in zip(self.displays, self.axes):
            disp.plot(axes=ax)

        self.fig.canvas.mpl_connect('pick_event', self.pick)

        self.fig.suptitle("Multiple display")
        self.fig.tight_layout()

        for slider in self.sliders:
            slider.observe(self.value_updated, 'value')
            display(slider)

        return self.fig

    def add_displays(
        self,
        displays: Union[Display, Sequence[Display]],
    ) -> None:
        if isinstance(displays, Display):
            self.displays.append(displays)
        else:
            self.displays.extend(displays)

    def init_axes(
        self,
        chart: Union[Figure, Axes, None] = None,
        fig: Optional[Figure] = None,
        axes: Union[Axes, Sequence[Axes], None] = None,
    ) -> Figure:
        fig, axes = _get_figure_and_axes(chart, fig, axes)

        fig, axes = _set_figure_layout(
            fig=fig, axes=axes, n_axes=len(self.displays),
        )

        return fig, axes

    def pick(self, event: Event) -> None:
        if self.clicked:
            self.restore_points_intensity()
            self.clicked = False
        if self.point_clicked is None:
            self.point_clicked = event.artist
            self.picked_disp = self.get_display_picked()
            self.reduce_points_intensity()
        elif self.point_clicked == event.artist:
            self.restore_points_intensity()
            self.point_clicked = None
        else:
            self.change_points_intensity(new_point=event.artist)
            self.point_clicked = event.artist

    def get_display_picked(self) -> int:
        for i in range(self.num_graphs):
            if self.axes[i] == self.point_clicked.axes:
                return self.displays[i]

    def reduce_points_intensity(self) -> None:
        for i in range(self.length_data):
            if not (
                np.ma.getdata(
                    self.picked_disp.id_function[i].get_offsets(),
                )[0][0]
                == np.ma.getdata(self.point_clicked.get_offsets())[0][0]
                and np.ma.getdata(
                    self.picked_disp.id_function[i].get_offsets(),
                )[0][1]
                == np.ma.getdata(self.point_clicked.get_offsets())[0][1]
            ):
                for d in self.displays:
                    if isinstance(d.id_function[i], list):
                        d.id_function[i][0].set_alpha(0.1)
                    else:
                        d.id_function[i].set_alpha(0.1)

    def restore_points_intensity(self) -> None:
        for i in range(self.length_data):
            for d in self.displays:
                if isinstance(d.id_function[i], list):
                    d.id_function[i][0].set_alpha(1)
                else:
                    d.id_function[i].set_alpha(1)

    def change_points_intensity(
        self,
        new_point: Artist = None,
        index: Union[int, None] = None,
    ) -> None:
        if index is None and new_point is not None:
            for i in range(self.length_data):
                if (
                    np.ma.getdata(
                        self.picked_disp.id_function[i].get_offsets(),
                    )[0][0]
                    == np.ma.getdata(new_point.get_offsets())[0][0]
                ) and (
                    np.ma.getdata(
                        self.picked_disp.id_function[i].get_offsets(),
                    )[0][1]
                    == np.ma.getdata(new_point.get_offsets())[0][1]
                ):
                    intensity = 1
                else:
                    intensity = 0.1

                for d in self.displays:
                    if isinstance(d.id_function[i], list):
                        d.id_function[i][0].set_alpha(intensity)
                    else:
                        d.id_function[i].set_alpha(intensity)
        elif index is not None:
            for j in range(self.length_data):
                intensity = 1 if index == j else 0.1
                for disp in self.displays:
                    if isinstance(disp.id_function[j], list):
                        disp.id_function[j][0].set_alpha(intensity)
                    else:
                        disp.id_function[j].set_alpha(intensity)

    def add_slider(
        self,
        criterion: Sequence[float],
    ) -> None:
        if self.length_data == len(criterion):
            full_desc = "Filter (" + str(self.ind) + ")"
            self.sliders.append(widgets.IntSlider(
                value=0,
                min=0,
                max=self.length_data - 1,
                step=1,
                description=full_desc,
                disabled=False,
                continuous_update=False,
                orientation='horizontal',
                readout=True,
                readout_format='d',
            ))
            self.ind += 1

            dic = dict(zip(criterion, range(self.length_data)))
            order_dic = collections.OrderedDict(sorted(dic.items()))
            self.criteria.append(order_dic.values())
        else:
            raise ValueError(
                "Slider criteria should be of the same size as the data",
            )

    def value_updated(self, change):
        temp = change['owner'].description.split("(")[1]
        temp = temp.split(")")[0]
        index_criteria = int(temp)
        index_picked = list(self.criteria[index_criteria])[change['new']]
        self.clicked = True
        self.change_points_intensity(index=index_picked)
