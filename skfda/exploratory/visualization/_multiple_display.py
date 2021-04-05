import collections
import copy
from typing import Any, List, Optional, Sequence, Union

import ipywidgets as widgets
from IPython.display import display
from matplotlib.artist import Artist
from matplotlib.axes import Axes
from matplotlib.backend_bases import Event
from matplotlib.figure import Figure

from ._baseplot import BasePlot
from ._utils import _get_figure_and_axes, _set_figure_layout


class MultipleDisplay:
    def __init__(
        self,
        displays: Union[BasePlot, List[BasePlot]],
        chart: Union[Figure, Axes, None] = None,
        fig: Optional[Figure] = None,
        axes: Optional[Sequence[Axes]] = None,
    ):
        if isinstance(displays, BasePlot):
            self.displays = [copy.copy(displays)]
        else:
            self.displays = []
            for d in displays:
                self.displays.append(copy.copy(d))
        self.point_clicked: Artist = None
        self.num_graphs = len(self.displays)
        self.length_data = self.displays[0].num_instances()
        self.sliders = []
        self.criteria = []
        self.ind = 0
        self.clicked = False
        self.index_clicked = -1
        self.tags = []
        self.previous_hovered = None
        self.init_axes(chart=chart, fig=fig, axes=axes)

    def plot(
        self,
    ):
        if self.num_graphs > 1:
            for d in self.displays[1:]:
                if d.num_instances() != self.length_data:
                    raise ValueError(
                        "Length of some data sets are not equal ",
                    )

        for disp, ax in zip(self.displays, self.axes):
            disp.set_figure_and_axes(axes=ax)
            disp.plot()
            self.tags.append(
                ax.annotate(
                    "",
                    xy=(0, 0),
                    xytext=(20, 20),
                    textcoords="offset points",
                    bbox=dict(boxstyle="round", fc="w"),
                    arrowprops=dict(arrowstyle="->"),
                ),
            )

        self.fig.canvas.mpl_connect('motion_notify_event', self.hover)
        self.fig.canvas.mpl_connect('pick_event', self.pick)

        for i in range(self.num_graphs):
            self.tags[i].set_visible(False)

        self.fig.suptitle("Multiple display")
        self.fig.tight_layout()

        for slider in self.sliders:
            slider.observe(self.value_updated, 'value')
            display(slider)

        return self.fig

    def update_annot(self, index):
        xdata_graph = self.previous_hovered.get_offsets()[0][0]
        ydata_graph = self.previous_hovered.get_offsets()[0][1]
        xdata_aprox = "{0:.2f}".format(xdata_graph)
        ydata_aprox = "{0:.2f}".format(ydata_graph)

        self.tags[index].xy = (xdata_graph, ydata_graph)
        text = "".join(["(", str(xdata_aprox), ", ", str(ydata_aprox), ")"])

        self.tags[index].set_text(text)
        self.tags[index].get_bbox_patch().set_facecolor(color='red')
        intensity = 0.4
        self.tags[index].get_bbox_patch().set_alpha(intensity)

    def hover(self, event: Event):
        index_axis = -1

        for i in range(self.num_graphs):
            if event.inaxes == self.axes[i]:
                index_axis = i
                for artist in self.displays[i].id_function:
                    is_graph, ind = artist.contains(event)
                    if is_graph and self.previous_hovered == artist:
                        return
                    if is_graph:
                        self.previous_hovered = artist
                        break
                break

        if index_axis != -1 and is_graph:
            self.update_annot(index_axis)
            self.tags[index_axis].set_visible(True)
            self.fig.canvas.draw_idle()
        elif self.tags[index_axis].get_visible():
            self.previous_hovered = None
            self.tags[index_axis].set_visible(False)
            self.fig.canvas.draw_idle()

    def add_displays(
        self,
        displays: Union[BasePlot, Sequence[BasePlot]],
    ) -> None:
        if isinstance(displays, BasePlot):
            self.displays.append(displays)
        else:
            self.displays.extend(displays)

    def init_axes(
        self,
        chart: Union[Figure, Axes, None] = None,
        fig: Optional[Figure] = None,
        axes: Union[Axes, Sequence[Axes], None] = None,
    ) -> None:
        fig, axes = _get_figure_and_axes(chart, fig, axes)
        fig, axes = _set_figure_layout(
            fig=fig, axes=axes, n_axes=len(self.displays),
        )

        self.fig = fig
        self.axes = axes

    def pick(self, event: Event) -> None:
        if self.clicked:
            self.point_clicked = event.artist
            self.change_points_intensity()
            self.clicked = False
        elif self.point_clicked is None:
            self.point_clicked = event.artist
            self.update_index_display_picked()
            self.reduce_points_intensity()
        elif self.point_clicked == event.artist:
            self.restore_points_intensity()
        else:
            self.point_clicked = event.artist
            self.change_points_intensity()

    def update_index_display_picked(self) -> int:
        for i in range(self.num_graphs):
            if self.axes[i] == self.point_clicked.axes:
                self.index_clicked = self.displays[i].id_function.index(
                    self.point_clicked,
                )
                return

    def reduce_points_intensity(self) -> None:
        for i in range(self.length_data):
            if i != self.index_clicked:
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
        self.point_clicked = None
        self.index_clicked = -1

    def change_points_intensity(
        self,
        old_index: Union[int, None] = None,
    ) -> None:
        if old_index is None:
            old_index = self.index_clicked
            self.update_index_display_picked()

        if self.index_clicked == old_index:
            self.restore_points_intensity()
            return

        for i in range(self.length_data):
            if i == self.index_clicked:
                intensity = 1
            elif i == old_index:
                intensity = 0.1
            else:
                intensity = -1

            if intensity != -1:
                self.change_display_intensity(i, intensity)

    def change_display_intensity(self, index: int, intensity: int) -> None:
        for d in self.displays:
            if isinstance(d.id_function[index], list):
                d.id_function[index][0].set_alpha(intensity)
            else:
                d.id_function[index].set_alpha(intensity)

    def add_slider(
        self,
        criterion: Sequence[float],
        widget_func: Any = widgets.BoundedIntText,
    ) -> None:
        if self.length_data == len(criterion):
            full_desc = "".join(["Filter (", str(self.ind), ")"])
            self.sliders.append(widget_func(
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
        old_index = self.index_clicked
        self.index_clicked = list(self.criteria[index_criteria])[change['new']]
        self.clicked = True
        if old_index == -1:
            self.reduce_points_intensity()
        else:
            self.change_points_intensity(old_index=old_index)
