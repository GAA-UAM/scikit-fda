import collections
import copy
from typing import List, Optional, Sequence, Union

from matplotlib.artist import Artist
from matplotlib.axes import Axes
from matplotlib.backend_bases import Event
from matplotlib.figure import Figure
from matplotlib.widgets import Slider, Widget

from ._baseplot import BasePlot
from ._utils import _get_figure_and_axes, _set_figure_layout


class MultipleDisplay:
    def __init__(
        self,
        displays: Union[BasePlot, List[BasePlot]],
        criteria: Union[
            Sequence[float],
            Sequence[Sequence[float]],
            None,
        ] = None,
        sliders: Union[Widget, Sequence[Widget], None] = None,
        label_sliders: Union[
            str,
            Sequence[str],
            None,
        ] = None,
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
        self.clicked = False
        self.index_clicked = -1
        self.tags = []
        self.previous_hovered = None
        self.fig = fig
        self.axes = axes
        self.chart = chart

        if criteria is not None and sliders is not None:
            if isinstance(sliders, collections.Iterable):
                if len(criteria) == len(sliders):
                    self.create_sliders(criteria, sliders, label_sliders)
                else:
                    raise ValueError(
                        "Size of criteria, and sliders should be equal.",
                    )
            else:
                self.create_sliders(criteria, sliders, label_sliders)
        else:
            self.init_axes(chart=self.chart, fig=self.fig, axes=self.axes)

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
            slider.on_changed(self.value_updated)

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
                    if isinstance(artist, List):
                        return
                    is_graph, ind = artist.contains(event)
                    if is_graph and self.previous_hovered == artist:
                        return
                    if is_graph:
                        self.previous_hovered = artist
                        break
                break

        for j in range(self.num_graphs, len(self.axes)):
            if event.inaxes == self.axes[j]:
                self.widget_index = j - self.num_graphs

        if index_axis != -1 and is_graph:
            self.update_annot(index_axis)
            self.tags[index_axis].set_visible(True)
            self.fig.canvas.draw_idle()
        elif self.tags[index_axis].get_visible():
            self.previous_hovered = None
            self.tags[index_axis].set_visible(False)
            self.fig.canvas.draw_idle()

    def init_axes(
        self,
        chart: Union[Figure, Axes, None] = None,
        fig: Optional[Figure] = None,
        axes: Union[Axes, Sequence[Axes], None] = None,
        extra: int = 0,
    ) -> None:
        fig, axes = _get_figure_and_axes(chart, fig, axes)
        fig, axes = _set_figure_layout(
            fig=fig, axes=axes, n_axes=len(self.displays) + extra,
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

    def create_sliders(
        self,
        criteria: Union[Sequence[float], Sequence[Sequence[float]]],
        sliders: Union[Widget, Sequence[Widget]],
        label_sliders: Union[str, Sequence[str], None] = None,
    ) -> None:
        if isinstance(criteria[0], collections.Iterable):
            for c in criteria:
                if len(c) != self.length_data:
                    raise ValueError(
                        "Slider criteria should be of the same size as data",
                    )

            self.init_axes(chart=self.chart, fig=self.fig, extra=len(criteria))

            if label_sliders is None:
                for i in range(len(criteria)):
                    self.__add_slider(i, criteria[i], sliders[i])
            elif isinstance(label_sliders, str):
                raise ValueError(
                    "Incorrect length of slider labels.",
                )
            elif len(label_sliders) == len(sliders):
                for i in range(len(criteria)):
                    self.__add_slider(
                        i,
                        criteria[i],
                        sliders[i],
                        label_sliders[i],
                    )
            else:
                raise ValueError(
                    "Incorrect length of slider labels.",
                )
        elif (
            len(criteria) == self.length_data
            and (isinstance(label_sliders, str) or label_sliders is None)
        ):
            self.init_axes(
                chart=self.chart,
                fig=self.fig,
                axes=self.axes,
                extra=1,
            )
            self.__add_slider(0, criteria, sliders, label_sliders)
        else:
            raise ValueError(
                "Slider criteria should be of the same size as data",
            )

    def __add_slider(
        self,
        ind_ax: int,
        criterion: Sequence[float],
        widget_func: Widget = Slider,
        label_slider: Optional[str] = None,
    ) -> None:
        if label_slider is None:
            full_desc = "".join(["Filter (", str(ind_ax), ")"])
        else:
            full_desc = label_slider
        self.sliders.append(
            widget_func(
                self.fig.axes[self.num_graphs + ind_ax],
                full_desc,
                valmin=0,
                valmax=self.length_data - 1,
                valinit=0,
            ),
        )

        dic = dict(zip(criterion, range(self.length_data)))
        order_dic = collections.OrderedDict(sorted(dic.items()))
        self.criteria.append(order_dic.values())

    def value_updated(self, value):
        # Make the changes of the slider discrete
        index = int(int(value / 0.5) * 0.5)
        old_index = self.index_clicked
        self.index_clicked = list(self.criteria[self.widget_index])[index]
        self.sliders[self.widget_index].valtext.set_text('{}'.format(index))
        self.clicked = True
        if old_index == -1:
            self.reduce_points_intensity()
        else:
            self.change_points_intensity(old_index=old_index)
