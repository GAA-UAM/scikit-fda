import collections
import copy
from typing import List, Optional, Sequence, Union

import numpy as np
from matplotlib.artist import Artist
from matplotlib.axes import Axes
from matplotlib.backend_bases import Event
from matplotlib.collections import PathCollection
from matplotlib.figure import Figure
from matplotlib.text import Annotation
from matplotlib.widgets import Slider, Widget

from ._baseplot import BasePlot
from ._utils import _get_axes_shape, _get_figure_and_axes, _set_figure_layout


class MultipleDisplay:
    """
    MultipleDisplay class used to combine and interact with plots.

    This module is used to combine different BasePlot objects that
    represent the same curves or surfaces, and represent them
    together in the same figure. Besides this, it includes
    the functionality necessary to interact with the graphics
    by clicking the points, hovering over them... Picking the points allow
    us to see our selected function standing out among the others in all
    the axes. It is also possible to add widgets to interact with the
    plots.
    Args:
        displays: Baseplot objects that will be plotted in the fig.
        criteria: Sequence of criteria used to order the points in the
            slider widget. The size should be equal to sliders, as each
            criterion is for one slider.
        sliders: Sequence of widgets that will be plotted.
        label_sliders: Label of each of the sliders.
        chart: Figure over with the graphs are plotted or axis over
            where the graphs are plotted. If None and ax is also
            None, the figure is initialized.
        fig: Figure over with the graphs are plotted in case ax is not
            specified. If None and ax is also None, the figure is
            initialized.
        axes: Axis where the graphs are plotted. If None, see param fig.
    Attributes:
        point_clicked: Artist object containing the last point clicked.
        num_graphs: Number of graphs that will be plotted.
        length_data: Number of instances or curves of the different displays.
        clicked: Boolean indicating whether a point has being clicked.
        index_clicked: Index of the function selected with the interactive
            module or widgets.
        tags: List of tags for each ax, that contain the information printedº
            while hovering.
        previous_hovered: Artist object containing of the last point hovered.
        is_updating: Boolean value that determines whether a widget
            is being updated.
    """

    def __init__(
        self,
        displays: Union[BasePlot, Sequence[BasePlot]],
        criteria: Union[
            Sequence[float],
            Sequence[Sequence[float]],
            None,
        ] = None,
        sliders: Union[Widget, Sequence[Widget], None] = None,
        label_sliders: Union[str, Sequence[str], None] = None,
        chart: Union[Figure, Axes, None] = None,
        fig: Optional[Figure] = None,
        axes: Optional[Sequence[Axes]] = None,
    ):
        if isinstance(displays, BasePlot):
            displays = (displays,)

        self.displays = [copy.copy(d) for d in displays]
        self.point_clicked: Artist = None
        self.num_graphs = sum(len(d.axes) for d in self.displays)
        self.length_data = self.displays[0].n_samples()
        self.sliders: List[Widget] = []
        self.criteria: List[List[int]] = []
        self.clicked = False
        self.index_clicked = -1
        self.tags: List[Annotation] = []
        self.previous_hovered = None
        self.is_updating = False

        if criteria is not None and sliders is not None:
            if isinstance(sliders, Sequence):
                if len(criteria) != len(sliders):
                    raise ValueError(
                        "Size of criteria, and sliders should be equal.",
                    )

            self.create_sliders(
                chart,
                fig=fig,
                axes=axes,
                criteria=criteria,
                sliders=sliders,
                label_sliders=label_sliders,
            )

        else:
            self.init_axes(chart, fig=fig, axes=axes)

    def plot(self) -> Figure:
        """
        Plot Multiple Display method.

        Plot the different BasePlot objects and widgets selected.
        Activates the interactivity functionality of clicking and
        hovering points. When clicking a point, the rest will be
        made partially transparent in all the corresponding graphs.
        Returns:
            fig: figure object in which the displays and
                widgets will be plotted.
        """
        if self.num_graphs > 1:
            for d in self.displays[1:]:
                if d.n_samples() != self.length_data:
                    raise ValueError(
                        "Length of some data sets are not equal ",
                    )

        for ax in self.axes:
            ax.clear()
            self.tags.append(
                ax.annotate(
                    "",
                    xy=(0, 0),
                    xytext=(20, 20),
                    textcoords="offset points",
                    bbox={
                        "boxstyle": "round",
                        "fc": "w",
                    },
                    arrowprops={
                        "arrowstyle": "->",
                    },
                ),
            )

        int_index = 0
        for disp in self.displays:
            axes_needed = len(disp.axes)
            end_index = axes_needed + int_index
            disp._set_figure_and_axes(axes=self.axes[int_index:end_index])
            disp.plot()
            int_index = end_index

        self.fig.canvas.mpl_connect('motion_notify_event', self.hover)
        self.fig.canvas.mpl_connect('pick_event', self.pick)

        for i in range(self.num_graphs):
            self.tags[i].set_visible(False)

        self.fig.suptitle("Multiple display")
        self.fig.tight_layout()

        for slider in self.sliders:
            slider.on_changed(self.value_updated)

        return self.fig

    def update_annot(self, index_ax: int, index_point: int) -> None:
        """
        Auxiliary method used to update the hovering annotations.

        Method used to update the annotations that appear while
        hovering a scattered point. The annotations indicate
        the index and coordinates of the point hovered.
        Args:
            index_ax: index of the ax being hovered.
            index_point: index of the point being hovered.
        """
        xdata_graph, ydata_graph = self.previous_hovered.get_offsets()[0]

        current_tag = self.tags[index_ax]
        current_tag.xy = (xdata_graph, ydata_graph)
        current_tag.xy = (xdata_graph, ydata_graph)
        text = f"{index_point}: ({xdata_graph:.2f}, {ydata_graph:.2f})"

        x_axis = self.axes[index_ax].get_xlim()
        self.x_axis = x_axis
        self.xdata_graph = xdata_graph
        if (xdata_graph - x_axis[0]) > (x_axis[1] - xdata_graph):
            current_tag.set_position((-80, 20))
        else:
            current_tag.set_position((20, 20))

        current_tag.set_text(text)
        current_tag.get_bbox_patch().set_facecolor(color='red')
        intensity = 0.4
        current_tag.get_bbox_patch().set_alpha(intensity)

    def hover(self, event: Event) -> None:
        """
        Activate the annotation when hovering a point.

        Callback method that activates the annotation when hovering
        a specific point in a graph. The annotation is a description
        of the point containing its coordinates.
        Args:
            event: event object containing the artist of the point
                hovered.
        """
        index_axis = -1

        index = 0
        for d in self.displays:
            for i, ax in enumerate(d.axes):
                if event.inaxes == ax:
                    index_axis = index

                    artists_array = d.artists[:, i]
                    for j, artist in enumerate(artists_array):
                        if not isinstance(artist, PathCollection):
                            return

                        is_graph, ind = artist.contains(event)
                        if is_graph and self.previous_hovered == artist:
                            return
                        if is_graph:
                            self.previous_hovered = artist
                            index_point = j
                            break
                    break

                else:
                    index += 1

        for k in range(self.num_graphs, len(self.axes)):
            if event.inaxes == self.axes[k]:
                self.widget_index = k - self.num_graphs

        if index_axis != -1 and is_graph:
            self.update_annot(index_axis, index_point)
            self.tags[index_axis].set_visible(True)
            self.fig.canvas.draw_idle()
        elif self.tags[index_axis].get_visible():
            self.previous_hovered = None
            self.tags[index_axis].set_visible(False)
            self.fig.canvas.draw_idle()

    def init_axes(
        self,
        chart: Union[Figure, Axes, None] = None,
        *,
        fig: Optional[Figure] = None,
        axes: Optional[Sequence[Axes]] = None,
        extra: int = 0,
    ) -> None:
        """
        Initialize the axes and figure.

        Args:
            extra: integer indicating the extra axes needed due to the
                necessity for them to plot the sliders.
        """
        widget_aspect = 1 / 4
        fig, axes = _get_figure_and_axes(chart, fig, axes)
        if len(axes) not in {0, self.num_graphs + extra}:
            raise ValueError("Invalid number of axes.")

        n_rows, n_cols = _get_axes_shape(self.num_graphs + extra)

        number_axes = n_rows * n_cols
        fig, axes = _set_figure_layout(
            fig=fig, axes=axes, n_axes=self.num_graphs + extra,
        )

        for i in range(self.num_graphs, number_axes):
            if i >= self.num_graphs + extra:
                axes[i].set_visible(False)
            else:
                axes[i].set_box_aspect(widget_aspect)

        self.fig = fig
        self.axes = axes

    def pick(self, event: Event) -> None:
        """
        Activate interactive functionality when picking a point.

        Callback method that is activated when a point is picked.
        If no point was clicked previously, all the points but the
        one selected will be more transparent in all the graphs.
        If a point was clicked already, this new point will be the
        one highlighted among the rest. If the same point is clicked,
        the initial state of the graphics is restored.
        Args:
            event: event object containing the artist of the point
                picked.
        """
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

    def update_index_display_picked(self) -> None:
        """Update the index corresponding to the display picked."""
        for d in self.displays:
            for i, a in enumerate(d.axes):
                if a == self.point_clicked.axes:
                    if len(d.axes) == 1:
                        self.index_clicked = np.where(
                            d.artists == self.point_clicked,
                        )[0][0]
                    else:
                        self.index_clicked = np.where(
                            d.artists[:, i] == self.point_clicked,
                        )[0][0]
                    return

    def reduce_points_intensity(self) -> None:
        """Reduce the transparency of all the points but the selected one."""
        for i in range(self.length_data):
            if i != self.index_clicked:
                for d in self.displays:
                    for artist in np.ravel(d.artists[i]):
                        artist.set_alpha(0.1)

        self.is_updating = True
        for criterium, slider in zip(self.criteria, self.sliders):
            val_widget = list(criterium).index(self.index_clicked)
            slider.set_val(val_widget)
        self.is_updating = False

    def restore_points_intensity(self) -> None:
        """Restore the original transparency of all the points."""
        for i in range(self.length_data):
            for d in self.displays:
                for artist in np.ravel(d.artists[i]):
                    artist.set_alpha(1)

        self.point_clicked = None
        self.index_clicked = -1

        self.is_updating = True
        for j in range(len(self.sliders)):
            self.sliders[j].set_val(0)
        self.is_updating = False

    def change_points_intensity(
        self,
        old_index: Union[int, None] = None,
    ) -> None:
        """
        Change the intensity of the points.

        Changes the intensity of the points, the highlighted one now
        will be the selected one and the one with old_index with have
        its transparency increased.
        Args:
            old_index: index of the last point clicked, as it should
                reduce its transparency.
        """
        if old_index is None:
            old_index = self.index_clicked
            self.update_index_display_picked()

        if self.index_clicked == old_index:
            self.restore_points_intensity()
            return

        for i in range(self.length_data):
            if i == self.index_clicked:
                intensity = 1.0
            elif i == old_index:
                intensity = 0.1
            else:
                intensity = -1

            if intensity != -1:
                self.change_display_intensity(i, intensity)

        self.is_updating = True
        for criterium, slider in zip(self.criteria, self.sliders):
            val_widget = list(criterium).index(self.index_clicked)
            slider.set_val(val_widget)
        self.is_updating = False

    def change_display_intensity(self, index: int, intensity: float) -> None:
        """
        Change the intensity of the point selected by index in every display.

        Args:
            index: index of the last point clicked, as it should
                reduce its transparency.
            intensity: new intensity of the points.
        """
        for d in self.displays:
            if len(d.artists) != 0:
                for artist in np.ravel(d.artists[index]):
                    artist.set_alpha(intensity)

    def create_sliders(
        self,
        chart: Union[Figure, Axes, None] = None,
        *,
        fig: Optional[Figure] = None,
        axes: Optional[Sequence[Axes]] = None,
        criteria: Union[Sequence[float], Sequence[Sequence[float]]],
        sliders: Union[Widget, Sequence[Widget]],
        label_sliders: Union[str, Sequence[str], None] = None,
    ) -> None:
        """
        Create the sliders with the criteria selected.

        Args:
            criteria: different criterion for each of the sliders.
            sliders: widget types.
            label_sliders: sequence of the names of each slider.
        """
        if isinstance(criteria[0], Sequence):
            for c in criteria:
                if len(c) != self.length_data:
                    raise ValueError(
                        "Slider criteria should be of the same size as data",
                    )

            self.init_axes(
                chart,
                fig=fig,
                axes=axes,
                extra=len(criteria),
            )

            if label_sliders is None:
                for i in range(len(criteria)):
                    self.add_slider(i, criteria[i], sliders[i])
            elif isinstance(label_sliders, str):
                raise ValueError(
                    "Incorrect length of slider labels.",
                )
            elif len(label_sliders) == len(sliders):
                for k, criterium in enumerate(criteria):
                    self.add_slider(
                        k,
                        criterium,
                        sliders[k],
                        label_sliders[k],
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
                chart,
                fig=fig,
                axes=axes,
                extra=1,
            )
            self.add_slider(0, criteria, sliders, label_sliders)
        else:
            raise ValueError(
                "Slider criteria should be of the same size as data",
            )

    def add_slider(
        self,
        ind_ax: int,
        criterion: Sequence[float],
        widget_func: Widget = Slider,
        label_slider: Optional[str] = None,
    ) -> None:
        """
        Add the slider to the MultipleDisplay object.

        Args:
            ind_ax: index of the selected ax for the widget.
            criterion: criterion used for the slider.
            widget_func: widget type.
            label_slider: names of the slider.
        """
        if label_slider is None:
            full_desc = "".join(["Filter (", str(ind_ax), ")"])
        else:
            full_desc = label_slider
        self.sliders.append(
            widget_func(
                self.axes[self.num_graphs + ind_ax],
                full_desc,
                valmin=0,
                valmax=self.length_data - 1,
                valinit=0,
            ),
        )

        self.axes[self.num_graphs + ind_ax].annotate(
            '0',
            xy=(0, -0.5),
            xycoords='axes fraction',
            annotation_clip=False,
        )
        self.axes[self.num_graphs + ind_ax].annotate(
            str(self.length_data - 1),
            xy=(0.95, -0.5),
            xycoords='axes fraction',
            annotation_clip=False,
        )

        dic = dict(zip(criterion, range(self.length_data)))
        order_dic = collections.OrderedDict(sorted(dic.items()))
        self.criteria.append(order_dic.values())

    def value_updated(self, value: int) -> None:
        """
        Update the graphs when a widget is clicked.

        Args:
            value: current value of the widget.
        """
        # Used to avoid entering in an etern loop
        if self.is_updating is True:
            return
        self.is_updating = True

        # Make the changes of the slider discrete
        index = int(int(value / 0.5) * 0.5)
        old_index = self.index_clicked
        self.index_clicked = list(self.criteria[self.widget_index])[index]
        self.sliders[self.widget_index].valtext.set_text(f'{index}')

        # Update the other sliders values
        for i, (c, s) in enumerate(zip(self.criteria, self.sliders)):
            if i != self.widget_index:
                val_widget = list(c).index(self.index_clicked)
                s.set_val(val_widget)

        self.is_updating = False

        self.clicked = True
        if old_index == -1:
            self.reduce_points_intensity()
        else:
            if self.index_clicked == old_index:
                self.clicked = False
            self.change_points_intensity(old_index=old_index)
