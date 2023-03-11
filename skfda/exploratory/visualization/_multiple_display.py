from __future__ import annotations

import copy
import itertools
from functools import partial
from typing import Generator, List, Sequence, Tuple, Type, cast

import numpy as np
from matplotlib.artist import Artist
from matplotlib.axes import Axes
from matplotlib.backend_bases import Event
from matplotlib.figure import Figure
from matplotlib.widgets import Slider, Widget

from ._baseplot import BasePlot
from ._utils import _get_axes_shape, _get_figure_and_axes, _set_figure_layout


def _set_val_noevents(widget: Widget, val: float) -> None:
    e = widget.eventson
    widget.eventson = False
    widget.set_val(val)
    widget.eventson = e


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
        length_data: Number of instances or curves of the different displays.
        clicked: Boolean indicating whether a point has being clicked.
        selected_sample: Index of the function selected with the interactive
            module or widgets.
    """

    def __init__(
        self,
        displays: BasePlot | Sequence[BasePlot],
        criteria: Sequence[float] | Sequence[Sequence[float]] = (),
        sliders: Type[Widget] | Sequence[Type[Widget]] = (),
        label_sliders: str | Sequence[str] | None = None,
        chart: Figure | Axes | None = None,
        fig: Figure | None = None,
        axes: Sequence[Axes] | None = None,
    ):
        if isinstance(displays, BasePlot):
            displays = (displays,)

        self.displays = [copy.copy(d) for d in displays]
        self._n_graphs = sum(d.n_subplots for d in self.displays)
        self.length_data = next(
            d.n_samples
            for d in self.displays
            if d.n_samples is not None
        )
        self.sliders: List[Widget] = []
        self.selected_sample: int | None = None

        if len(criteria) != 0 and not isinstance(criteria[0], Sequence):
            criteria = cast(Sequence[float], criteria)
            criteria = (criteria,)

        criteria = cast(Sequence[Sequence[float]], criteria)
        self.criteria = criteria

        if not isinstance(sliders, Sequence):
            sliders = (sliders,)

        if isinstance(label_sliders, str):
            label_sliders = (label_sliders,)

        if len(criteria) != len(sliders):
            raise ValueError(
                f"Size of criteria, and sliders should be equal "
                f"(have {len(criteria)} and {len(sliders)}).",
            )

        self._init_axes(
            chart,
            fig=fig,
            axes=axes,
            extra=len(criteria),
        )

        self._create_sliders(
            criteria=criteria,
            sliders=sliders,
            label_sliders=label_sliders,
        )

    def _init_axes(
        self,
        chart: Figure | Axes | None = None,
        *,
        fig: Figure | None = None,
        axes: Sequence[Axes] | None = None,
        extra: int = 0,
    ) -> None:
        """
        Initialize the axes and figure.

        Args:
            chart: Figure over with the graphs are plotted or axis over
                where the graphs are plotted. If None and ax is also
                None, the figure is initialized.
            fig: Figure over with the graphs are plotted in case ax is not
                specified. If None and ax is also None, the figure is
                initialized.
            axes: Axis where the graphs are plotted. If None, see param fig.
            extra: integer indicating the extra axes needed due to the
                necessity for them to plot the sliders.

        """
        widget_aspect = 1 / 8
        fig, axes = _get_figure_and_axes(chart, fig, axes)
        if len(axes) not in {0, self._n_graphs + extra}:
            raise ValueError("Invalid number of axes.")

        n_rows, n_cols = _get_axes_shape(self._n_graphs + extra)

        dim = list(
            itertools.chain.from_iterable(
                [d.dim] * d.n_subplots
                for d in self.displays
            ),
        ) + [2] * extra

        number_axes = n_rows * n_cols
        fig, axes = _set_figure_layout(
            fig=fig,
            axes=axes,
            n_axes=self._n_graphs + extra,
            dim=dim,
        )

        for i in range(self._n_graphs, number_axes):
            if i >= self._n_graphs + extra:
                axes[i].set_visible(False)
            else:
                axes[i].set_box_aspect(widget_aspect)

        self.fig = fig
        self.axes = axes

    def _create_sliders(
        self,
        *,
        criteria: Sequence[Sequence[float]],
        sliders: Sequence[Type[Widget]],
        label_sliders: Sequence[str] | None = None,
    ) -> None:
        """
        Create the sliders with the criteria selected.

        Args:
            criteria: Different criterion for each of the sliders.
            sliders: Widget types.
            label_sliders: Sequence of the names of each slider.

        """
        for c in criteria:
            if len(c) != self.length_data:
                raise ValueError(
                    "Slider criteria should be of the same size as data",
                )

        for k, criterion in enumerate(criteria):
            label = label_sliders[k] if label_sliders else None

            self.add_slider(
                axes=self.axes[self._n_graphs + k],
                criterion=criterion,
                widget_class=sliders[k],
                label=label,
            )

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
        if self._n_graphs > 1:
            for d in self.displays[1:]:
                if (
                    d.n_samples is not None
                    and d.n_samples != self.length_data
                ):
                    raise ValueError(
                        "Length of some data sets are not equal ",
                    )

        for ax in self.axes[:self._n_graphs]:
            ax.clear()

        int_index = 0
        for disp in self.displays:
            axes_needed = disp.n_subplots
            end_index = axes_needed + int_index
            disp._set_figure_and_axes(axes=self.axes[int_index:end_index])
            disp.plot()
            int_index = end_index

        self.fig.canvas.mpl_connect('pick_event', self.pick)

        self.fig.suptitle("Multiple display")
        self.fig.tight_layout()

        return self.fig

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
        selected_sample = self._sample_from_artist(event.artist)

        if selected_sample is not None:
            if self.selected_sample == selected_sample:
                self._deselect_samples()
            else:
                self._select_sample(selected_sample)

    def _sample_from_artist(self, artist: Artist) -> int | None:
        """Return the sample corresponding to an artist."""
        for d in self.displays:

            if d.artists is None:
                continue

            for i, a in enumerate(d.axes_):
                if a == artist.axes:
                    if len(d.axes_) == 1:
                        return np.where(  # type: ignore[no-any-return]
                            d.artists == artist,
                        )[0][0]
                    else:
                        return np.where(  # type: ignore[no-any-return]
                            d.artists[:, i] == artist,
                        )[0][0]

        return None

    def _visit_artists(self) -> Generator[Tuple[int, Artist], None, None]:
        for i in range(self.length_data):
            for d in self.displays:
                if d.artists is None:
                    continue

                yield from ((i, artist) for artist in np.ravel(d.artists[i]))

    def _select_sample(self, selected_sample: int) -> None:
        """Reduce the transparency of all the points but the selected one."""
        for i, artist in self._visit_artists():
            artist.set_alpha(1.0 if i == selected_sample else 0.1)

        for criterion, slider in zip(self.criteria, self.sliders):
            val_widget = criterion[selected_sample]
            _set_val_noevents(slider, val_widget)

        self.selected_sample = selected_sample
        self.fig.canvas.draw_idle()

    def _deselect_samples(self) -> None:
        """Restore the original transparency of all the points."""
        for _, artist in self._visit_artists():
            artist.set_alpha(1)

        self.selected_sample = None
        self.fig.canvas.draw_idle()

    def add_slider(
        self,
        axes: Axes,
        criterion: Sequence[float],
        widget_class: Type[Widget] = Slider,
        label: str | None = None,
    ) -> None:
        """
        Add the slider to the MultipleDisplay object.

        Args:
            axes: Axes for the widget.
            criterion: Criterion used for the slider.
            widget_class: Widget type.
            label: Name of the slider.
        """
        full_desc = "" if label is None else label

        ordered_criterion_values, ordered_criterion_indexes = zip(
            *sorted(zip(criterion, range(self.length_data))),
        )

        widget = widget_class(
            ax=axes,
            label=full_desc,
            valmin=ordered_criterion_values[0],
            valmax=ordered_criterion_values[-1],
            valinit=ordered_criterion_values[0],
            valstep=ordered_criterion_values,
            valfmt="%.3g",
        )

        self.sliders.append(widget)

        axes.annotate(
            f"{ordered_criterion_values[0]:.3g}",
            xy=(0, -0.5),
            xycoords='axes fraction',
            annotation_clip=False,
        )

        axes.annotate(
            f"{ordered_criterion_values[-1]:.3g}",
            xy=(0.95, -0.5),
            xycoords='axes fraction',
            annotation_clip=False,
        )

        on_changed_function = partial(
            self._value_updated,
            ordered_criterion_values=ordered_criterion_values,
            ordered_criterion_indexes=ordered_criterion_indexes,
        )

        widget.on_changed(on_changed_function)

    def _value_updated(
        self,
        value: float,
        ordered_criterion_values: Sequence[float],
        ordered_criterion_indexes: Sequence[int],
    ) -> None:
        """
        Update the graphs when a widget is clicked.

        Args:
            value: Current value of the widget.
            ordered_criterion_values: Ordered values of the criterion.
            ordered_criterion_indexes: Sample numbers ordered using the
                criterion.

        """
        value_index = int(np.searchsorted(ordered_criterion_values, value))
        self.selected_sample = ordered_criterion_indexes[value_index]
        self._select_sample(self.selected_sample)
