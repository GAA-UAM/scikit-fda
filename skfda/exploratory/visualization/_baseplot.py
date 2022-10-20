"""BasePlot Module.

This module contains the abstract class of which inherit all
the visualization modules, containing the basic functionality
common to all of them.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Sequence, Tuple

import matplotlib.pyplot as plt
from matplotlib.artist import Artist
from matplotlib.axes import Axes
from matplotlib.backend_bases import LocationEvent, MouseEvent
from matplotlib.collections import PathCollection
from matplotlib.colors import ListedColormap
from matplotlib.figure import Figure
from matplotlib.text import Annotation

from ...representation import FData
from ...typing._numpy import NDArrayInt, NDArrayObject
from ._utils import _figure_to_svg, _get_figure_and_axes, _set_figure_layout


class BasePlot(ABC):
    """
    BasePlot class.

    Attributes:
        artists: List of Artist objects corresponding
            to every instance of our plot. They will be used to modify
            the visualization with interactivity and widgets.
        fig: Figure over with the graphs are plotted.
        axes: Sequence of axes where the graphs are plotted.
    """

    @abstractmethod
    def __init__(
        self,
        chart: Figure | Axes | None = None,
        *,
        fig: Figure | None = None,
        axes: Axes | Sequence[Axes] | None = None,
        n_rows: int | None = None,
        n_cols: int | None = None,
        c: NDArrayInt | None = None,
        cmap_bold: ListedColormap = None,
        x_label: str | None = None,
        y_label: str | None = None,
    ) -> None:
        self.artists: NDArrayObject | None = None
        self.chart = chart
        self.fig = fig
        self.axes = axes
        self.n_rows = n_rows
        self.n_cols = n_cols
        self._tag = self._create_annotation()
        self.c = c
        self.cmap_bold = cmap_bold
        self.x_label = x_label
        self.y_label = y_label

    def _plot(
        self,
        fig: Figure,
        axes: Sequence[Axes],
    ) -> None:
        pass

    def plot(
        self,
    ) -> Figure:
        """
        Plot the object and its data.

        Returns:
            Figure: figure object in which the displays and
                widgets will be plotted.
        """
        fig: Figure | None = getattr(self, "fig_", None)
        axes: Sequence[Axes] | None = getattr(self, "axes_", None)

        if fig is None:
            fig, axes = self._set_figure_and_axes(
                self.chart,
                fig=self.fig,
                axes=self.axes,
            )

        assert axes is not None

        if self.x_label is not None:
            axes[0].set_xlabel(self.x_label)
        if self.y_label is not None:
            axes[0].set_ylabel(self.y_label)

        self._plot(fig, axes)

        self._hover_event_id = fig.canvas.mpl_connect(
            'motion_notify_event',
            self.hover,
        )

        return fig

    @property
    def dim(self) -> int:
        """Get the number of dimensions for this plot."""
        return 2

    @property
    def n_subplots(self) -> int:
        """Get the number of subplots that this plot uses."""
        return 1

    @property
    def n_samples(self) -> int | None:
        """Get the number of instances that will be used for interactivity."""
        return None

    def _set_figure_and_axes(
        self,
        chart: Figure | Axes | None = None,
        *,
        fig: Figure | None = None,
        axes: Axes | Sequence[Axes] | None = None,
    ) -> Tuple[Figure, Sequence[Axes]]:
        fig, axes = _get_figure_and_axes(chart, fig, axes)
        fig, axes = _set_figure_layout(
            fig=fig,
            axes=axes,
            dim=self.dim,
            n_axes=self.n_subplots,
            n_rows=self.n_rows,
            n_cols=self.n_cols,
        )

        self.fig_ = fig
        self.axes_ = axes

        return fig, axes

    def _repr_svg_(self) -> str:
        """Automatically represents the object as an svg when calling it."""
        self.fig = self.plot()
        plt.close(self.fig)
        return _figure_to_svg(self.fig)

    def _create_annotation(self) -> Annotation:
        tag = Annotation(
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
            annotation_clip=False,
            clip_on=False,
        )

        tag.get_bbox_patch().set_facecolor(color='khaki')
        intensity = 0.8
        tag.get_bbox_patch().set_alpha(intensity)

        return tag

    def _update_annotation(
        self,
        tag: Annotation,
        *,
        axes: Axes,
        sample_number: int,
        fdata: FData | None,
        position: Tuple[float, float],
    ) -> None:
        """
        Auxiliary method used to update the hovering annotations.

        Method used to update the annotations that appear while
        hovering a scattered point. The annotations indicate
        the index and coordinates of the point hovered.
        Args:
            tag: Annotation to update.
            axes: Axes were the annotation belongs.
            sample_number: Number of the current sample.
        """
        xdata_graph, ydata_graph = position

        tag.xy = (xdata_graph, ydata_graph)

        sample_name = (
            fdata.sample_names[sample_number]
            if fdata is not None
            else None
        )

        sample_descr = f" ({sample_name})" if sample_name is not None else ""

        text = (
            f"{sample_number}{sample_descr}: "
            f"({xdata_graph:.3g}, {ydata_graph:.3g})"
        )
        tag.set_text(text)

        x_axis = axes.get_xlim()
        y_axis = axes.get_ylim()

        label_xpos = -60
        label_ypos = 20
        if (xdata_graph - x_axis[0]) > (x_axis[1] - xdata_graph):
            label_xpos = -80

        if (ydata_graph - y_axis[0]) > (y_axis[1] - ydata_graph):
            label_ypos = -20

        if tag.figure:
            tag.remove()
        tag.figure = None
        axes.add_artist(tag)
        tag.set_transform(axes.transData)
        tag.set_position((label_xpos, label_ypos))

    def _sample_artist_from_event(
        self,
        event: LocationEvent,
    ) -> Tuple[int, FData | None, Artist] | None:
        """Get the number, fdata and artist under a location event."""
        if self.artists is None:
            return None

        try:
            i = self.axes_.index(event.inaxes)
        except ValueError:
            return None

        for j, artist in enumerate(self.artists[:, i]):
            if not isinstance(artist, PathCollection):
                return None

            if artist.contains(event)[0]:
                return j, getattr(self, "fdata", None), artist

        return None

    def hover(self, event: MouseEvent) -> None:
        """
        Activate the annotation when hovering a point.

        Callback method that activates the annotation when hovering
        a specific point in a graph. The annotation is a description
        of the point containing its coordinates.

        Args:
            event: event object containing the artist of the point
                hovered.

        """
        found_artist = self._sample_artist_from_event(event)

        if event.inaxes is not None and found_artist is not None:
            sample_number, fdata, artist = found_artist

            self._update_annotation(
                self._tag,
                axes=event.inaxes,
                sample_number=sample_number,
                fdata=fdata,
                position=artist.get_offsets()[0],
            )
            self._tag.set_visible(True)
            self.fig_.canvas.draw_idle()
        elif self._tag.get_visible():
            self._tag.set_visible(False)
            self.fig_.canvas.draw_idle()
