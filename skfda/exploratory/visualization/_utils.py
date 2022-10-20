from __future__ import annotations

import io
import math
import re
from itertools import repeat
from typing import Sequence, Tuple, TypeVar, Union

import matplotlib.backends.backend_svg
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from typing_extensions import Protocol, TypeAlias

from ...representation._functional_data import FData

non_close_text = '[^>]*?'
svg_width_regex = re.compile(
    f'(<svg {non_close_text}width="){non_close_text}("{non_close_text}>)',
)
svg_width_replacement = r'\g<1>100%\g<2>'
svg_height_regex = re.compile(
    f'(<svg {non_close_text})height="{non_close_text}"({non_close_text}>)',
)
svg_height_replacement = r'\g<1>\g<2>'

ColorLike: TypeAlias = Union[
    Tuple[float, float, float],
    Tuple[float, float, float, float],
    str,
    Sequence[float],
]

K = TypeVar('K', contravariant=True)
V = TypeVar('V', covariant=True)


class Indexable(Protocol[K, V]):
    """Class Indexable used to type _get_color_info."""

    def __getitem__(self, __key: K) -> V:
        pass

    def __len__(self) -> int:
        pass


def _create_figure() -> Figure:
    """Create figure using the default backend."""
    return plt.figure()


def _figure_to_svg(figure: Figure) -> str:
    """Return the SVG representation of a figure."""
    old_canvas = figure.canvas
    matplotlib.backends.backend_svg.FigureCanvas(figure)
    output = io.BytesIO()
    figure.savefig(output, format='svg')
    figure.set_canvas(old_canvas)
    data = output.getvalue()
    decoded_data = data.decode('utf-8')

    new_data = svg_width_regex.sub(
        svg_width_replacement,
        decoded_data,
        count=1,
    )

    return svg_height_regex.sub(
        svg_height_replacement,
        new_data,
        count=1,
    )


def _get_figure_and_axes(
    chart: Figure | Axes | Sequence[Axes] | None = None,
    fig: Figure | None = None,
    axes: Axes | Sequence[Axes] | None = None,
) -> Tuple[Figure, Sequence[Axes]]:
    """Obtain the figure and axes from the arguments."""
    num_defined = sum(e is not None for e in (chart, fig, axes))
    if num_defined > 1:
        raise ValueError(
            "Only one of chart, fig and axes parameters"
            "can be passed as an argument.",
        )

    # Parse chart argument
    if chart is not None:
        if isinstance(chart, matplotlib.figure.Figure):
            fig = chart
        else:
            axes = chart

    if fig is None and axes is None:
        new_fig = _create_figure()
        new_axes = []

    elif fig is not None:
        new_fig = fig
        new_axes = fig.axes

    else:
        assert axes is not None
        if isinstance(axes, Axes):
            axes = [axes]

        new_fig = axes[0].figure
        new_axes = axes

    return new_fig, new_axes


def _get_axes_shape(
    n_axes: int,
    n_rows: int | None = None,
    n_cols: int | None = None,
) -> Tuple[int, int]:
    """Get the number of rows and columns of the subplots."""
    if (
        (n_rows is not None and n_cols is not None)
        and ((n_rows * n_cols) < n_axes)
    ):
        raise ValueError(
            f"The number of rows ({n_rows}) multiplied by "
            f"the number of columns ({n_cols}) "
            f"is less than the number of required "
            f"axes ({n_axes})",
        )

    if n_rows is None and n_cols is None:
        new_n_cols = int(math.ceil(math.sqrt(n_axes)))
        new_n_rows = int(math.ceil(n_axes / new_n_cols))
    elif n_rows is None and n_cols is not None:
        new_n_cols = n_cols
        new_n_rows = int(math.ceil(n_axes / n_cols))
    elif n_cols is None and n_rows is not None:
        new_n_cols = int(math.ceil(n_axes / n_rows))
        new_n_rows = n_rows

    return new_n_rows, new_n_cols


def _projection_from_dim(dim: int) -> str:

    if dim == 2:
        return 'rectilinear'
    elif dim == 3:
        return '3d'

    raise NotImplementedError(
        "Only bidimensional or tridimensional plots are supported.",
    )


def _set_figure_layout(
    fig: Figure,
    axes: Sequence[Axes],
    dim: int | Sequence[int] = 2,
    n_axes: int = 1,
    n_rows: int | None = None,
    n_cols: int | None = None,
) -> Tuple[Figure, Sequence[Axes]]:
    """
    Set the figure axes for plotting.

    Args:
        fig: Figure over with the graphs are plotted in case ax is not
            specified.
        axes: Axis over where the graphs are plotted.
        dim: Dimension of the plot. Either 2 for a 2D plot or 3 for a 3D plot.
        n_axes: Number of subplots.
        n_rows: Designates the number of rows of the figure to plot the
            different dimensions of the image. Can only be passed if no axes
            are specified.
        n_cols: Designates the number of columns of the figure to plot the
            different dimensions of the image. Can only be passed if no axes
            are specified.

    Returns:
        (tuple): tuple containing:

            * fig (figure): figure object in which the graphs are plotted.
            * axes (list): axes in which the graphs are plotted.

    """
    if len(axes) not in {0, n_axes}:
        raise ValueError(
            f"The number of axes ({len(axes)}) must be 0 (to create them)"
            f" or equal to the number of axes needed "
            f"({n_axes} in this case).",
        )

    if len(axes) != 0 and (n_rows is not None or n_cols is not None):
        raise ValueError(
            "The number of columns and/or number of rows of "
            "the figure, in which each dimension of the "
            "image is plotted, can only be customized in case "
            "that no axes are provided.",
        )

    if len(axes) == 0:
        # Create the axes

        n_rows, n_cols = _get_axes_shape(n_axes, n_rows, n_cols)

        for i in range(n_rows):
            for j in range(n_cols):
                subplot_index = i * n_cols + j
                if subplot_index < n_axes:
                    plot_dim = (
                        dim if isinstance(dim, int) else dim[subplot_index]
                    )

                fig.add_subplot(
                    n_rows,
                    n_cols,
                    subplot_index + 1,
                    projection=_projection_from_dim(plot_dim),
                )

        axes = fig.axes

    else:
        # Check that the projections are right
        projections = (
            repeat(_projection_from_dim(dim))
            if isinstance(dim, int)
            else (_projection_from_dim(d) for d in dim)
        )

        for a, proj in zip(axes, projections):
            if a.name != proj:
                raise ValueError(
                    f"The projection of the axes is {a.name} "
                    f"but should be {proj}",
                )

    return fig, axes


def _set_labels(
    fdata: FData,
    fig: Figure,
    axes: Sequence[Axes],
    patches: Sequence[matplotlib.patches.Patch] | None = None,
) -> None:
    """Set labels if any.

    Args:
        fdata: functional data object.
        fig: figure object containing the axes that implement
            set_xlabel and set_ylabel, and set_zlabel in case
            of a 3d projection.
        axes: axes objects that implement set_xlabel and set_ylabel,
            and set_zlabel in case of a 3d projection; used if
            fig is None.
        patches: objects used to generate each entry in the legend.

    """
    # Dataset name
    if fdata.dataset_name is not None:
        fig.suptitle(fdata.dataset_name)

    # Legend
    if patches is not None:
        fig.legend(handles=patches)
    elif patches is not None:
        axes[0].legend(handles=patches)

    assert len(axes) >= fdata.dim_codomain

    # Axis labels
    if axes[0].name == '3d':
        for i, a in enumerate(axes):
            if fdata.argument_names[0] is not None:
                a.set_xlabel(fdata.argument_names[0])
            if fdata.argument_names[1] is not None:
                a.set_ylabel(fdata.argument_names[1])
            if fdata.coordinate_names[i] is not None:
                a.set_zlabel(fdata.coordinate_names[i])
    else:
        for i in range(fdata.dim_codomain):
            if fdata.argument_names[0] is not None:
                axes[i].set_xlabel(fdata.argument_names[0])
            if fdata.coordinate_names[i] is not None:
                axes[i].set_ylabel(fdata.coordinate_names[i])


def _change_luminosity(color: ColorLike, amount: float = 0.5) -> ColorLike:
    """
    Change the given color luminosity by the given amount.

    Input can be matplotlib color string, hex string, or RGB tuple.

    Note:
        Based on https://stackoverflow.com/a/49601444/2455333
    """
    import colorsys

    import matplotlib.colors as mc
    try:
        c = mc.cnames[color]
    except TypeError:
        c = color
    c = colorsys.rgb_to_hls(*mc.to_rgb(c))

    intensity = (amount - 0.5) * 2
    up = intensity > 0
    intensity = abs(intensity)

    lightness = c[1]
    if up:
        new_lightness = lightness + intensity * (1 - lightness)
    else:
        new_lightness = lightness - intensity * lightness

    return colorsys.hls_to_rgb(c[0], new_lightness, c[2])


def _darken(color: ColorLike, amount: float = 0) -> ColorLike:
    return _change_luminosity(color, 0.5 - amount / 2)


def _lighten(color: ColorLike, amount: float = 0) -> ColorLike:
    return _change_luminosity(color, 0.5 + amount / 2)
