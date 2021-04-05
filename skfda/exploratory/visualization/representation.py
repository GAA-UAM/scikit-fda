"""Representation Module.

This module contains the functionality related
with plotting and scattering our different datasets.
It allows multiple modes and colors, which could
be set manually or automatically depending on values
like depth measures.
"""

from typing import (
    Any,
    List,
    Mapping,
    Protocol,
    Optional,
    Sequence,
    Tuple,
    TypeVar,
    Union,
)

import matplotlib.cm
import matplotlib.patches
import numpy as np
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from ... import FDataGrid
from ..._utils import _to_domain_range, constants
from ...representation._functional_data import FData
from ...representation._typing import DomainRangeLike
from ._utils import (
    _get_figure_and_axes,
    _set_figure_layout_for_fdata,
    _set_labels,
)

K = TypeVar('K', contravariant=True)
V = TypeVar('V', covariant=True)
T = TypeVar('T', FDataGrid, np.ndarray)


class Indexable(Protocol[K, V]):
    def __getitem__(self, __key: K) -> V:
        pass

def _get_label_colors(
    n_labels: int,
    group_colors: Union[Sequence[Any], Mapping[K, Any], None],
) -> np.ndarray:
    """Get the colors of each label."""
    if group_colors is not None:
        if len(group_colors) != n_labels:
            raise ValueError(
                "There must be a color in group_colors "
                "for each of the labels that appear in "
                "group.",
            )
    else:
        colormap = matplotlib.cm.get_cmap()
        group_colors = colormap(np.arange(n_labels) / (n_labels - 1))

    return group_colors

def _get_color_info(
    fdata: T,
    group: Optional[Sequence[K]],
    group_names: Optional[Indexable[K, str]],
    group_colors: Optional[Indexable[K, Any]],
    legend: bool,
    kwargs: Any,
) -> Tuple[Any, Optional[List[matplotlib.patches.Patch]]]:

    patches = None

    if group is not None:
        # In this case, each curve has a label, and all curves with the same
        # label should have the same color

        group_unique, group_indexes = np.unique(group, return_inverse=True)
        n_labels = len(group_unique)

        if group_colors is not None:
            group_colors_array = np.array(
                [group_colors[g] for g in group_unique],
            )
        else:
            prop_cycle = matplotlib.rcParams['axes.prop_cycle']
            cycle_colors = prop_cycle.by_key()['color']

            group_colors_array = np.take(
                cycle_colors, np.arange(n_labels), mode='wrap',
            )

        sample_colors = group_colors_array[group_indexes]

        group_names_array = None

        if group_names is not None:
            group_names_array = np.array(
                [group_names[g] for g in group_unique],
            )
        elif legend is True:
            group_names_array = group_unique

        if group_names_array is not None:
            patches = [
                matplotlib.patches.Patch(color=c, label=l)
                for c, l in zip(group_colors_array, group_names_array)
            ]

    else:
        # In this case, each curve has a different color unless specified
        # otherwise

        if 'color' in kwargs:
            sample_colors = fdata.n_samples * [kwargs.get("color")]
            kwargs.pop('color')

        elif 'c' in kwargs:
            sample_colors = fdata.n_samples * [kwargs.get("c")]
            kwargs.pop('c')

        else:
            sample_colors = None

    return sample_colors, patches


class GraphPlot:
    """
    Class used to plot the FDataGrid object graph as hypersurfaces.

    When plotting functional data, we can either choose manually a color,
    a group of colors for the representations. Besides, we can use a list of
    variables (depths, scalar regression targets...) can be used as an
    argument to display the functions wtih a gradient of colors.

    Args:
        fdata: functional data set that we want to plot.
        gradient_color_list: list of real values used to determine the color
            in which each of the instances will be plotted.
        max_grad: maximum value that the gradient_list can take, it will be
            used to normalize the ``gradient_color_list``. If not
            declared it will be initialized to the maximum value of
            gradient_list
        min_grad: minimum value that the gradient_list can take, it will be
            used to normalize the ``gradient_color_list``. If not
            declared it will be initialized to the minimum value of
            gradient_list.

    """

    def __init__(
        self,
        fdata: FData,
        gradient_color_list: Optional[Sequence[float]] = None,
        max_grad: Optional[float] = None,
        min_grad: Optional[float] = None,
    ) -> None:
        self.fdata = fdata
        self.gradient_color_list = gradient_color_list
        if self.gradient_color_list is not None:
            if len(self.gradient_color_list) != fdata.n_samples:
                raise ValueError(
                    "The length of the gradient color"
                    "list should be the same as the number"
                    "of samples in fdata",
                )

            if min_grad is None:
                self.min_grad = min(self.gradient_color_list)
            else:
                self.min_grad = min_grad

            if max_grad is None:
                self.max_grad = max(self.gradient_color_list)
            else:
                self.max_grad = max_grad

            aux_list = [
                grad_color - self.min_grad
                for grad_color in self.gradient_color_list
            ]

            self.gradient_list: Sequence[float] = (
                [
                    aux / (self.max_grad - self.min_grad)
                    for aux in aux_list
                ]
            )
        else:
            self.gradient_list = []

    def plot(
        self,
        chart: Union[Figure, Axes, None] = None,
        *,
        fig: Optional[Figure] = None,
        ax: Optional[Axes] = None,
        n_rows: Optional[int] = None,
        n_cols: Optional[int] = None,
        n_points: Union[int, Tuple[int, int], None] = None,
        domain_range: Optional[DomainRangeLike] = None,
        group: Union[Sequence[Any], None] = None,
        group_colors: Union[Sequence[Any], Mapping[K, Any], None] = None,
        group_names: Union[Sequence[str], Mapping[K, str], None] = None,
        colormap_name: str = 'autumn',
        legend: bool = False,
        **kwargs: Any,
    ) -> Figure:
        """
        Plot the graph.

        Plots each coordinate separately. If the :term:`domain` is one
        dimensional, the plots will be curves, and if it is two
        dimensional, they will be surfaces. There are two styles of
        visualizations, one that displays the functions without any
        criteria choosing the colors and a new one that displays the
        function with a gradient of colors depending on the initial
        gradient_color_list (normalized in gradient_list).

        Args:
            chart: figure over
                with the graphs are plotted or axis over where the graphs are
                plotted. If None and ax is also None, the figure is
                initialized.
            fig : figure over with the graphs are
                plotted in case ax is not specified. If None and ax is also
                None, the figure is initialized.
            ax: axis over where the graphs are plotted. If None, see param fig.
            n_rows : designates the number of rows of the figure
                to plot the different dimensions of the image. Only specified
                if fig and ax are None.
            n_cols: designates the number of columns of the
                figure to plot the different dimensions of the image. Only
                specified if fig and ax are None.
            n_points: Number of points to evaluate in
                the plot. In case of surfaces a tuple of length 2 can be pased
                with the number of points to plot in each axis, otherwise the
                same number of points will be used in the two axes. By default
                in unidimensional plots will be used 501 points; in surfaces
                will be used 30 points per axis, wich makes a grid with 900
                points.
            domain_range: Range where the
                function will be plotted. In objects with unidimensional domain
                the domain range should be a tuple with the bounds of the
                interval; in the case of surfaces a list with 2 tuples with
                the ranges for each dimension. Default uses the domain range
                of the functional object.
            group: contains integers from [0 to number of
                labels) indicating to which group each sample belongs to. Then,
                the samples with the same label are plotted in the same color.
                If None, the default value, each sample is plotted in the color
                assigned by matplotlib.pyplot.rcParams['axes.prop_cycle'].
            group_colors: colors in which groups are
                represented, there must be one for each group. If None, each
                group is shown with distict colors in the "Greys" colormap.
            group_names: name of each of the groups which appear
                in a legend, there must be one for each one. Defaults to None
                and the legend is not shown. Implies `legend=True`.
            colormap_name: name of the colormap to be used. By default we will
                use autumn.
            legend: if `True`, show a legend with the groups. If
                `group_names` is passed, it will be used for finding the names
                to display in the legend. Otherwise, the values passed to
                `group` will be used.
            kwargs: if dim_domain is 1, keyword arguments to be passed to
                the matplotlib.pyplot.plot function; if dim_domain is 2,
                keyword arguments to be passed to the
                matplotlib.pyplot.plot_surface function.

        Returns:
            fig: the figure in which the graphs are plotted.

        """
        fig, axes = _get_figure_and_axes(chart, fig, ax)
        fig, axes = _set_figure_layout_for_fdata(
            self.fdata, fig, axes, n_rows, n_cols,
        )

        if domain_range is None:
            self.domain_range = self.fdata.domain_range
        else:
            self.domain_range = _to_domain_range(domain_range)

        if len(self.gradient_list) == 0:
            sample_colors, patches = _get_color_info(
                self.fdata, group, group_names, group_colors, legend, kwargs,
            )
        else:
            patches = None
            colormap = matplotlib.cm.get_cmap(colormap_name)
            colormap = colormap.reversed()

            sample_colors = [None] * self.fdata.n_samples
            for i in range(self.fdata.n_samples):
                sample_colors[i] = colormap(self.gradient_list[i])
        
        self.sample_colors = sample_colors

        color_dict: Mapping[str, Any] = {}

        if self.fdata.dim_domain == 1:

            if n_points is None:
                self.n_points = constants.N_POINTS_UNIDIMENSIONAL_PLOT_MESH

            # Evaluates the object in a linspace
            eval_points = np.linspace(*self.domain_range[0], self.n_points)
            mat = self.fdata(eval_points)

            for i in range(self.fdata.dim_codomain):
                for j in range(self.fdata.n_samples):

                    set_color_dict(sample_colors, j, color_dict)

                    axes[i].plot(
                        eval_points, mat[j, ..., i].T, **color_dict, **kwargs,
                    )

        else:

            # Selects the number of points
            if n_points is None:
                n_points_tuple = 2 * (constants.N_POINTS_SURFACE_PLOT_AX,)
            elif isinstance(n_points, int):
                n_points_tuple = (n_points, n_points)
            elif len(n_points) != 2:
                raise ValueError(
                    "n_points should be a number or a tuple of "
                    "length 2, and has length {}.".format(len(n_points)),
                )

            # Axes where will be evaluated
            x = np.linspace(*self.domain_range[0], n_points_tuple[0])
            y = np.linspace(*self.domain_range[1], n_points_tuple[1])

            # Evaluation of the functional object
            Z = self.fdata((x, y), grid=True)

            X, Y = np.meshgrid(x, y, indexing='ij')

            for k in range(self.fdata.dim_codomain):
                for h in range(self.fdata.n_samples):

                    set_color_dict(sample_colors, h, color_dict)

                    axes[k].plot_surface(
                        X,
                        Y,
                        Z[h, ..., k],
                        **color_dict,
                        **kwargs,
                    )

        _set_labels(self.fdata, fig, axes, patches)

        return fig


class ScatterPlot:
    """
    Class used to scatter the FDataGrid object.

    Args:
        fdata: functional data set that we want to plot.
        grid_points: points to plot.

    """

    def __init__(
        self,
        fdata: FData,
        grid_points: np.ndarray = None,
    ) -> None:
        self.fdata = fdata
        self.grid_points = grid_points

    def plot(
        self,
        chart: Union[Figure, Axes, None] = None,
        *,
        fig: Optional[Figure] = None,
        ax: Optional[Axes] = None,
        n_rows: Optional[int] = None,
        n_cols: Optional[int] = None,
        domain_range: Union[Tuple[int, int], DomainRangeLike, None] = None,
        group: Union[Sequence[Any], None] = None,
        group_colors: Union[Sequence[Any], Mapping[K, Any], None] = None,
        group_names: Union[Sequence[str], Mapping[K, str], None] = None,
        legend: bool = False,
        **kwargs: Any,
    ) -> Figure:
        """
        Scatter FDataGrid object.

        Args:
            chart: figure over with the graphs are plotted or axis
                over where the graphs are plotted. If None and ax
                is also None, the figure is initialized.
            fig: figure over with the graphs are plotted in case ax is not
                specified. If None and ax is also
                None, the figure is initialized.
            ax: axis over where the graphs are plotted. If None, see param fig.
            n_rows: designates the number of rows of the figure
                to plot the different dimensions of the image. Only specified
                if fig and ax are None.
            n_cols: designates the number of columns of the
                figure to plot the different dimensions of the image. Only
                specified if fig and ax are None.
            domain_range: Range where the function will be
                plotted. In objects with unidimensional domain
                the domain range should be a tuple with the bounds of the
                interval; in the case of surfaces a list with 2 tuples with
                the ranges for each dimension. Default uses the domain range
                of the functional object.
            group: contains integers from [0 to number of labels)
                indicating to which group each sample belongs to. Then,
                the samples with the same label are plotted in the same color.
                If None, the default value, each sample is plotted in the color
                assigned by matplotlib.pyplot.rcParams['axes.prop_cycle'].
            group_colors: colors in which groups are represented,
                there must be one for each group. If None, each
                group is shown with distict colors in the "Greys" colormap.
            group_names: name of each of the groups which appear
                in a legend, there must be one for each one. Defaults to None
                and the legend is not shown. Implies `legend=True`.
            legend: if `True`, show a legend with the groups. If
                `group_names` is passed, it will be used for finding the names
                to display in the legend. Otherwise, the values passed to
                `group` will be used.
            kwargs: if dim_domain is 1, keyword arguments to be passed to
                the matplotlib.pyplot.plot function; if dim_domain is 2,
                keyword arguments to be passed to the
                matplotlib.pyplot.plot_surface function.

        Returns:
            fig (figure object): figure object in which the graphs are plotted.

        """
        evaluated_points = None

        if self.grid_points is None:
            # This can only be done for FDataGrid
            self.grid_points = self.fdata.grid_points
            evaluated_points = self.fdata.data_matrix

        if evaluated_points is None:
            evaluated_points = self.fdata(
                self.grid_points, grid=True,
            )

        fig, axes = _get_figure_and_axes(chart, fig, ax)
        fig, axes = _set_figure_layout_for_fdata(
            self.fdata, fig, axes, n_rows, n_cols,
        )

        if domain_range is None:
            self.domain_range = self.fdata.domain_range
        else:
            self.domain_range = _to_domain_range(domain_range)

        sample_colors, patches = _get_color_info(
            self.fdata, group, group_names, group_colors, legend, kwargs,
        )

        color_dict: Mapping[str, Any] = {}

        if self.fdata.dim_domain == 1:

            for i in range(self.fdata.dim_codomain):
                for j in range(self.fdata.n_samples):

                    set_color_dict(sample_colors, j, color_dict)

                    axes[i].scatter(
                        self.grid_points[0],
                        evaluated_points[j, ..., i].T,
                        **color_dict,
                        **kwargs,
                    )

        else:

            X = self.fdata.grid_points[0]
            Y = self.fdata.grid_points[1]
            X, Y = np.meshgrid(X, Y)

            for k in range(self.fdata.dim_codomain):
                for h in range(self.fdata.n_samples):

                    set_color_dict(sample_colors, h, color_dict)

                    axes[k].scatter(
                        X,
                        Y,
                        evaluated_points[h, ..., k].T,
                        **color_dict,
                        **kwargs,
                    )

        _set_labels(self.fdata, fig, axes, patches)

        return fig


def set_color_dict(
    sample_colors: Any,
    ind: int,
    color_dict: Mapping[str, Any],
) -> None:
    """
    Auxiliary method used to update color_dict.

    Sets the new color of the color
    dict thanks to sample colors and index.
    """
    if sample_colors is not None:
        color_dict["color"] = sample_colors[ind]
