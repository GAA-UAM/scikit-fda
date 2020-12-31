
import matplotlib.cm
import matplotlib.patches

import numpy as np

from ..._utils import _tuple_of_arrays, constants
from ._utils import (_get_figure_and_axes, _set_figure_layout_for_fdata,
                     _set_labels)


def _get_label_colors(n_labels, group_colors=None):
    """Get the colors of each label"""

    if group_colors is not None:
        if len(group_colors) != n_labels:
            raise ValueError("There must be a color in group_colors "
                             "for each of the labels that appear in "
                             "group.")
    else:
        colormap = matplotlib.cm.get_cmap()
        group_colors = colormap(np.arange(n_labels) / (n_labels - 1))

    return group_colors


def _get_color_info(fdata, group, group_names, group_colors, legend, kwargs):

    patches = None

    if group is not None:
        # In this case, each curve has a label, and all curves with the same
        # label should have the same color

        group_unique, group_indexes = np.unique(group, return_inverse=True)
        n_labels = len(group_unique)

        if group_colors is not None:
            group_colors_array = np.array(
                [group_colors[g] for g in group_unique])
        else:
            prop_cycle = matplotlib.rcParams['axes.prop_cycle']
            cycle_colors = prop_cycle.by_key()['color']

            group_colors_array = np.take(
                cycle_colors, np.arange(n_labels), mode='wrap')

        sample_colors = group_colors_array[group_indexes]

        group_names_array = None

        if group_names is not None:
            group_names_array = np.array(
                [group_names[g] for g in group_unique])
        elif legend is True:
            group_names_array = group_unique

        if group_names_array is not None:
            patches = [matplotlib.patches.Patch(color=c, label=l)
                       for c, l in zip(group_colors_array, group_names_array)]

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


def plot_graph(fdata, chart=None, *, fig=None, axes=None,
               n_rows=None, n_cols=None, n_points=None,
               domain_range=None,
               group=None, group_colors=None, group_names=None,
               legend: bool = False,
               **kwargs):
    """Plot the FDatGrid object graph as hypersurfaces.

    Plots each coordinate separately. If the :term:`domain` is one dimensional,
    the plots will be curves, and if it is two dimensional, they will be
    surfaces.

    Args:
        chart (figure object, axe or list of axes, optional): figure over
            with the graphs are plotted or axis over where the graphs are
            plotted. If None and ax is also None, the figure is
            initialized.
        fig (figure object, optional): figure over with the graphs are
            plotted in case ax is not specified. If None and ax is also
            None, the figure is initialized.
        axes (list of axis objects, optional): axis over where the graphs are
            plotted. If None, see param fig.
        n_rows (int, optional): designates the number of rows of the figure
            to plot the different dimensions of the image. Only specified
            if fig and ax are None.
        n_cols(int, optional): designates the number of columns of the
            figure to plot the different dimensions of the image. Only
            specified if fig and ax are None.
        n_points (int or tuple, optional): Number of points to evaluate in
            the plot. In case of surfaces a tuple of length 2 can be pased
            with the number of points to plot in each axis, otherwise the
            same number of points will be used in the two axes. By default
            in unidimensional plots will be used 501 points; in surfaces
            will be used 30 points per axis, wich makes a grid with 900
            points.
        domain_range (tuple or list of tuples, optional): Range where the
            function will be plotted. In objects with unidimensional domain
            the domain range should be a tuple with the bounds of the
            interval; in the case of surfaces a list with 2 tuples with
            the ranges for each dimension. Default uses the domain range
            of the functional object.
        group (list of int): contains integers from [0 to number of
            labels) indicating to which group each sample belongs to. Then,
            the samples with the same label are plotted in the same color.
            If None, the default value, each sample is plotted in the color
            assigned by matplotlib.pyplot.rcParams['axes.prop_cycle'].
        group_colors (list of colors): colors in which groups are
            represented, there must be one for each group. If None, each
            group is shown with distict colors in the "Greys" colormap.
        group_names (list of str): name of each of the groups which appear
            in a legend, there must be one for each one. Defaults to None
            and the legend is not shown. Implies `legend=True`.
        legend (bool): if `True`, show a legend with the groups. If
            `group_names` is passed, it will be used for finding the names
            to display in the legend. Otherwise, the values passed to
            `group` will be used.
        **kwargs: if dim_domain is 1, keyword arguments to be passed to
            the matplotlib.pyplot.plot function; if dim_domain is 2,
            keyword arguments to be passed to the
            matplotlib.pyplot.plot_surface function.

    Returns:
        fig (figure object): figure object in which the graphs are plotted.

    """

    fig, axes = _get_figure_and_axes(chart, fig, axes)
    fig, axes = _set_figure_layout_for_fdata(fdata, fig, axes, n_rows, n_cols)

    if domain_range is None:
        domain_range = fdata.domain_range
    else:
        domain_range = _tuple_of_arrays(domain_range)

    sample_colors, patches = _get_color_info(
        fdata, group, group_names, group_colors, legend, kwargs)

    if fdata.dim_domain == 1:

        if n_points is None:
            n_points = constants.N_POINTS_UNIDIMENSIONAL_PLOT_MESH

        # Evaluates the object in a linspace
        eval_points = np.linspace(*domain_range[0], n_points)
        mat = fdata(eval_points)

        color_dict = {}

        for i in range(fdata.dim_codomain):
            for j in range(fdata.n_samples):

                if sample_colors is not None:
                    color_dict["color"] = sample_colors[j]

                axes[i].plot(eval_points, mat[j, ..., i].T,
                             **color_dict, **kwargs)

    else:

        # Selects the number of points
        if n_points is None:
            npoints = 2 * (constants.N_POINTS_SURFACE_PLOT_AX,)
        elif np.isscalar(npoints):
            npoints = (npoints, npoints)
        elif len(npoints) != 2:
            raise ValueError(f"n_points should be a number or a tuple of "
                             f"length 2, and has length {len(npoints)}")

        # Axes where will be evaluated
        x = np.linspace(*domain_range[0], npoints[0])
        y = np.linspace(*domain_range[1], npoints[1])

        # Evaluation of the functional object
        Z = fdata((x, y), grid=True)

        X, Y = np.meshgrid(x, y, indexing='ij')

        color_dict = {}

        for i in range(fdata.dim_codomain):
            for j in range(fdata.n_samples):

                if sample_colors is not None:
                    color_dict["color"] = sample_colors[j]

                axes[i].plot_surface(X, Y, Z[j, ..., i],
                                     **color_dict, **kwargs)

    _set_labels(fdata, fig, axes, patches)

    return fig


def plot_scatter(fdata, chart=None, *, grid_points=None,
                 fig=None, axes=None,
                 n_rows=None, n_cols=None, domain_range=None,
                 group=None, group_colors=None, group_names=None,
                 legend: bool = False,
                 **kwargs):
    """Plot the FDatGrid object.

    Args:
        chart (figure object, axe or list of axes, optional): figure over
            with the graphs are plotted or axis over where the graphs are
            plotted. If None and ax is also None, the figure is
            initialized.
        grid_points (ndarray): points to plot.
        fig (figure object, optional): figure over with the graphs are
            plotted in case ax is not specified. If None and ax is also
            None, the figure is initialized.
        axes (list of axis objects, optional): axis over where the graphs are
            plotted. If None, see param fig.
        n_rows (int, optional): designates the number of rows of the figure
            to plot the different dimensions of the image. Only specified
            if fig and ax are None.
        n_cols(int, optional): designates the number of columns of the
            figure to plot the different dimensions of the image. Only
            specified if fig and ax are None.
        domain_range (tuple or list of tuples, optional): Range where the
            function will be plotted. In objects with unidimensional domain
            the domain range should be a tuple with the bounds of the
            interval; in the case of surfaces a list with 2 tuples with
            the ranges for each dimension. Default uses the domain range
            of the functional object.
        group (list of int): contains integers from [0 to number of
            labels) indicating to which group each sample belongs to. Then,
            the samples with the same label are plotted in the same color.
            If None, the default value, each sample is plotted in the color
            assigned by matplotlib.pyplot.rcParams['axes.prop_cycle'].
        group_colors (list of colors): colors in which groups are
            represented, there must be one for each group. If None, each
            group is shown with distict colors in the "Greys" colormap.
        group_names (list of str): name of each of the groups which appear
            in a legend, there must be one for each one. Defaults to None
            and the legend is not shown. Implies `legend=True`.
        legend (bool): if `True`, show a legend with the groups. If
            `group_names` is passed, it will be used for finding the names
            to display in the legend. Otherwise, the values passed to
            `group` will be used.
        **kwargs: if dim_domain is 1, keyword arguments to be passed to
            the matplotlib.pyplot.plot function; if dim_domain is 2,
            keyword arguments to be passed to the
            matplotlib.pyplot.plot_surface function.

    Returns:
        fig (figure object): figure object in which the graphs are plotted.

    """

    evaluated_points = None

    if grid_points is None:
        # This can only be done for FDataGrid
        grid_points = fdata.grid_points
        evaluated_points = fdata.data_matrix

    if evaluated_points is None:
        evaluated_points = fdata(
            grid_points, grid=True)

    fig, axes = _get_figure_and_axes(chart, fig, axes)
    fig, axes = _set_figure_layout_for_fdata(fdata, fig, axes, n_rows, n_cols)

    if domain_range is None:
        domain_range = fdata.domain_range
    else:
        domain_range = _tuple_of_arrays(domain_range)

    sample_colors, patches = _get_color_info(
        fdata, group, group_names, group_colors, legend, kwargs)

    if fdata.dim_domain == 1:

        color_dict = {}

        for i in range(fdata.dim_codomain):
            for j in range(fdata.n_samples):

                if sample_colors is not None:
                    color_dict["color"] = sample_colors[j]

                axes[i].scatter(grid_points[0],
                                evaluated_points[j, ..., i].T,
                                **color_dict, **kwargs)

    else:

        X = fdata.grid_points[0]
        Y = fdata.grid_points[1]
        X, Y = np.meshgrid(X, Y)

        color_dict = {}

        for i in range(fdata.dim_codomain):
            for j in range(fdata.n_samples):

                if sample_colors is not None:
                    color_dict["color"] = sample_colors[j]

                axes[i].scatter(X, Y,
                                evaluated_points[j, ..., i].T,
                                **color_dict, **kwargs)

    _set_labels(fdata, fig, axes, patches)

    return fig
