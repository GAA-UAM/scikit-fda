import math

import matplotlib.axes
import matplotlib.cm
import matplotlib.figure
import matplotlib.patches

import numpy as np

from ..._utils import _create_figure, _list_of_arrays, constants


def _get_figure_and_axes(chart=None, fig=None, axes=None):
    """Obtain the figure and axes from the arguments."""

    num_defined = sum(e is not None for e in (chart, fig, axes))
    if num_defined > 1:
        raise ValueError("Only one of chart, fig and axes parameters"
                         "can be passed as an argument.")

    # Parse chart argument
    if chart is not None:
        if isinstance(chart, matplotlib.figure.Figure):
            fig = chart
        else:
            axes = chart

    if fig is None and axes is None:
        fig = fig = _create_figure()
        axes = []

    elif fig is not None:
        axes = fig.axes

    else:
        if isinstance(axes, matplotlib.axes.Axes):
            axes = [axes]

        fig = axes[0].figure

    return fig, axes


def _get_axes_shape(fdata, n_rows=None, n_cols=None):
    """Get the number of rows and columns of the subplots"""

    if ((n_rows is not None and n_cols is not None)
            and ((n_rows * n_cols) < fdata.dim_codomain)):
        raise ValueError(f"The number of rows ({n_rows}) multiplied by "
                         f"the number of columns ({n_cols}) "
                         f"is less than the dimension of "
                         f"the image ({fdata.dim_codomain})")

    if n_rows is None and n_cols is None:
        n_cols = int(math.ceil(math.sqrt(fdata.dim_codomain)))
        n_rows = int(math.ceil(fdata.dim_codomain / n_cols))
    elif n_rows is None and n_cols is not None:
        n_rows = int(math.ceil(fdata.dim_codomain / n_cols))
    elif n_cols is None and n_rows is not None:
        n_cols = int(math.ceil(fdata.dim_codomain / n_rows))

    return n_rows, n_cols


def _set_figure_layout_for_fdata(fdata, fig=None, axes=None,
                                 n_rows=None, n_cols=None):
    """Set the figure axes for plotting a
    :class:`~skfda.representation.FData` object.

    Args:
        fdata (FData): functional data object.
        fig (figure object): figure over with the graphs are
            plotted in case ax is not specified.
        ax (list of axis objects): axis over where the graphs are
            plotted.
        n_rows (int, optional): designates the number of rows of the figure
            to plot the different dimensions of the image. Can only be passed
            if no axes are specified.
        n_cols (int, optional): designates the number of columns of the
            figure to plot the different dimensions of the image. Can only be
            passed if no axes are specified.

    Returns:
        (tuple): tuple containing:

            * fig (figure): figure object in which the graphs are plotted.
            * axes (list): axes in which the graphs are plotted.

    """
    if fdata.dim_domain > 2:
        raise NotImplementedError("Plot only supported for functional data"
                                  " modeled in at most 3 dimensions.")

    if len(axes) != 0 and len(axes) != fdata.dim_codomain:
        raise ValueError("The number of axes must be 0 (to create them) or "
                         "equal to the dimension of the image.")

    if len(axes) != 0 and (n_rows is not None or n_cols is not None):
        raise ValueError("The number of columns and/or number of rows of "
                         "the figure, in which each dimension of the "
                         "image is plotted, can only be customized in case "
                         "that no axes are provided.")

    if fdata.dim_domain == 1:
        projection = 'rectilinear'
    else:
        projection = '3d'

    if len(axes) == 0:
        # Create the axes

        n_rows, n_cols = _get_axes_shape(fdata, n_rows, n_cols)
        fig.subplots(nrows=n_rows, ncols=n_cols,
                     subplot_kw={"projection": projection})
        axes = fig.axes

    else:
        # Check that the projections are right

        if not all(a.name == projection for a in axes):
            raise ValueError(f"The projection of the axes should be "
                             f"{projection}")

    return fig, axes


def _get_label_colors(n_labels, label_colors=None):
    """Get the colors of each label"""

    if label_colors is not None:
        if len(label_colors) != n_labels:
            raise ValueError("There must be a color in label_colors "
                             "for each of the labels that appear in "
                             "sample_labels.")
    else:
        colormap = matplotlib.cm.get_cmap()
        label_colors = colormap(np.arange(n_labels) / (n_labels - 1))

    return label_colors


def _get_color_info(fdata, sample_labels, label_names, label_colors, kwargs):

    patches = None

    if sample_labels is not None:
        # In this case, each curve has a label, and all curves with the same
        # label should have the same color

        sample_labels = np.asarray(sample_labels)

        n_labels = np.max(sample_labels) + 1

        if np.any((sample_labels < 0) | (sample_labels >= n_labels)) or \
                not np.all(np.isin(range(n_labels), sample_labels)):
            raise ValueError("Sample_labels must contain at least an "
                             "occurence of numbers between 0 and number "
                             "of distint sample labels.")

        label_colors = _get_label_colors(n_labels, label_colors)
        sample_colors = np.asarray(label_colors)[sample_labels]

        if label_names is not None:
            if len(label_names) != n_labels:
                raise ValueError("There must be a name in  label_names "
                                 "for each of the labels that appear in "
                                 "sample_labels.")

            patches = [matplotlib.patches.Patch(color=c, label=l)
                       for c, l in zip(label_colors, label_names)]

    else:
        # In this case, each curve has a different color unless specified
        # otherwise

        if 'color' in kwargs:
            sample_colors = fdata.n_samples * [kwargs.get("color")]
            kwargs.pop('color')

        elif 'c' in kwargs:
            sample_colors = fdata.n_samples * [kwargs.get("color")]
            kwargs.pop('c')

        else:
            sample_colors = None

    return sample_colors, patches


def _set_labels(fdata, fig=None, axes=None, patches=None):
    """Set labels if any.

    Args:
        fdata (FData): functional data object.
        fig (figure object): figure object containing the axes that
            implement set_xlabel and set_ylabel, and set_zlabel in case
            of a 3d projection.
        axes (list of axes): axes objects that implement set_xlabel and
            set_ylabel, and set_zlabel in case of a 3d projection; used if
            fig is None.
        patches (list of mpatches.Patch); objects used to generate each
            entry in the legend.

    """

    # Dataset name
    if fdata.dataset_label is not None:
        fig.suptitle(fdata.dataset_label)

    # Legend
    if patches is not None and fdata.dim_codomain > 1:
        fig.legend(handles=patches)
    elif patches is not None:
        axes[0].legend(handles=patches)

    # Axis labels
    if fdata.axes_labels is not None:
        if axes[0].name == '3d':
            for i in range(fdata.dim_codomain):
                if fdata.axes_labels[0] is not None:
                    axes[i].set_xlabel(fdata.axes_labels[0])
                if fdata.axes_labels[1] is not None:
                    axes[i].set_ylabel(fdata.axes_labels[1])
                if fdata.axes_labels[i + 2] is not None:
                    axes[i].set_zlabel(fdata.axes_labels[i + 2])
        else:
            for i in range(fdata.dim_codomain):
                if fdata.axes_labels[0] is not None:
                    axes[i].set_xlabel(fdata.axes_labels[0])
                if fdata.axes_labels[i + 1] is not None:
                    axes[i].set_ylabel(fdata.axes_labels[i + 1])


def plot_hypersurfaces(fdata, chart=None, *, derivative=0, fig=None, axes=None,
                       n_rows=None, n_cols=None, n_points=None,
                       domain_range=None,
                       sample_labels=None, label_colors=None, label_names=None,
                       **kwargs):
    """Plot the FDatGrid object as hypersurfaces.

    Plots each coordinate separately. If the domain is one dimensional, the
    plots will be curves, and if it is two dimensional, they will be surfaces.

    Args:
        chart (figure object, axe or list of axes, optional): figure over
            with the graphs are plotted or axis over where the graphs are
            plotted. If None and ax is also None, the figure is
            initialized.
        derivative (int or tuple, optional): Order of derivative to be
            plotted. In case of surfaces a tuple with the order of
            derivation in each direction can be passed. See
            :func:`evaluate` to obtain more information. Defaults 0.
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
        sample_labels (list of int): contains integers from [0 to number of
            labels) indicating to which group each sample belongs to. Then,
            the samples with the same label are plotted in the same color.
            If None, the default value, each sample is plotted in the color
            assigned by matplotlib.pyplot.rcParams['axes.prop_cycle'].
        label_colors (list of colors): colors in which groups are
            represented, there must be one for each group. If None, each
            group is shown with distict colors in the "Greys" colormap.
        label_names (list of str): name of each of the groups which appear
            in a legend, there must be one for each one. Defaults to None
            and the legend is not shown.
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
        domain_range = _list_of_arrays(domain_range)

    sample_colors, patches = _get_color_info(
        fdata, sample_labels, label_names, label_colors, kwargs)

    if fdata.dim_domain == 1:

        if n_points is None:
            n_points = constants.N_POINTS_UNIDIMENSIONAL_PLOT_MESH

        # Evaluates the object in a linspace
        eval_points = np.linspace(*domain_range[0], n_points)
        mat = fdata(eval_points, derivative=derivative, keepdims=True)

        color_dict = {}

        for i in range(fdata.dim_codomain):
            for j in range(fdata.n_samples):

                if sample_labels is not None:
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
        Z = fdata((x, y), derivative=derivative, grid=True, keepdims=True)

        X, Y = np.meshgrid(x, y, indexing='ij')

        color_dict = {}

        for i in range(fdata.dim_codomain):
            for j in range(fdata.n_samples):

                if sample_labels is not None:
                    color_dict["color"] = sample_colors[j]

                axes[i].plot_surface(X, Y, Z[j, ..., i],
                                     **color_dict, **kwargs)

    _set_labels(fdata, fig, axes, patches)

    return fig


def plot_scatter(fdata, chart=None, *, sample_points=None, derivative=0,
                 fig=None, axes=None,
                 n_rows=None, n_cols=None, n_points=None, domain_range=None,
                 sample_labels=None, label_colors=None, label_names=None,
                 **kwargs):
    """Plot the FDatGrid object.

    Args:
        chart (figure object, axe or list of axes, optional): figure over
            with the graphs are plotted or axis over where the graphs are
            plotted. If None and ax is also None, the figure is
            initialized.
        sample_points (ndarray): points to plot.
        derivative (int or tuple, optional): Order of derivative to be
            plotted. In case of surfaces a tuple with the order of
            derivation in each direction can be passed. See
            :func:`evaluate` to obtain more information. Defaults 0.
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
        sample_labels (list of int): contains integers from [0 to number of
            labels) indicating to which group each sample belongs to. Then,
            the samples with the same label are plotted in the same color.
            If None, the default value, each sample is plotted in the color
            assigned by matplotlib.pyplot.rcParams['axes.prop_cycle'].
        label_colors (list of colors): colors in which groups are
            represented, there must be one for each group. If None, each
            group is shown with distict colors in the "Greys" colormap.
        label_names (list of str): name of each of the groups which appear
            in a legend, there must be one for each one. Defaults to None
            and the legend is not shown.
        **kwargs: if dim_domain is 1, keyword arguments to be passed to
            the matplotlib.pyplot.plot function; if dim_domain is 2,
            keyword arguments to be passed to the
            matplotlib.pyplot.plot_surface function.

    Returns:
        fig (figure object): figure object in which the graphs are plotted.

    """

    if sample_points is None:
        # This can only be done for FDataGrid
        sample_points = fdata.sample_points
        evaluated_points = fdata.data_matrix
    else:
        evaluated_points = fdata(sample_points, grid=True)

    fig, axes = _get_figure_and_axes(chart, fig, axes)
    fig, axes = _set_figure_layout_for_fdata(fdata, fig, axes, n_rows, n_cols)

    if domain_range is None:
        domain_range = fdata.domain_range
    else:
        domain_range = _list_of_arrays(domain_range)

    sample_colors, patches = _get_color_info(
        fdata, sample_labels, label_names, label_colors, kwargs)

    if fdata.dim_domain == 1:

        color_dict = {}

        for i in range(fdata.dim_codomain):
            for j in range(fdata.n_samples):

                if sample_labels is not None:
                    color_dict["color"] = sample_colors[j]

                axes[i].scatter(sample_points[0],
                                evaluated_points[j, ..., i].T,
                                **color_dict, **kwargs)

    else:

        X = fdata.sample_points[0]
        Y = fdata.sample_points[1]
        X, Y = np.meshgrid(X, Y)

        color_dict = {}

        for i in range(fdata.dim_codomain):
            for j in range(fdata.n_samples):

                if sample_labels is not None:
                    color_dict["color"] = sample_colors[j]

                axes[i].scatter(X, Y,
                                evaluated_points[j, ..., i].T,
                                **color_dict, **kwargs)

    _set_labels(fdata, fig, axes, patches)

    return fig
