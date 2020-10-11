"""Clustering Plots Module."""

from matplotlib.ticker import MaxNLocator
from mpldatacursor import datacursor
from sklearn.exceptions import NotFittedError

from sklearn.utils.validation import check_is_fitted
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np

from ...ml.clustering import FuzzyCMeans
from ._utils import (_darken,
                     _get_figure_and_axes, _set_figure_layout_for_fdata,
                     _set_figure_layout, _set_labels)


__author__ = "Amanda Hernando Bernabé"
__email__ = "amanda.hernando@estudiante.uam.es"


def _check_if_estimator(estimator):
    """Checks the argument *estimator* is actually an estimator that
    implements the *fit* method.

    Args:
        estimator (BaseEstimator object): estimator used to calculate the
            clusters.
    """
    msg = ("This %(name)s instance has no attribute \"fit\".")
    if not hasattr(estimator, "fit"):
        raise AttributeError(msg % {'name': type(estimator).__name__})


def _plot_clustering_checks(estimator, fdata, sample_colors, sample_labels,
                            cluster_colors, cluster_labels,
                            center_colors, center_labels):
    """Checks the arguments *sample_colors*, *sample_labels*, *cluster_colors*,
    *cluster_labels*, *center_colors*, *center_labels*, passed to the plot
    functions, have the correct dimensions.

    Args:
        estimator (BaseEstimator object): estimator used to calculate the
            clusters.
        fdata (FData object): contains the samples which are grouped
            into different clusters.
        sample_colors (list of colors): contains in order the colors of each
            sample of the fdatagrid.
        sample_labels (list of str): contains in order the labels of each
            sample of the fdatagrid.
        cluster_colors (list of colors): contains in order the colors of each
            cluster the samples of the fdatagrid are classified into.
        cluster_labels (list of str): contains in order the names of each
            cluster the samples of the fdatagrid are classified into.
        center_colors (list of colors): contains in order the colors of each
            centroid of the clusters the samples of the fdatagrid are
            classified into.
        center_labels list of colors): contains in order the labels of each
            centroid of the clusters the samples of the fdatagrid are
            classified into.

    """

    if sample_colors is not None and len(
            sample_colors) != fdata.n_samples:
        raise ValueError(
            "sample_colors must contain a color for each sample.")

    if sample_labels is not None and len(
            sample_labels) != fdata.n_samples:
        raise ValueError(
            "sample_labels must contain a label for each sample.")

    if cluster_colors is not None and len(
            cluster_colors) != estimator.n_clusters:
        raise ValueError(
            "cluster_colors must contain a color for each cluster.")

    if cluster_labels is not None and len(
            cluster_labels) != estimator.n_clusters:
        raise ValueError(
            "cluster_labels must contain a label for each cluster.")

    if center_colors is not None and len(
            center_colors) != estimator.n_clusters:
        raise ValueError(
            "center_colors must contain a color for each center.")

    if center_labels is not None and len(
            center_labels) != estimator.n_clusters:
        raise ValueError(
            "centers_labels must contain a label for each center.")


def _plot_clusters(estimator, fdata, *, chart=None, fig=None, axes=None,
                   n_rows=None, n_cols=None,
                   labels, sample_labels, cluster_colors, cluster_labels,
                   center_colors, center_labels, center_width, colormap):
    """Implementation of the plot of the FDataGrid samples by clusters.

    Args:
        estimator (BaseEstimator object): estimator used to calculate the
            clusters.
        fdatagrid (FDataGrd object): contains the samples which are grouped
            into different clusters.
        fig (figure object): figure over which the graphs are plotted in
            case ax is not specified. If None and ax is also None, the figure
            is initialized.
        axes (list of axes objects): axes over where the graphs are plotted.
            If None, see param fig.
        n_rows(int): designates the number of rows of the figure to plot the
            different dimensions of the image. Only specified if fig and
            ax are None.
        n_cols(int): designates the number of columns of the figure to plot
            the different dimensions of the image. Only specified if fig
            and ax are None.
        labels (numpy.ndarray, int: (n_samples, dim_codomain)): 2-dimensional
            matrix where each row contains the number of cluster cluster
            that observation belongs to.
        sample_labels (list of str): contains in order the labels of each
            sample of the fdatagrid.
        cluster_colors (list of colors): contains in order the colors of each
            cluster the samples of the fdatagrid are classified into.
        cluster_labels (list of str): contains in order the names of each
            cluster the samples of the fdatagrid are classified into.
        center_colors (list of colors): contains in order the colors of each
            centroid of the clusters the samples of the fdatagrid are
            classified into.
        center_labels list of colors): contains in order the labels of each
            centroid of the clusters the samples of the fdatagrid are
            classified into.
        center_width (int): width of the centroids.
        colormap(colormap): colormap from which the colors of the plot are
            taken.

    Returns:
        (tuple): tuple containing:

            fig (figure object): figure object in which the graphs are plotted
                in case ax is None.

            ax (axes object): axes in which the graphs are plotted.
    """
    fig, axes = _get_figure_and_axes(chart, fig, axes)
    fig, axes = _set_figure_layout_for_fdata(fdata, fig, axes, n_rows, n_cols)

    _plot_clustering_checks(estimator, fdata, None, sample_labels,
                            cluster_colors, cluster_labels, center_colors,
                            center_labels)

    if sample_labels is None:
        sample_labels = [f'$SAMPLE: {i}$' for i in range(fdata.n_samples)]

    if cluster_colors is None:
        cluster_colors = colormap(
            np.arange(estimator.n_clusters) / (estimator.n_clusters - 1))

    if cluster_labels is None:
        cluster_labels = [
            f'$CLUSTER: {i}$' for i in range(estimator.n_clusters)]

    if center_colors is None:
        center_colors = [_darken(c, 0.5) for c in cluster_colors]

    if center_labels is None:
        center_labels = [
            f'$CENTER: {i}$' for i in range(estimator.n_clusters)]

    colors_by_cluster = cluster_colors[labels]

    patches = []
    for i in range(estimator.n_clusters):
        patches.append(
            mpatches.Patch(color=cluster_colors[i],
                           label=cluster_labels[i]))

    for j in range(fdata.dim_codomain):
        for i in range(fdata.n_samples):
            axes[j].plot(fdata.grid_points[0],
                         fdata.data_matrix[i, :, j],
                         c=colors_by_cluster[i],
                         label=sample_labels[i])
        for i in range(estimator.n_clusters):
            axes[j].plot(fdata.grid_points[0],
                         estimator.cluster_centers_.data_matrix[i, :, j],
                         c=center_colors[i],
                         label=center_labels[i],
                         linewidth=center_width)
        axes[j].legend(handles=patches)
        datacursor(formatter='{label}'.format)

    _set_labels(fdata, fig, axes)

    return fig


def plot_clusters(estimator, X, chart=None, fig=None, axes=None,
                  n_rows=None, n_cols=None,
                  sample_labels=None, cluster_colors=None,
                  cluster_labels=None, center_colors=None,
                  center_labels=None,
                  center_width=3,
                  colormap=plt.cm.get_cmap('rainbow')):
    """Plot of the FDataGrid samples by clusters.

    The clusters are calculated with the estimator passed as a parameter. If
    the estimator is not fitted, the fit method is called.
    Once each sample is assigned a label the plotting can be done.
    Each group is assigned a color described in a leglend.

    Args:
        estimator (BaseEstimator object): estimator used to calculate the
            clusters.
        X (FDataGrd object): contains the samples which are grouped
            into different clusters.
        fig (figure object): figure over which the graphs are plotted in
            case ax is not specified. If None and ax is also None, the figure
            is initialized.
        axes (list of axis objects): axis over where the graphs are plotted.
            If None, see param fig.
        n_rows (int): designates the number of rows of the figure to plot the
            different dimensions of the image. Only specified if fig and
            ax are None.
        n_cols (int): designates the number of columns of the figure to plot
            the different dimensions of the image. Only specified if fig
            and ax are None.
        sample_labels (list of str): contains in order the labels of each
        sample of the fdatagrid.
        cluster_colors (list of colors): contains in order the colors of each
            cluster the samples of the fdatagrid are classified into.
        cluster_labels (list of str): contains in order the names of each
            cluster the samples of the fdatagrid are classified into.
        center_colors (list of colors): contains in order the colors of each
            centroid of the clusters the samples of the fdatagrid are
            classified into.
        center_labels (list of colors): contains in order the labels of each
            centroid of the clusters the samples of the fdatagrid are
            classified into.
        center_width (int): width of the centroid curves.
        colormap(colormap): colormap from which the colors of the plot are
            taken. Defaults to `rainbow`.

    Returns:
        (tuple): tuple containing:

            fig (figure object): figure object in which the graphs are plotted
            in case ax is None.

            ax (axes object): axes in which the graphs are plotted.
    """
    _check_if_estimator(estimator)
    try:
        check_is_fitted(estimator)
        estimator._check_test_data(X)
    except NotFittedError:
        estimator.fit(X)

    if isinstance(estimator, FuzzyCMeans):
        labels = np.argmax(estimator.labels_, axis=1)
    else:
        labels = estimator.labels_

    return _plot_clusters(estimator=estimator, fdata=X,
                          fig=fig, axes=axes, n_rows=n_rows, n_cols=n_cols,
                          labels=labels, sample_labels=sample_labels,
                          cluster_colors=cluster_colors,
                          cluster_labels=cluster_labels,
                          center_colors=center_colors,
                          center_labels=center_labels,
                          center_width=center_width,
                          colormap=colormap)


def _get_labels(x_label, y_label, title, xlabel_str):
    """Sets the arguments *xlabel*, *ylabel*, *title* passed to the plot
    functions :func:`plot_cluster_lines
    <skfda.exploratory.visualization.clustering_plots.plot_cluster_lines>` and
    :func:`plot_cluster_bars
    <skfda.exploratory.visualization.clustering_plots.plot_cluster_bars>`,
    in case they are not set yet.

    Args:
        xlabel (lstr): Label for the x-axes.
        ylabel (str): Label for the y-axes.
        title (str): Title for the figure where the clustering results are
            ploted.
        xlabel_str (str): In case xlabel is None, string to use for the labels
            in the x-axes.

    Returns:
        xlabel (str): Labels for the x-axes.
        ylabel (str): Labels for the y-axes.
        title (str): Title for the figure where the clustering results are
            plotted.
    """
    if x_label is None:
        x_label = xlabel_str

    if y_label is None:
        y_label = "Degree of membership"

    if title is None:
        title = "Degrees of membership of the samples to each cluster"

    return x_label, y_label, title


def plot_cluster_lines(estimator, X, chart=None, fig=None, axes=None,
                       sample_colors=None, sample_labels=None,
                       cluster_labels=None,
                       colormap=plt.cm.get_cmap('rainbow'),
                       x_label=None, y_label=None, title=None):
    """Implementation of the plotting of the results of the
    :func:`Fuzzy K-Means <fda.clustering.fuzzy_kmeans>` method.


    A kind of Parallel Coordinates plot is generated in this function with the
    membership values obtained from the algorithm. A line is plotted for each
    sample with the values for each cluster. See `Clustering Example
    <../auto_examples/plot_clustering.html>`_.

    Args:
        estimator (BaseEstimator object): estimator used to calculate the
            clusters.
        X (FDataGrd object): contains the samples which are grouped
            into different clusters.
        fig (figure object, optional): figure over which the graph is
            plotted in case ax is not specified. If None and ax is also None,
            the figure is initialized.
        axes (axes object, optional): axis over where the graph is  plotted.
            If None, see param fig.
        sample_colors (list of colors, optional): contains in order the colors
            of each sample of the fdatagrid.
        sample_labels (list of str, optional): contains in order the labels
            of each sample  of the fdatagrid.
        cluster_labels (list of str, optional): contains in order the names of
            each cluster the samples of the fdatagrid are classified into.
        colormap(colormap, optional): colormap from which the colors of the
            plot are taken.
        x_label (str): Label for the x-axis. Defaults to "Cluster".
        y_label (str): Label for the y-axis. Defaults to
            "Degree of membership".
        title (str, optional): Title for the figure where the clustering
            results are ploted.
            Defaults to "Degrees of membership of the samples to each cluster".

    Returns:
        (tuple): tuple containing:

            fig (figure object): figure object in which the graphs are plotted
                in case ax is None.

            ax (axes object): axes in which the graphs are plotted.

    """
    fdata = X
    _check_if_estimator(estimator)

    if not isinstance(estimator, FuzzyCMeans):
        raise ValueError("The estimator must be a FuzzyCMeans object.")

    try:
        check_is_fitted(estimator)
        estimator._check_test_data(X)
    except NotFittedError:
        estimator.fit(X)

    fig, axes = _get_figure_and_axes(chart, fig, axes)
    fig, axes = _set_figure_layout(fig, axes)

    _plot_clustering_checks(estimator, fdata, sample_colors, sample_labels,
                            None, cluster_labels, None, None)

    x_label, y_label, title = _get_labels(x_label, y_label, title, "Cluster")

    if sample_colors is None:
        cluster_colors = colormap(np.arange(estimator.n_clusters) /
                                  (estimator.n_clusters - 1))
        labels_by_cluster = np.argmax(estimator.labels_, axis=1)
        sample_colors = cluster_colors[labels_by_cluster]

    if sample_labels is None:
        sample_labels = ['$SAMPLE: {}$'.format(i) for i in
                         range(fdata.n_samples)]

    if cluster_labels is None:
        cluster_labels = ['${}$'.format(i) for i in
                          range(estimator.n_clusters)]

    axes[0].get_xaxis().set_major_locator(MaxNLocator(integer=True))
    for i in range(fdata.n_samples):
        axes[0].plot(np.arange(estimator.n_clusters),
                     estimator.labels_[i],
                     label=sample_labels[i],
                     color=sample_colors[i])
    axes[0].set_xticks(np.arange(estimator.n_clusters))
    axes[0].set_xticklabels(cluster_labels)
    axes[0].set_xlabel(x_label)
    axes[0].set_ylabel(y_label)
    datacursor(formatter='{label}'.format)

    fig.suptitle(title)
    return fig


def plot_cluster_bars(estimator, X, chart=None, fig=None, axes=None, sort=-1,
                      sample_labels=None, cluster_colors=None,
                      cluster_labels=None, colormap=plt.cm.get_cmap('rainbow'),
                      x_label=None, y_label=None, title=None):
    """Implementation of the plotting of the results of the
    :func:`Fuzzy K-Means <fda.clustering.fuzzy_kmeans>` method.


    A kind of barplot is generated in this function with the membership values
    obtained from the algorithm. There is a bar for each sample whose height is
    1 (the sum of the membership values of a sample add to 1), and the part
    proportional to each cluster is coloured with the corresponding color. See
    `Clustering Example <../auto_examples/plot_clustering.html>`_.

    Args:
        estimator (BaseEstimator object): estimator used to calculate the
            clusters.
        X (FDataGrd object): contains the samples which are grouped
            into different clusters.
        fig (figure object, optional): figure over which the graph is
            plotted in case ax is not specified. If None and ax is also None,
            the figure is initialized.
        axes (axes object, optional): axes over where the graph is  plotted.
            If None, see param fig.
        sort(int, optional): Number in the range [-1, n_clusters) designating
            the cluster whose labels are sorted in a decrementing order.
            Defaults to -1, in this case, no sorting is done.
        sample_labels (list of str, optional): contains in order the labels
            of each sample  of the fdatagrid.
        cluster_labels (list of str, optional): contains in order the names of
            each cluster the samples of the fdatagrid are classified into.
        cluster_colors (list of colors): contains in order the colors of each
            cluster the samples of the fdatagrid are classified into.
        colormap(colormap, optional): colormap from which the colors of the
            plot are taken.
        x_label (str): Label for the x-axis. Defaults to "Sample".
        y_label (str): Label for the y-axis. Defaults to
            "Degree of membership".
        title (str): Title for the figure where the clustering results are
            plotted.
            Defaults to "Degrees of membership of the samples to each cluster".

    Returns:
        (tuple): tuple containing:

            fig (figure object): figure object in which the graph is plotted
                in case ax is None.

            ax (axis object): axis in which the graph is plotted.

    """
    fdata = X
    _check_if_estimator(estimator)

    if not isinstance(estimator, FuzzyCMeans):
        raise ValueError("The estimator must be a FuzzyCMeans object.")

    try:
        check_is_fitted(estimator)
        estimator._check_test_data(X)
    except NotFittedError:
        estimator.fit(X)

    if sort < -1 or sort >= estimator.n_clusters:
        raise ValueError(
            "The sorting number must belong to the interval [-1, n_clusters)")

    fig, axes = _get_figure_and_axes(chart, fig, axes)
    fig, axes = _set_figure_layout(fig, axes)

    _plot_clustering_checks(estimator, fdata, None, sample_labels,
                            cluster_colors, cluster_labels, None, None)

    x_label, y_label, title = _get_labels(x_label, y_label, title, "Sample")

    if sample_labels is None:
        sample_labels = np.arange(fdata.n_samples)

    if cluster_colors is None:
        cluster_colors = colormap(
            np.arange(estimator.n_clusters) / (estimator.n_clusters - 1))

    if cluster_labels is None:
        cluster_labels = [f'$CLUSTER: {i}$' for i in
                          range(estimator.n_clusters)]

    patches = []
    for i in range(estimator.n_clusters):
        patches.append(
            mpatches.Patch(color=cluster_colors[i], label=cluster_labels[i]))

    if sort != -1:
        sample_indices = np.argsort(-estimator.labels_[:, sort])
        sample_labels = np.copy(sample_labels[sample_indices])
        labels_dim = np.copy(estimator.labels_[sample_indices])

        temp_labels = np.copy(labels_dim[:, 0])
        labels_dim[:, 0] = labels_dim[:, sort]
        labels_dim[:, sort] = temp_labels

        temp_color = np.copy(cluster_colors[0])
        cluster_colors[0] = cluster_colors[sort]
        cluster_colors[sort] = temp_color
    else:
        labels_dim = estimator.labels_

    conc = np.zeros((fdata.n_samples, 1))
    labels_dim = np.concatenate((conc, labels_dim), axis=-1)
    for i in range(estimator.n_clusters):
        axes[0].bar(np.arange(fdata.n_samples),
                    labels_dim[:, i + 1],
                    bottom=np.sum(labels_dim[:, :(i + 1)], axis=1),
                    color=cluster_colors[i])
    axes[0].set_xticks(np.arange(fdata.n_samples))
    axes[0].set_xticklabels(sample_labels)
    axes[0].set_xlabel(x_label)
    axes[0].set_ylabel(y_label)
    axes[0].legend(handles=patches)

    fig.suptitle(title)
    return fig
