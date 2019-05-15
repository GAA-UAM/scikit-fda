"""Clustering Plots Module."""

from ...ml.clustering.base_kmeans import FuzzyKMeans
import numpy as np
import matplotlib.pyplot as plt
from mpldatacursor import datacursor
import matplotlib.patches as mpatches
from matplotlib.ticker import MaxNLocator
from sklearn.exceptions import NotFittedError

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


def _plot_clustering_checks(estimator, fdatagrid, sample_colors, sample_labels,
                            cluster_colors, cluster_labels,
                            center_colors, center_labels):
    """Checks the arguments *sample_colors*, *sample_labels*, *cluster_colors*,
    *cluster_labels*, *center_colors*, *center_labels*, passed to the plot
    functions, have the correct dimensions.

    Args:
        estimator (BaseEstimator object): estimator used to calculate the
            clusters.
        fdatagrid (FDataGrd object): contains the samples which are grouped
            into different clusters.
        sample_colors (list of colors): contains in order the colors of each
            sample of the fdatagrid.
        sample_labels (list of str): contains in order the labels of each sample
            of the fdatagrid.
        cluster_colors (list of colors): contains in order the colors of each
            cluster the samples of the fdatagrid are classified into.
        cluster_labels (list of str): contains in order the names of each
            cluster the samples of the fdatagrid are classified into.
        center_colors (list of colors): contains in order the colors of each
            centroid of the clusters the samples of the fdatagrid are classified into.
        center_labels list of colors): contains in order the labels of each
            centroid of the clusters the samples of the fdatagrid are classified into.

    """

    if sample_colors is not None and len(
            sample_colors) != fdatagrid.nsamples:
        raise ValueError(
            "sample_colors must contain a color for each sample.")

    if sample_labels is not None and len(
            sample_labels) != fdatagrid.nsamples:
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


def _plot_clusters(estimator, fdatagrid, fig, ax, nrows, ncols, labels,
                   sample_labels, cluster_colors, cluster_labels,
                   center_colors, center_labels, colormap):
    """Implementation of the plot of the FDataGrid samples by clusters.

    Args:
        estimator (BaseEstimator object): estimator used to calculate the
            clusters.
        fdatagrid (FDataGrd object): contains the samples which are grouped
            into different clusters.
        fig (figure object): figure over which the graphs are plotted in
            case ax is not specified. If None and ax is also None, the figure
            is initialized.
        ax (list of axis objects): axis over where the graphs are plotted.
            If None, see param fig.
        nrows(int): designates the number of rows of the figure to plot the
            different dimensions of the image. Only specified if fig and
            ax are None.
        ncols(int): designates the number of columns of the figure to plot
            the different dimensions of the image. Only specified if fig
            and ax are None.
        labels (numpy.ndarray, int: (nsamples, ndim_image)): 2-dimensional
            matrix where each row contains the number of cluster cluster
            that observation belongs to.
        sample_labels (list of str): contains in order the labels of each sample
            of the fdatagrid.
        cluster_colors (list of colors): contains in order the colors of each
            cluster the samples of the fdatagrid are classified into.
        cluster_labels (list of str): contains in order the names of each
            cluster the samples of the fdatagrid are classified into.
        center_colors (list of colors): contains in order the colors of each
            centroid of the clusters the samples of the fdatagrid are classified into.
        center_labels list of colors): contains in order the labels of each
            centroid of the clusters the samples of the fdatagrid are classified into.
        colormap(colormap): colormap from which the colors of the plot are taken.

    Returns:
        (tuple): tuple containing:

            fig (figure object): figure object in which the graphs are plotted in case ax is None.

            ax (axes object): axes in which the graphs are plotted.
    """
    fig, ax = fdatagrid.generic_plotting_checks(fig, ax, nrows, ncols)

    _plot_clustering_checks(estimator, fdatagrid, None, sample_labels,
                            cluster_colors, cluster_labels, center_colors,
                            center_labels)

    if sample_labels is None:
        sample_labels = ['$SAMPLE: {}$'.format(i) for i in
                         range(fdatagrid.nsamples)]

    if cluster_colors is None:
        cluster_colors = colormap(
            np.arange(estimator.n_clusters) / (estimator.n_clusters - 1))

    if cluster_labels is None:
        cluster_labels = ['$CLUSTER: {}$'.format(i) for i in
                          range(estimator.n_clusters)]

    if center_colors is None:
        center_colors = ["black"] * estimator.n_clusters

    if center_labels is None:
        center_labels = ['$CENTER: {}$'.format(i) for i in
                         range(estimator.n_clusters)]

    colors_by_cluster = cluster_colors[labels]

    patches = []
    for i in range(estimator.n_clusters):
        patches.append(
            mpatches.Patch(color=cluster_colors[i],
                           label=cluster_labels[i]))

    for j in range(fdatagrid.ndim_image):
        for i in range(fdatagrid.nsamples):
            ax[j].plot(fdatagrid.sample_points[0],
                       fdatagrid.data_matrix[i, :, j],
                       c=colors_by_cluster[i, j],
                       label=sample_labels[i])
        for i in range(estimator.n_clusters):
            ax[j].plot(fdatagrid.sample_points[0],
                       estimator.cluster_centers_.data_matrix[i, :, j],
                       c=center_colors[i], label=center_labels[i])
        ax[j].legend(handles=patches)
        datacursor(formatter='{label}'.format)

    fdatagrid.set_labels(fig, ax)

    return fig, ax


def plot_clusters(estimator, X, fig=None, ax=None, nrows=None, ncols=None,
                  sample_labels=None, cluster_colors=None,
                  cluster_labels=None, center_colors=None, center_labels=None,
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
        ax (list of axis objects): axis over where the graphs are plotted.
            If None, see param fig.
        nrows(int): designates the number of rows of the figure to plot the
            different dimensions of the image. Only specified if fig and
            ax are None.
        ncols(int): designates the number of columns of the figure to plot
            the different dimensions of the image. Only specified if fig
            and ax are None.
        sample_labels (list of str): contains in order the labels of each sample
            of the fdatagrid.
        cluster_colors (list of colors): contains in order the colors of each
            cluster the samples of the fdatagrid are classified into.
        cluster_labels (list of str): contains in order the names of each
            cluster the samples of the fdatagrid are classified into.
        center_colors (list of colors): contains in order the colors of each
            centroid of the clusters the samples of the fdatagrid are classified into.
        center_labels (list of colors): contains in order the labels of each
            centroid of the clusters the samples of the fdatagrid are classified into.
        colormap(colormap): colormap from which the colors of the plot are
            taken. Defaults to `rainbow`.

    Returns:
        (tuple): tuple containing:

            fig (figure object): figure object in which the graphs are plotted in case ax is None.

            ax (axes object): axes in which the graphs are plotted.
    """
    _check_if_estimator(estimator)
    try:
        estimator._check_is_fitted()
        estimator._check_test_data(X)
    except NotFittedError:
        estimator.fit(X)

    if isinstance(estimator, FuzzyKMeans):
        labels = np.argmax(estimator.labels_, axis=-1)
    else:
        labels = estimator.labels_

    return _plot_clusters(estimator=estimator, fdatagrid=X,
                          fig=fig, ax=ax, nrows=nrows, ncols=ncols,
                          labels=labels, sample_labels=sample_labels,
                          cluster_colors=cluster_colors,
                          cluster_labels=cluster_labels,
                          center_colors=center_colors,
                          center_labels=center_labels, colormap=colormap)


def _labels_checks(fdatagrid, xlabels, ylabels, title, xlabel_str):
    """Checks the arguments *xlabels*, *ylabels*, *title* passed to the plot
    functions :func:`plot_cluster_lines
    <skfda.exploratory.visualization.clustering_plots.plot_cluster_lines>` and
    :func:`plot_cluster_bars <skfda.exploratory.visualization.clustering_plots.plot_cluster_bars>`.
    In case they are not set yet, they are given a value.

    Args:
        xlabels (list of str): Labels for the x-axes.
        ylabels (list of str): Labels for the y-axes.
        title (str): Title for the figure where the clustering results are ploted.
        xlabel_str (str): In case xlabels is None, string to use fro the labels
            in the x-axes.

    Returns:
        xlabels (list of str): Labels for the x-axes.
        ylabels (list of str): Labels for the y-axes.
        title (str): Title for the figure where the clustering results are ploted.
    """

    if xlabels is not None and len(xlabels) != fdatagrid.ndim_image:
        raise ValueError(
            "xlabels must contain a label for each dimension on the domain.")

    if ylabels is not None and len(ylabels) != fdatagrid.ndim_image:
        raise ValueError(
            "xlabels must contain a label for each dimension on the domain.")

    if xlabels is None:
        xlabels = [xlabel_str] * fdatagrid.ndim_image

    if ylabels is None:
        ylabels = ["Membership grade"] * fdatagrid.ndim_image

    if title is None:
        title = "Membership grades of the samples to each cluster"

    return xlabels, ylabels, title


def plot_cluster_lines(estimator, X, fig=None, ax=None, nrows=None, ncols=None,
                       sample_colors=None, sample_labels=None,
                       cluster_labels=None,
                       colormap=plt.cm.get_cmap('rainbow'), xlabels=None,
                       ylabels=None, title=None):
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
        fig (figure object, optional): figure over which the graphs are
            plotted in case ax is not specified. If None and ax is also None,
            the figure is initialized.
        ax (list of axis objects, optional): axis over where the graphs are
            plotted. If None, see param fig.
        nrows(int, optional): designates the number of rows of the figure
            to plot the different dimensions of the image. Only specified
            if fig and ax are None.
        ncols(int, optional): designates the number of columns of the figure
            to plot the different dimensions of the image. Only specified if
            fig and ax are None.
        sample_colors (list of colors, optional): contains in order the colors of each
            sample of the fdatagrid.
        sample_labels (list of str, optional): contains in order the labels
            of each sample  of the fdatagrid.
        cluster_labels (list of str, optional): contains in order the names of each
            cluster the samples of the fdatagrid are classified into.
        colormap(colormap, optional): colormap from which the colors of the plot are taken.
        xlabels (list of str, optional): Labels for the x-axes. Defaults to
            ["Cluster"] * fdatagrid.ndim_image.
        ylabels (list of str, optional): Labels for the y-axes. Defaults to
            ["Membership grade"] * fdatagrid.ndim_image.
        title (str, optional): Title for the figure where the clustering results are ploted.
            Defaults to "Membership grades of the samples to each cluster".

    Returns:
        (tuple): tuple containing:

            fig (figure object): figure object in which the graphs are plotted in case ax is None.

            ax (axes object): axes in which the graphs are plotted.

    """
    _check_if_estimator(estimator)

    if not isinstance(estimator, FuzzyKMeans):
        raise ValueError("The estimator must be a FuzzyKMeans object.")

    try:
        estimator._check_is_fitted()
        estimator._check_test_data(X)
    except NotFittedError:
        estimator.fit(X)

    fdatagrid = X

    fig, ax = fdatagrid.generic_plotting_checks(fig, ax, nrows, ncols)

    _plot_clustering_checks(estimator, fdatagrid, sample_colors, sample_labels,
                            None, cluster_labels, None, None)

    xlabels, ylabels, title = _labels_checks(fdatagrid, xlabels, ylabels,
                                             title, "Cluster")

    if sample_colors is None:
        cluster_colors = colormap(np.arange(estimator.n_clusters) /
                                  (estimator.n_clusters - 1))
        labels_by_cluster = np.argmax(estimator.labels_, axis=-1)
        sample_colors = cluster_colors[labels_by_cluster]

    if sample_labels is None:
        sample_labels = ['$SAMPLE: {}$'.format(i) for i in
                         range(fdatagrid.nsamples)]

    if cluster_labels is None:
        cluster_labels = ['${}$'.format(i) for i in
                          range(estimator.n_clusters)]

    for j in range(fdatagrid.ndim_image):
        ax[j].get_xaxis().set_major_locator(MaxNLocator(integer=True))
        for i in range(fdatagrid.nsamples):
            ax[j].plot(np.arange(estimator.n_clusters),
                       estimator.labels_[i, j, :],
                       label=sample_labels[i], color=sample_colors[i, j])
        ax[j].set_xticks(np.arange(estimator.n_clusters))
        ax[j].set_xticklabels(cluster_labels)
        ax[j].set_xlabel(xlabels[j])
        ax[j].set_ylabel(ylabels[j])
        datacursor(formatter='{label}'.format)

    fig.suptitle(title)
    return fig, ax


def plot_cluster_bars(estimator, X, fig=None, ax=None, nrows=None, ncols=None,
                      sort=-1, sample_labels=None, cluster_colors=None,
                      cluster_labels=None, colormap=plt.cm.get_cmap('rainbow'),
                      xlabels=None, ylabels=None, title=None):
    """Implementation of the plotting of the results of the
    :func:`Fuzzy K-Means <fda.clustering.fuzzy_kmeans>` method.


    A kind of barplot is generated in this function with the
    membership values obtained from the algorithm. There is a bar for each sample
    whose height is 1 (the sum of the membership values of a sample add to 1), and
    the part proportional to each cluster is coloured with the corresponding color.
    See `Clustering Example <../auto_examples/plot_clustering.html>`_.

    Args:
        estimator (BaseEstimator object): estimator used to calculate the
            clusters.
        X (FDataGrd object): contains the samples which are grouped
            into different clusters.
        fig (figure object, optional): figure over which the graphs are
            plotted in case ax is not specified. If None and ax is also None,
            the figure is initialized.
        ax (list of axis objects, optional): axis over where the graphs are
            plotted. If None, see param fig.
        nrows(int, optional): designates the number of rows of the figure
            to plot the different dimensions of the image. Only specified
            if fig and ax are None.
        ncols(int, optional): designates the number of columns of the figure
            to plot the different dimensions of the image. Only specified if
            fig and ax are None.
        sort(int, optional): Number in the range [-1, n_clusters) designating
            the cluster whose labels are sorted in a decrementing order.
            Defaults to -1, in this case, no sorting is done.
        sample_labels (list of str, optional): contains in order the labels
            of each sample  of the fdatagrid.
        cluster_labels (list of str, optional): contains in order the names of each
            cluster the samples of the fdatagrid are classified into.
        cluster_colors (list of colors): contains in order the colors of each
            cluster the samples of the fdatagrid are classified into.
        colormap(colormap, optional): colormap from which the colors of the plot are taken.
        xlabels (list of str): Labels for the x-axes. Defaults to
            ["Sample"] * fdatagrid.ndim_image.
        ylabels (list of str): Labels for the y-axes. Defaults to
            ["Membership grade"] * fdatagrid.ndim_image.
        title (str): Title for the figure where the clustering results are ploted.
            Defaults to "Membership grades of the samples to each cluster".

    Returns:
        (tuple): tuple containing:

            fig (figure object): figure object in which the graphs are plotted in case ax is None.

            ax (axes object): axes in which the graphs are plotted.

    """
    _check_if_estimator(estimator)

    if not isinstance(estimator, FuzzyKMeans):
        raise ValueError("The estimator must be a FuzzyKMeans object.")

    try:
        estimator._check_is_fitted()
        estimator._check_test_data(X)
    except NotFittedError:
        estimator.fit(X)

    fdatagrid = X

    fig, ax = fdatagrid.generic_plotting_checks(fig, ax, nrows, ncols)

    if sort < -1 or sort >= estimator.n_clusters:
        raise ValueError(
            "The sorting number must belong to the interval [-1, n_clusters)")

    _plot_clustering_checks(estimator, fdatagrid, None, sample_labels,
                            cluster_colors, cluster_labels, None, None)

    xlabels, ylabels, title = _labels_checks(fdatagrid, xlabels, ylabels,
                                             title, "Sample")

    if sample_labels is None:
        sample_labels = np.arange(fdatagrid.nsamples)

    if cluster_colors is None:
        cluster_colors = colormap(
            np.arange(estimator.n_clusters) / (estimator.n_clusters - 1))

    if cluster_labels is None:
        cluster_labels = ['$CLUSTER: {}$'.format(i) for i in
                          range(estimator.n_clusters)]

    patches = []
    for i in range(estimator.n_clusters):
        patches.append(
            mpatches.Patch(color=cluster_colors[i], label=cluster_labels[i]))

    for j in range(fdatagrid.ndim_image):
        sample_labels_dim = np.copy(sample_labels)
        cluster_colors_dim = np.copy(cluster_colors)
        if sort != -1:
            sample_indices = np.argsort(-estimator.labels_[:, j, sort])
            sample_labels_dim = np.copy(sample_labels[sample_indices])
            labels_dim = np.copy(estimator.labels_[sample_indices, j])

            temp_labels = np.copy(labels_dim[:, 0])
            labels_dim[:, 0] = labels_dim[:, sort]
            labels_dim[:, sort] = temp_labels

            temp_color = np.copy(cluster_colors_dim[0])
            cluster_colors_dim[0] = cluster_colors_dim[sort]
            cluster_colors_dim[sort] = temp_color
        else:
            labels_dim = np.squeeze(estimator.labels_[:, j])

        conc = np.zeros((fdatagrid.nsamples, 1))
        labels_dim = np.concatenate((conc, labels_dim), axis=-1)
        for i in range(estimator.n_clusters):
            ax[j].bar(np.arange(fdatagrid.nsamples),
                      labels_dim[:, i + 1],
                      bottom=np.sum(labels_dim[:, :(i + 1)], axis=1),
                      color=cluster_colors_dim[i])
        ax[j].set_xticks(np.arange(fdatagrid.nsamples))
        ax[j].set_xticklabels(sample_labels_dim)
        ax[j].set_xlabel(xlabels[j])
        ax[j].set_ylabel(ylabels[j])
        ax[j].legend(handles=patches)

    fig.suptitle(title)
    return fig, ax
