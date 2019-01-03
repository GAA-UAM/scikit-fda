"""Graphics Module to visualize FDataGrid."""
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from sklearn.covariance import MinCovDet
from scipy.stats import variation
from scipy import stats
import math

from .depth_measures import *
from .grid import FDataGrid

__author__ = "Amanda Hernando Bernabé"
__email__ = "amanda.hernando@estudiante.uam.es"


def fdboxplot(fdgrid, fig=None, method=modified_band_depth, prob=[0.5], fullout=False, factor=1.5, barcol="blue",
              colormap=plt.cm.get_cmap('RdPu'), outliercol="red"):
    """Implementation of the functional boxplot.

    Args:
        fdgrid (FDataGrid): Object to be visualized.
		fig (figure object, optional): figure where the graphs are plotted.
			Defaults to matplotlib.pyplot.figure.
        method (depth function, optional): Method used to order the data (FM_depth, band_depth, modified_band_depth).
			Defaults to modified_band_depth.
        prob (list of float, optional): List with float numbers (in the range from 1 to 0) that indicate which central regions to
			represent. Defaults to [0.5] which represents the 0.5 central region.
		fullout (boolean): If true, the entire curve of the outlier samples is shown.
		factor (double): Number used to calculate the outlying envelope.
        colormap (matplotlib.colors.LinearSegmentedColormap): Colormap from which the colors to represent the central regions are selected.
        outliercol (string): Color of the outliers.
        barcol (string): Color of the envelopes and vertical lines.

    Returns:
        _plot: List of mpl_toolkits.mplot3d.art3d.Poly3DCollection, mpl_toolkits.mplot3d.art3d.Line3DCollection and
                mpl_toolkits.mplot3d.art3d.Line3D.

    """
    if fdgrid.ndim_domain > 1:
        raise ValueError("Function only supports FDataGrid with domain dimension 1.")

    if ((method != FM_depth) & (method != band_depth) & (method != modified_band_depth)):
        raise ValueError("Undefined function.")

    if sorted(prob, reverse=True) != prob:
        raise ValueError("Probabilities required to be in descending order.")

    if min(prob) < 0 or max(prob) > 1:
        raise ValueError("Probabilities must be between 0 and 1.")

    if not isinstance(colormap, matplotlib.colors.LinearSegmentedColormap):
        raise ValueError("colormap must be of type matplotlib.colors.LinearSegmentedColormap")

    # Obtaining the necessary colors of the colormap.
    tones = np.linspace(0.1, 1.0, len(prob) + 1, endpoint=False)[1:]
    color = colormap(tones)

    if fullout:
        var_zorder = 4
    else:
        var_zorder = 1

    depth = method(fdgrid)
    indices_descencing_depth = (-depth).argsort(axis=0)

    if fig == None:
        fig, ax = fdgrid.set_figure_and_axes()

    _plot = []

    for m in range(fdgrid.ndim_image):

        for i in range(len(prob)):

            indices_samples = indices_descencing_depth[:, m][:math.ceil(fdgrid.nsamples * prob[i])]
            samples_used = fdgrid.data_matrix[indices_samples, :, m]
            max_samples_used = np.amax(samples_used, axis=0)
            min_samples_used = np.amin(samples_used, axis=0)

            if prob[i] == 0.5:
                # central envelope
                _plot.append(ax[m].plot(fdgrid.sample_points[0], max_samples_used.T, fdgrid.sample_points[0],
                           min_samples_used.T, color=barcol, zorder=4))

                # outlying envelope
                max_value = np.amax(fdgrid.data_matrix[:, :, m], axis=0)
                min_value = np.amin(fdgrid.data_matrix[:, :, m], axis=0)
                iqr = np.absolute(max_samples_used - min_samples_used)
                oulying_max_envelope = np.minimum(max_samples_used + iqr * factor, max_value)
                oulying_min_envelope = np.maximum(min_samples_used - iqr * factor, min_value)
                _plot.append(ax[m].plot(fdgrid.sample_points[0], oulying_max_envelope.flatten(),
                           fdgrid.sample_points[0], oulying_min_envelope.flatten(), color=barcol, zorder=4))

                # vertical lines
                index = math.ceil(fdgrid.ncol / 2)
                x = fdgrid.sample_points[0][index]
                _plot.append(ax[m].plot([x, x], [oulying_max_envelope[index], max_samples_used[index]], color=barcol, zorder=4))
                _plot.append(ax[m].plot([x, x], [oulying_min_envelope[index], min_samples_used[index]], color=barcol, zorder=4))

                # outliers
                outliers = np.zeros(fdgrid.nsamples)
                for j in list(range(fdgrid.nsamples)):
                    outliers_above = (oulying_max_envelope < fdgrid.data_matrix[j, :, m])
                    outliers_below = (oulying_min_envelope > fdgrid.data_matrix[j, :, m])
                    if (outliers_above.sum() > 0 or outliers_below.sum() > 0):
                        outliers[j] = 1
                        _plot.append(ax[m].plot(fdgrid.sample_points[0], fdgrid.data_matrix[j, :, m], color=outliercol,
                                   linestyle='--', zorder=1))
            # central regions
            _plot.append(ax[m].fill_between(fdgrid.sample_points[0], max_samples_used.flatten(),
                               min_samples_used.flatten(), facecolor=color[i], zorder=var_zorder))

        # mean sample
        _plot.append(ax[m].plot(fdgrid.sample_points[0], fdgrid.data_matrix[indices_descencing_depth[0, m], :, m].T, color="black",
                   zorder=5))

        fdgrid.set_labels(fig)
        fdgrid.arrange_layout(fig)

        return _plot, outliers

def surface_boxplot(fdgrid, fig=None, method=modified_band_depth, factor=1.5, boxcol="black", outcol="grey"):
    """Implementation of the surface boxplot.

    Args:
        fdgrid (FDataGrid): Object to be visualized.
		fig (figure object, optional): aigure where the graphs are plotted.
			Defaults to matplotlib.pyplot.figure.
        method (depth function, optional): Method used to order the data (FM_depth, band_depth, modified_band_depth).
			Defaults to modified_band_depth.
		factor (double): Number used to calculate the outlying envelope.
        boxcol (string): Color of the box: mean, central envelopes and vertical lines.
        outboxcol (string): Color of the outlying envelopes.

    Returns:
        _plot: List of mpl_toolkits.mplot3d.art3d.Poly3DCollection, mpl_toolkits.mplot3d.art3d.Line3DCollection and
                mpl_toolkits.mplot3d.art3d.Line3D.

    """
    if fdgrid.ndim_domain != 2:
        raise ValueError("Function only supports FDataGrid with domain dimension 2.")

    if ((method != FM_depth) & (method != band_depth) & (method != modified_band_depth)):
        raise ValueError("Undefined function.")

    depth = method(fdgrid)
    indices_descencing_depth = (-depth).argsort(axis=0)

    if fig == None:
        fig, ax = fdgrid.set_figure_and_axes()

    x = fdgrid.sample_points[0]
    lx = len(x)
    y = fdgrid.sample_points[1]
    ly = len(y)
    X, Y = np.meshgrid(x, y)

    _plot = []

    for m in range(fdgrid.ndim_image):

        indices_samples = indices_descencing_depth[:, m][:math.ceil(fdgrid.nsamples * 0.5)]
        samples_used = fdgrid.data_matrix[indices_samples, :, :, m]
        max_samples_used = np.amax(samples_used, axis=0)
        min_samples_used = np.amin(samples_used, axis=0)

        # mean sample
        _plot.append(
            ax[m].plot_wireframe(X, Y, np.squeeze(fdgrid.data_matrix[indices_descencing_depth[0, m], :, :, m]).T,
                                 rstride=ly, cstride=lx, color=boxcol))
        _plot.append(ax[m].plot_surface(X, Y, np.squeeze(fdgrid.data_matrix[indices_descencing_depth[0, m], :, :, m]).T,
                                        color=boxcol, alpha=0.8))

        # central envelope
        _plot.append(ax[m].plot_surface(X, Y, np.squeeze(max_samples_used).T, color=boxcol, alpha=0.5))
        _plot.append(ax[m].plot_wireframe(X, Y, np.squeeze(max_samples_used).T, rstride=ly, cstride=lx, color=boxcol))
        _plot.append(ax[m].plot_surface(X, Y, np.squeeze(min_samples_used).T, color=boxcol, alpha=0.5))
        _plot.append(ax[m].plot_wireframe(X, Y, np.squeeze(min_samples_used).T, rstride=ly, cstride=lx, color=boxcol))

        # box vertical lines
        for indices in [(0, 0), (0, ly - 1), (lx - 1, 0), (lx - 1, ly - 1)]:
            x_corner = fdgrid.sample_points[0][indices[0]]
            y_corner = fdgrid.sample_points[1][indices[1]]
            _plot.append(ax[m].plot([x_corner, x_corner], [y_corner, y_corner],
                                    [min_samples_used[indices[0], indices[1]],
                                     max_samples_used[indices[0], indices[1]]], color=boxcol))

        # outlying envelope
        max_value = np.amax(fdgrid.data_matrix[:, :, :, m], axis=0)
        min_value = np.amin(fdgrid.data_matrix[:, :, :, m], axis=0)
        iqr = np.absolute(max_samples_used - min_samples_used)
        oulying_max_envelope = np.minimum(max_samples_used + iqr * factor, max_value)
        oulying_min_envelope = np.maximum(min_samples_used - iqr * factor, min_value)
        _plot.append(ax[m].plot_surface(X, Y, np.squeeze(oulying_max_envelope).T, color=outcol, alpha=0.3))
        _plot.append(
            ax[m].plot_wireframe(X, Y, np.squeeze(oulying_max_envelope).T, rstride=ly, cstride=lx, color=outcol))
        _plot.append(ax[m].plot_surface(X, Y, np.squeeze(oulying_min_envelope).T, color=outcol, alpha=0.3))
        _plot.append(
            ax[m].plot_wireframe(X, Y, np.squeeze(oulying_min_envelope).T, rstride=ly, cstride=lx, color=outcol))

        # vertical lines from central to outlying envelope
        x_index = math.floor(lx / 2)
        x_central = x[x_index]
        y_index = math.floor(ly / 2)
        y_central = y[y_index]
        _plot.append(ax[m].plot([x_central, x_central], [y_central, y_central],
                                [oulying_max_envelope[x_index, y_index], max_samples_used[x_index, y_index]],
                                color=boxcol))
        _plot.append(ax[m].plot([x_central, x_central], [y_central, y_central],
                                [oulying_min_envelope[x_index, y_index], min_samples_used[x_index, y_index]],
                                color=boxcol))

    fdgrid.set_labels(fig)
    fdgrid.arrange_layout(fig)

    return _plot


def MS_plot(fdgrid, depth_method, dim_weights=None, pointwise_weights=None, ax=None):

    if fdgrid.ndim_image > 1:
        raise NotImplementedError("Only support 1 dimension on the image.")

    # The depths of the samples are calculated giving them an ordering.
    outlyingness = directional_outlyingness(fdgrid, depth_method, dim_weights, pointwise_weights)
    mean_dir_outl, variation_dir_outl = outlyingness[1], outlyingness[2]
    X = np.array(list(zip(mean_dir_outl, variation_dir_outl)))
    cov = MinCovDet(store_precision=True).fit(X)
    rbd_2 = cov.mahalanobis(X)
    s_jj = np.diag(cov.covariance_)
    c = np.mean(s_jj)
    m = 2 / np.square(variation(s_jj))
    p = fdgrid.ndim_image
    dfn = p + 1
    dfd = m - p
    q = 0.993
    cutoff_value = stats.f.ppf(q, dfn, dfd, loc=0, scale=1)
    scaling = c * (m - p) / m / (p + 1)
    outliers = (scaling * rbd_2 > cutoff_value)
    outliers = outliers.astype(str)
    outliers[np.where(outliers == 'True')] = 'red'
    outliers[np.where(outliers == 'False')] = 'blue'

    if ax is None:
        ax = matplotlib.pyplot.gca()
    ax.scatter(mean_dir_outl, variation_dir_outl, color=outliers)
    ax.set_xlabel('MO')
    ax.set_ylabel('VO')
    ax.set_title('MS-Plot')

    return X


def clustering(fdgrid, ax=None, num_clusters=2):
    if fdgrid.ndim_image > 1 and fdgrid.ndim_domain:
        raise NotImplementedError("Only support 1 dimension on the image and on the domain.")

    if num_clusters < 2 or num_clusters > 10:
        raise ValueError("The number of clusters must be between 2 and 10, both included.")

    repetitions = 0
    distances_to_centers = np.empty((fdgrid.nsamples, num_clusters))
    centers = np.empty((num_clusters, len(fdgrid.sample_points[0])))
    centers_aux = np.empty((num_clusters, len(fdgrid.sample_points[0])))

    for i in range(num_clusters):
        centers[i] = fdgrid.data_matrix[math.floor(i * fdgrid.nsamples / num_clusters)].flatten()

    while not np.array_equal(centers, centers_aux) and repetitions < 100:
        centers_aux = centers
        for i in range(fdgrid.nsamples):
            for j in range(num_clusters):
                distances_to_centers[i, j] = scipy.spatial.distance.euclidean(fdgrid.data_matrix[i], centers[j])
        clustering = np.argmin(distances_to_centers, axis=1)
        for i in range(num_clusters):
            indices = np.where(clustering == i)
            centers[i] = np.average(fdgrid.data_matrix[indices, :, :].flatten()
                                    .reshape((len(indices[0]), len(fdgrid.sample_points[0]))), axis=0)
        repetitions += 1

    colors_samples = np.empty(fdgrid.nsamples).astype(str)
    for i in range(num_clusters):
        colors_samples[np.where(clustering == i)] = "C{}".format(i)

    if ax is None:
        ax = matplotlib.pyplot.gca()

    for i in range(fdgrid.nsamples):
        ax.plot(fdgrid.sample_points[0], fdgrid.data_matrix[i, :, 0].T, color=colors_samples[i])

    return clustering