"""Clustering."""

from typing import TYPE_CHECKING

import lazy_loader as lazy

__getattr__, __dir__, __all__ = lazy.attach(
    __name__,
    submod_attrs={
        "_hierarchical": ["AgglomerativeClustering"],
        "_kmeans": ["FuzzyCMeans", "KMeans"],
        "_neighbors_clustering": ["NearestNeighbors"],
    },
)

if TYPE_CHECKING:
    from ._hierarchical import (
        AgglomerativeClustering as AgglomerativeClustering,
    )
    from ._kmeans import FuzzyCMeans as FuzzyCMeans, KMeans as KMeans
    from ._neighbors_clustering import NearestNeighbors as NearestNeighbors
