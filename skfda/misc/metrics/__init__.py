"""Metrics, norms and related utilities."""

from typing import TYPE_CHECKING

import lazy_loader as lazy

__getattr__, __dir__, __all__ = lazy.attach(
    __name__,
    submod_attrs={
        "_angular": ["angular_distance"],
        "_fisher_rao": [
            "_fisher_rao_warping_distance",
            "fisher_rao_amplitude_distance",
            "fisher_rao_distance",
            "fisher_rao_phase_distance",
        ],
        "_lp_distances": [
            "LpDistance",
            "l1_distance",
            "l2_distance",
            "linf_distance",
            "lp_distance",
        ],
        "_lp_norms": [
            "LpNorm",
            "l1_norm",
            "l2_norm",
            "linf_norm",
            "lp_norm",
        ],
        "_mahalanobis": ["MahalanobisDistance"],
        "_parse": ["PRECOMPUTED"],
        "_utils": [
            "NormInducedMetric",
            "PairwiseMetric",
            "TransformationMetric",
            "pairwise_metric_optimization",
        ],
    },
)

if TYPE_CHECKING:
    from ._angular import angular_distance as angular_distance
    from ._fisher_rao import (
        _fisher_rao_warping_distance as _fisher_rao_warping_distance,
        fisher_rao_amplitude_distance as fisher_rao_amplitude_distance,
        fisher_rao_distance as fisher_rao_distance,
        fisher_rao_phase_distance as fisher_rao_phase_distance,
    )
    from ._lp_distances import (
        LpDistance as LpDistance,
        l1_distance as l1_distance,
        l2_distance as l2_distance,
        linf_distance as linf_distance,
        lp_distance as lp_distance,
    )
    from ._lp_norms import (
        LpNorm as LpNorm,
        l1_norm as l1_norm,
        l2_norm as l2_norm,
        linf_norm as linf_norm,
        lp_norm as lp_norm,
    )
    from ._mahalanobis import MahalanobisDistance as MahalanobisDistance
    from ._parse import PRECOMPUTED as PRECOMPUTED
    from ._utils import (
        NormInducedMetric as NormInducedMetric,
        PairwiseMetric as PairwiseMetric,
        TransformationMetric as TransformationMetric,
        pairwise_metric_optimization as pairwise_metric_optimization,
    )
