"""Metrics, norms and related utilities."""

from ._angular import angular_distance
from ._fisher_rao import (
    _fisher_rao_warping_distance,
    fisher_rao_amplitude_distance,
    fisher_rao_distance,
    fisher_rao_phase_distance,
)
from ._lp_distances import (
    LpDistance,
    l1_distance,
    l2_distance,
    linf_distance,
    lp_distance,
)
from ._lp_norms import LpNorm, l1_norm, l2_norm, linf_norm, lp_norm
from ._typing import PRECOMPUTED, Metric, Norm
from ._utils import (
    NormInducedMetric,
    PairwiseMetric,
    TransformationMetric,
    pairwise_metric_optimization,
)
