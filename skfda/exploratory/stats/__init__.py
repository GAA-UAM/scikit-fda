"""Statistics."""
from ._fisher_rao import _fisher_rao_warping_mean, fisher_rao_karcher_mean
from ._functional_transformers import (
    local_averages,
    number_up_crossings,
    occupation_measure,
)
from ._stats import (
    cov,
    depth_based_median,
    geometric_median,
    gmean,
    mean,
    modified_epigraph_index,
    trim_mean,
    var,
)
