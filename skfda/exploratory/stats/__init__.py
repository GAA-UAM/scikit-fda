"""Statistics."""

from typing import TYPE_CHECKING

import lazy_loader as lazy

__getattr__, __dir__, __all__ = lazy.attach(
    __name__,
    submodules=["covariance"],
    submod_attrs={
        "_fisher_rao": [
            "_fisher_rao_warping_mean",
            "fisher_rao_karcher_mean",
        ],
        "_stats": [
            "cov",
            "depth_based_median",
            "geometric_median",
            "gmean",
            "mean",
            "modified_epigraph_index",
            "std",
            "trim_mean",
            "var",
            "individual_observation_mean",
            "grand_mean",
            "root_integrated_sample_variance",
        ],
    },
)

if TYPE_CHECKING:
    from ._fisher_rao import (
        _fisher_rao_warping_mean as _fisher_rao_warping_mean,
        fisher_rao_karcher_mean as fisher_rao_karcher_mean,
    )
    from ._stats import (
        cov as cov,
        depth_based_median as depth_based_median,
        geometric_median as geometric_median,
        gmean as gmean,
        grand_mean as grand_mean,
        individual_observation_mean as individual_observation_mean,
        mean as mean,
        modified_epigraph_index as modified_epigraph_index,
        root_integrated_sample_variance as root_integrated_sample_variance,
        std as std,
        trim_mean as trim_mean,
        var as var,
    )
