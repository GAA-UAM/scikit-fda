"""Statistics."""

from typing import TYPE_CHECKING

import lazy_loader as lazy

__getattr__, __dir__, __all__ = lazy.attach(
    __name__,
    submod_attrs={
        "_fisher_rao": [
            "_fisher_rao_warping_mean",
            "fisher_rao_karcher_mean",
        ],
        "_functional_transformers": [
            "local_averages",
            "number_crossings",
            "occupation_measure",
            "unconditional_central_moment",
            "unconditional_expected_value",
            "unconditional_moment",
        ],
        "_stats": [
            "cov",
            "depth_based_median",
            "geometric_median",
            "gmean",
            "mean",
            "modified_epigraph_index",
            "trim_mean",
            "var",
        ],
    },
)

if TYPE_CHECKING:
    from ._fisher_rao import (
        _fisher_rao_warping_mean as _fisher_rao_warping_mean,
        fisher_rao_karcher_mean as fisher_rao_karcher_mean,
    )
    from ._functional_transformers import (
        local_averages as local_averages,
        number_crossings as number_crossings,
        occupation_measure as occupation_measure,
        unconditional_central_moment as unconditional_central_moment,
        unconditional_expected_value as unconditional_expected_value,
        unconditional_moment as unconditional_moment,
    )
    from ._stats import (
        cov as cov,
        depth_based_median as depth_based_median,
        geometric_median as geometric_median,
        gmean as gmean,
        mean as mean,
        modified_epigraph_index as modified_epigraph_index,
        trim_mean as trim_mean,
        var as var,
    )
