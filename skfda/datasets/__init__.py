from typing import TYPE_CHECKING

import lazy_loader as lazy

__getattr__, __dir__, __all__ = lazy.attach(
    __name__,
    submod_attrs={
        "_real_datasets": [
            "fdata_constructor",
            "fetch_aemet",
            "fetch_cran",
            "fetch_gait",
            "fetch_growth",
            "fetch_handwriting",
            "fetch_mco",
            "fetch_medflies",
            "fetch_nox",
            "fetch_octane",
            "fetch_phoneme",
            "fetch_tecator",
            "fetch_ucr",
            "fetch_weather",
        ],
        "_samples_generators": [
            "make_gaussian",
            "make_gaussian_process",
            "make_multimodal_landmarks",
            "make_multimodal_samples",
            "make_random_warping",
            "make_sinusoidal_process",
        ],
    },
)

if TYPE_CHECKING:
    from ._real_datasets import (
        fdata_constructor as fdata_constructor,
        fetch_aemet as fetch_aemet,
        fetch_cran as fetch_cran,
        fetch_gait as fetch_gait,
        fetch_growth as fetch_growth,
        fetch_handwriting as fetch_handwriting,
        fetch_mco as fetch_mco,
        fetch_medflies as fetch_medflies,
        fetch_nox as fetch_nox,
        fetch_octane as fetch_octane,
        fetch_phoneme as fetch_phoneme,
        fetch_tecator as fetch_tecator,
        fetch_ucr as fetch_ucr,
        fetch_weather as fetch_weather,
    )
    from ._samples_generators import (
        make_gaussian as make_gaussian,
        make_gaussian_process as make_gaussian_process,
        make_multimodal_landmarks as make_multimodal_landmarks,
        make_multimodal_samples as make_multimodal_samples,
        make_random_warping as make_random_warping,
        make_sinusoidal_process as make_sinusoidal_process,
    )
