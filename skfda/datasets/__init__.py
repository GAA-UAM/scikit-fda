from ._real_datasets import (fdata_constructor, fetch_cran,
                             fetch_ucr,
                             fetch_phoneme, fetch_growth,
                             fetch_tecator, fetch_medflies,
                             fetch_weather, fetch_aemet)
from ._samples_generators import (make_gaussian_process,
                                  make_sinusoidal_process,
                                  make_multimodal_samples,
                                  make_multimodal_landmarks,
                                  make_random_warping)
