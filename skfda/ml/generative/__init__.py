"""Diffusion-based functional data generation."""
from typing import TYPE_CHECKING

import lazy_loader as lazy

__getattr__, __dir__, __all__ = lazy.attach(
    __name__,
    submod_attrs={
        "_diffusion_model": ["FunctionalDiffusionGenerator"],
        "_diffusion_process": [
            "ForwardDiffusionProcess",
            "VariancePreservingDiffusionProcess",
            "VarianceExplodingDiffusionProcess",
            "CirculantSymmetricMatrixDiffusionProcess",
        ],
        "_reverse_diffusion": [
            "EulerMaruyamaIntegrator",
            "ProbabilityFlowODEReverseProcess",
            "ReverseDiffusionProcess",
            "RK4Integrator",
            "SDEReverseDiffusionProcess",
        ],
        "_score_model": ["ScoreModel", "UNetScoreModel"],
    },
)

if TYPE_CHECKING:
    from ._diffusion_model import (
        FunctionalDiffusionGenerator as FunctionalDiffusionGenerator,
    )
    from ._diffusion_process import (
        CirculantSymmetricMatrixDiffusionProcess as CirculantSymmetricMatrixDiffusionProcess,  # noqa: E501
        ForwardDiffusionProcess as ForwardDiffusionProcess,
        VarianceExplodingDiffusionProcess as VarianceExplodingDiffusionProcess,
        VariancePreservingDiffusionProcess as VariancePreservingDiffusionProcess,  # noqa: E501
    )
    from ._reverse_diffusion import (
        EulerMaruyamaIntegrator as EulerMaruyamaIntegrator,
        ProbabilityFlowODEReverseProcess as ProbabilityFlowODEReverseProcess,
        ReverseDiffusionProcess as ReverseDiffusionProcess,
        RK4Integrator as RK4Integrator,
        SDEReverseDiffusionProcess as SDEReverseDiffusionProcess,
    )
    from ._score_model import (
        ScoreModel as ScoreModel,
        UNetScoreModel as UNetScoreModel,
    )
