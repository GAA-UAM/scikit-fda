"""Diffusion-based functional data generation."""
from typing import TYPE_CHECKING

import lazy_loader as lazy

__getattr__, __dir__, __all__ = lazy.attach(
    __name__,
    submod_attrs={
        "diffusion_model": ["FunctionalDiffusionGenerator"],
        "diffusion_process": [
            "ForwardDiffusionProcess",
            "VariancePreservingDiffusionProcess",
        ],
        "reverse_diffusion": [
            "EulerMaruyamaIntegrator",
            "ProbabilityFlowODEReverseProcess",
            "ReverseDiffusionProcess",
            "RK4Integrator",
            "SDEReverseDiffusionProcess",
        ],
        "score_model": ["ScoreModel", "UNetScoreModel"],
    },
)

if TYPE_CHECKING:
    from .diffusion_model import (
        FunctionalDiffusionGenerator as FunctionalDiffusionGenerator,
    )
    from .diffusion_process import (
        ForwardDiffusionProcess as ForwardDiffusionProcess,
        VariancePreservingDiffusionProcess as VariancePreservingDiffusionProcess,  # noqa: E501
    )
    from .reverse_diffusion import (
        EulerMaruyamaIntegrator as EulerMaruyamaIntegrator,
        ProbabilityFlowODEReverseProcess as ProbabilityFlowODEReverseProcess,
        ReverseDiffusionProcess as ReverseDiffusionProcess,
        RK4Integrator as RK4Integrator,
        SDEReverseDiffusionProcess as SDEReverseDiffusionProcess,
    )
    from .score_model import (
        ScoreModel as ScoreModel,
        UNetScoreModel as UNetScoreModel,
    )
