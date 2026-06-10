.. _generative-module:

Generative
==========

Module with classes to perform generative modelling of functional data
using diffusion processes.

Diffusion Process
-----------------

.. autosummary::
   :toctree: autosummary

   skfda.ml.generative.ForwardDiffusionProcess
   skfda.ml.generative.VariancePreservingDiffusionProcess
   skfda.ml.generative.VarianceExplodingDiffusionProcess
   skfda.ml.generative.CirculantSymmetricMatrixDiffusionProcess

Reverse Diffusion
-----------------

.. autosummary::
   :toctree: autosummary

   skfda.ml.generative.ReverseDiffusionProcess
   skfda.ml.generative.SDEReverseDiffusionProcess
   skfda.ml.generative.ProbabilityFlowODEReverseProcess
   skfda.ml.generative.EulerMaruyamaIntegrator
   skfda.ml.generative.RK4Integrator

Score Model
-----------

.. autosummary::
   :toctree: autosummary

   skfda.ml.generative.ScoreModel
   skfda.ml.generative.UNetScoreModel

Diffusion Generator
-------------------

.. autosummary::
   :toctree: autosummary

   skfda.ml.generative.FunctionalDiffusionGenerator
