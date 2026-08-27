Known horizon
=============

Read more in the :doc:`user guide <../user_guide/stochastic_knownhorizon>`.

Protocol
--------

.. currentmodule:: statrl.settings.bandits.stochastic.knownhorizon

.. autosummary::
   :toctree: generated/
   :nosignatures:

   agent.BanditAgent
   interaction.BanditInteraction

The environment is
:class:`~statrl.settings.bandits.stochastic.anytime.environment.StochasticBanditEnv`,
re-exported unchanged as
``statrl.settings.bandits.stochastic.knownhorizon.environment.StochasticBanditEnv``.
Knowing the horizon changes what the *agent* may do, not what the problem is, so
every environment of the anytime setting works here as is.

Wrappers
--------

Adapters between this setting and the anytime one, letting agents of both be
compared in a single experiment.

.. autosummary::
   :toctree: generated/
   :nosignatures:

   wrappers.wrapper_anytime_knownhorizon.AnytimeToKnownHorizonAgentWrapper
   wrappers.wrapper_anytime_knownhorizon.KnownHorizonToAnytimeAgentWrapper
