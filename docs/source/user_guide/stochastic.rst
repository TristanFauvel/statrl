Stochastic bandits
==================

*Module:* ``statrl.settings.bandits.stochastic``

In the **stochastic** setting each arm has a fixed, unknown reward distribution, and
pulling an arm draws an independent sample from it. ``statrl`` ships two variants,
distinguished by what the agent knows about the time horizon:

Every stochastic-bandit agent implements the same three methods:

``reset()``
   Start a new, independent run (clear all statistics).

``select_arm() -> int``
   Return the index of the arm to pull next.

``update(arm, reward)``
   Incorporate the observed reward for the chosen arm.

An environment exposes its arms through ``number_arms`` and ``means`` (the latter
for evaluation and oracle construction only) and samples rewards through
``step(arm)``. The loop ``BanditInteraction().run(env, learner, horizon)`` ties them
together and returns a cumulative-score :class:`numpy.ndarray` — of the arms'
*expected* rewards, not the realized ones, so the regret curve carries no reward
noise. See :doc:`../core_concepts`.


.. toctree::
   :maxdepth: 1

   stochastic_anytime
   stochastic_knownhorizon
