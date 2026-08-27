User guide
==========

``statrl`` organises reinforcement-learning problems as a **taxonomy of
settings**. Each setting is a self-contained package under ``statrl.settings``
defining three things:

- an **environment** — the problem (arms and reward distributions, or states,
  actions, and a transition kernel),
- an **agent** — the learning algorithm, and
- an **interaction loop** — the function running an agent against an
  environment for a fixed horizon and recording performance.

New here? Read :doc:`../core_concepts` first: it explains the protocol every
setting shares, and how regret is defined.

Choosing a setting
------------------

.. list-table::
   :header-rows: 1
   :widths: 34 40 26

   * - If your problem is...
     - Use
     - Reference algorithm
   * - Fixed unknown reward distributions, unknown horizon
     - :doc:`stochastic_anytime`
     - IMED
   * - The same, but the horizon is known in advance
     - :doc:`stochastic_knownhorizon`
     - IMED, via a wrapper
   * - Actions committed in blocks, feedback delayed
     - :doc:`batch`
     - BIMED
   * - A continuous action space with adversarial rewards
     - :doc:`adversarial_lipschitz`
     - ALF
   * - Actions that move a state
     - :doc:`mdp`
     - IMED-RL, PSRL

Settings
--------

.. toctree::
   :maxdepth: 2

   stochastic
   batch
   adversarial_lipschitz
   mdp

Running experiments
-------------------

.. toctree::
   :maxdepth: 1

   experiments
