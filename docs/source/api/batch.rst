Batched bandits
===============

Read more in the :doc:`user guide <../user_guide/batch>`.

Protocol
--------

.. currentmodule:: statrl.settings.bandits.batch

.. autosummary::
   :toctree: generated/
   :nosignatures:

   agent.BatchBanditAgent
   environment.BatchMAB
   interaction.BatchBanditInteraction

Agents
------

.. autosummary::
   :toctree: generated/
   :nosignatures:

   agents.BatchIMED.BatchIMED
   agents.BatchIMED.BatchIMED2
   agents.BIMED.BIMED
   agents.BCB.BCB
   agents.BCB.BCBnaif
   agents.BABA.BABA
   agents._Oracle.Oracle
   agents._Random.Random

Environments
------------

.. autosummary::
   :toctree: generated/
   :nosignatures:

   envs.parametric.BatchBernoulliBandit
   envs.parametric.BatchGaussianBandit
   envs.parametric.BatchTruncatedGaussianBandit

Batch schedules
---------------

A schedule maps a round index to the size of that round's batch. Pass one by
name to the environment factories above, or directly to
:class:`~statrl.settings.bandits.batch.environment.BatchMAB`.

.. autosummary::
   :toctree: generated/
   :nosignatures:

   envs.parametric.baba_schedule
   envs.parametric.exotic_schedule1
   envs.parametric.exotic_schedule2
