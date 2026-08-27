Experiments
===========

Read more in the :doc:`user guide <../user_guide/experiments>`.

Entry point
-----------

.. currentmodule:: statrl.experiments

.. autosummary::
   :toctree: generated/
   :nosignatures:

   massiveruns.runLargeMulticoreExperiment

Pipeline
--------

The stages :func:`~statrl.experiments.massiveruns.runLargeMulticoreExperiment`
chains together. Call them directly only to reuse one stage on its own.

.. autosummary::
   :toctree: generated/
   :nosignatures:

   onerun.Interaction
   parallelruns.multicoreRuns
   onerun.oneRunWithDump
   analyzeruns.computeScoreDiffs
   plotruns.plotScoreDiffs

Persistence
-----------

.. autosummary::
   :toctree: generated/
   :nosignatures:

   utils.dump
   utils.clear_auxiliaryfiles
   utils.load
   utils.make
