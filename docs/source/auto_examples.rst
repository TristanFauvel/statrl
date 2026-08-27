Examples
========

Runnable scripts under ``examples/`` in the repository. Each writes a logfile
and regret figures to ``results/``:

.. code-block:: bash

   python examples/plot_bandit_regret.py

Comparing bandit algorithms
---------------------------

The shape of every experiment in ``statrl``: an environment, the agents to
compare, an oracle to define zero regret, and one call to the harness. Also
shows how the choice of divergence changes IMED's behaviour on the same
instance.

.. literalinclude:: ../../examples/plot_bandit_regret.py
   :language: python
   :linenos:

Writing your own agent
----------------------

A greedy agent and an epsilon-greedy one, benchmarked against IMED. The
interesting part is what greedy gets wrong, and how that failure shows up in the
quantile bands of the figure rather than in the mean curve.

.. literalinclude:: ../../examples/custom_agent.py
   :language: python
   :linenos:

See also
--------

Short, self-contained snippets live in the docstrings themselves, under
``Examples``; they run in CI, so they cannot drift from the code. The
:doc:`quickstart` is the shortest path from install to a regret number.
