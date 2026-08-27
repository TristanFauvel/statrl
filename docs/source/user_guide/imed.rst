IMED: Indexed Minimum Empirical Divergence
==========================================

**IMED** is an index-based strategy for stochastic multi-armed bandits, and is
*asymptotically optimal*: its regret matches the [Lai1985]_ lower bound, so no
algorithm can do better by more than a vanishing amount. It achieves this with
**no tuning parameter** — no confidence level, no exploration bonus, no
schedule. See [Honda2015]_.

Core principle
--------------

Each arm :math:`a` is given the index

.. math::

   I_a(t) = N_a(t)\,\mathrm{kl}\!\left(\hat{\mu}_a(t), \hat{\mu}^\star(t)\right)
            + \log N_a(t),

where :math:`N_a(t)` is the number of pulls of arm :math:`a`,
:math:`\hat{\mu}_a(t)` its empirical mean, :math:`\hat{\mu}^\star(t) =
\max_a \hat{\mu}_a(t)` the best empirical mean, and :math:`\mathrm{kl}` a
divergence of the reward family. The arm played is

.. math::

   a_t = \arg\min_a I_a(t).

Why this balances exploration and exploitation
----------------------------------------------

Read the two terms separately.

The first, :math:`N_a \,\mathrm{kl}(\hat{\mu}_a, \hat{\mu}^\star)`, is large
when the data confidently place arm ``a`` below the leader — it is, up to a
constant, the log-likelihood ratio against ``a`` being optimal. An arm that has
been sampled a lot *and* looks clearly worse gets a large index and is dropped.

The second, :math:`\log N_a`, grows with the pull count alone. It is what keeps
the leader itself from being played forever: since the leader's divergence term
is zero, its index is exactly :math:`\log N_a`, which rises every time it is
pulled until some other arm's index falls below it.

Minimizing the sum therefore favours arms that are either under-sampled or not
yet statistically distinguishable from the best — which is precisely the set an
optimal algorithm must keep testing.

Two consequences worth knowing:

- **Untried arms are played first.** An arm with :math:`N_a = 0` is given index
  ``0``, the smallest the index can take, so every arm is pulled once before any
  is repeated.
- **Ties are broken at random.** :func:`~statrl.settings.utils.randmin` picks
  uniformly among minimizers. This matters at the start of a run, when all
  indexes are still ``0``; :func:`numpy.argmin` would always return arm 0.

Choosing the divergence
-----------------------

IMED is asymptotically optimal *for the divergence of the true reward family*.
Pass the matching one:

.. list-table::
   :header-rows: 1
   :widths: 30 30 40

   * - Rewards
     - Divergence
     - Note
   * - Bernoulli, or bounded in :math:`[0, 1]`
     - :func:`~statrl.settings.utils.klBern`
     - The sharpest choice for binary rewards.
   * - Gaussian, or :math:`\sigma^2`-sub-Gaussian
     - :func:`~statrl.settings.utils.klGauss`
     - The default. Also a valid, looser choice for bounded rewards.
   * - Poisson
     - :func:`~statrl.settings.utils.klPoisson`
     -
   * - Exponential
     - :func:`~statrl.settings.utils.klExp`
     -
   * - Unknown, but with a known bound
     - :func:`~statrl.settings.utils.KLinf_threshold`
     - Non-parametric; used by the batched variants, see :doc:`batch`.

A mismatched divergence stays well defined and the algorithm still runs — it
simply loses the optimality guarantee. Comparing two choices on one instance is
a one-line experiment:

.. code-block:: python

   agents = [IMED(nA, klBern, name="IMED-Bern"), IMED(nA, klGauss, name="IMED-Gauss")]

Give them distinct ``name`` values, or their results collide in the dump
filenames and plot legend.

Using it
--------

.. doctest::

   >>> from statrl.settings.bandits.stochastic.anytime.envs.parametric import BernoulliBandit
   >>> from statrl.settings.bandits.stochastic.anytime.agents.IMED import IMED
   >>> from statrl.settings.bandits.stochastic.anytime.interaction import BanditInteraction
   >>> from statrl.settings.utils import klBern
   >>>
   >>> env = BernoulliBandit([0.2, 0.9, 0.5])
   >>> agent = IMED(env.number_arms, kullback=klBern)
   >>> scores = BanditInteraction().run(env, agent, horizon=2000)
   >>> int(agent.nbDraws.argmax())        # concentrates on the best arm
   1

Cost and assumptions
--------------------

Both a pull and an update cost :math:`O(K)` time, and the agent holds
:math:`O(K)` state. The update recomputes *every* index rather than only the
pulled arm's, because they all share :math:`\hat{\mu}^\star(t)`.

IMED assumes stationary rewards drawn independently at each pull. It is
sensitive to the numerical stability of the divergence near the boundary, which
is why :func:`~statrl.settings.utils.klBern` clips its arguments into
:math:`[\varepsilon, 1-\varepsilon]`.

Variants
--------

- :class:`~statrl.settings.bandits.stochastic.batch.agents.BIMED.BIMED` — batched
  and distribution-free, using :math:`K_{\inf}` in place of a parametric
  divergence. See :doc:`batch`.
- :class:`~statrl.settings.markovdecisionprocess.discrete_nostructure.agents.IMED_RL.IMEDRL`
  — the same index carried to ergodic MDPs, with the divergence taken over
  reward *plus next-state bias*. See :doc:`mdp`.

See :class:`~statrl.settings.bandits.stochastic.anytime.agents.IMED.IMED` in the
API reference for the full parameter and attribute list.
