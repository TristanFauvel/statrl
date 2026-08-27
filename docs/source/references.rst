Bibliography
============

Papers the shipped algorithms and environments come from. Each is also cited in
the ``References`` section of the corresponding class docstring.

Algorithms
----------

.. [Honda2015] Honda, J. and Takemura, A.
   "Non-asymptotic analysis of a new bandit algorithm for semi-bounded rewards."
   *Journal of Machine Learning Research*, 16(113):3721-3756, 2015.

   Introduces IMED and the non-parametric :math:`K_{\inf}`. Implemented by
   :class:`~statrl.settings.bandits.stochastic.anytime.agents.IMED.IMED`,
   :func:`~statrl.settings.utils.KLinf_threshold`, and the batched variant
   :class:`~statrl.settings.bandits.stochastic.batch.agents.BIMED.BIMED`.

.. [Pesquerel2022] Pesquerel, F. and Maillard, O.-A.
   "IMED-RL: Regret optimal learning of ergodic Markov decision processes."
   *Advances in Neural Information Processing Systems (NeurIPS)*, 2022.

   Extends the IMED index to ergodic MDPs. Implemented by
   :class:`~statrl.settings.markovdecisionprocess.discrete_nostructure.agents.IMED_RL.IMEDRL`.

.. [Osband2013] Osband, I., Russo, D. and Van Roy, B.
   "(More) efficient reinforcement learning via posterior sampling."
   *Advances in Neural Information Processing Systems (NeurIPS)*, 2013.

   Posterior sampling for reinforcement learning. Implemented by
   :class:`~statrl.settings.markovdecisionprocess.discrete_nostructure.agents.PSRL.PSRL`.

.. [Maillard2010] Maillard, O.-A. and Munos, R.
   "Online learning in adversarial Lipschitz environments."
   *European Conference on Machine Learning and Knowledge Discovery in
   Databases (ECML PKDD)*, 305-320, 2010.

   Discretization plus exponential weights over a continuous metric space.
   Implemented by
   :class:`~statrl.settings.bandits.adversarial.lipschitz.agents.ALF.ALFLearner`.

.. [Jin2021] Jin, T., Tang, J., Xu, P., Huang, K., Xiao, X. and Gu, Q.
   "Almost optimal anytime algorithm for batched multi-armed bandits."
   *International Conference on Machine Learning (ICML)*, 2021.

   The five-phase epoch structure of
   :class:`~statrl.settings.bandits.stochastic.batch.agents.BABA.BABA`.

.. [Gautron2024] Gautron, R., Maillard, O.-A., Preux, P. and Corbeels, M.
   "Bandits with bounded CVaR constraints."
   2024.

   Non-parametric Thompson sampling with a Dirichlet prior anchored at the
   reward bound. Implemented, in the ``CVaR = Expectation`` regime, by
   :class:`~statrl.settings.bandits.stochastic.batch.agents.BCB.BCB`.

Environments
------------

.. [Strehl2008] Strehl, A. L. and Littman, M. L.
   "An analysis of model-based interval estimation for Markov decision
   processes."
   *Journal of Computer and System Sciences*, 74(8):1309-1331, 2008.

   Source of the RiverSwim benchmark, implemented by
   :class:`~statrl.settings.markovdecisionprocess.discrete_nostructure.envs.riverswim.RiverSwim`.

.. [Sutton1999] Sutton, R. S., Precup, D. and Singh, S.
   "Between MDPs and semi-MDPs: A framework for temporal abstraction in
   reinforcement learning."
   *Artificial Intelligence*, 112(1-2):181-211, 1999.

   Source of the four-room gridworld, built by
   :func:`~statrl.settings.markovdecisionprocess.gridworld.envs.gridworlds.fourRoomMap`.

Background
----------

.. [Lai1985] Lai, T. L. and Robbins, H.
   "Asymptotically efficient adaptive allocation rules."
   *Advances in Applied Mathematics*, 6(1):4-22, 1985.

   The lower bound that "asymptotically optimal" refers to throughout this
   documentation.
