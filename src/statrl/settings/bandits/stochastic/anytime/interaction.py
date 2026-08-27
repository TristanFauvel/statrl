

import numpy as np
from statrl.settings.bandits.stochastic.anytime.environment import StochasticBanditEnv
from statrl.settings.bandits.stochastic.anytime.agent import BanditAgent


from statrl.settings.bandits.stochastic.anytime.renderers.textrenderer import Textrenderer
from statrl.experiments.onerun import Interaction
class BanditInteraction(Interaction):
    """Interaction loop for the anytime stochastic bandit setting.

    Drives ``select_arm`` -> ``step`` -> ``update`` for a fixed number of
    rounds and returns the cumulative *expected* score.

    See Also
    --------
    statrl.settings.bandits.stochastic.knownhorizon.interaction.BanditInteraction :
        The counterpart that passes the horizon to ``reset``.

    Examples
    --------
    >>> from statrl.settings.bandits.stochastic.anytime.envs.parametric import BernoulliBandit
    >>> from statrl.settings.bandits.stochastic.anytime.agents.IMED import IMED
    >>> env = BernoulliBandit([0.2, 0.9, 0.5])
    >>> scores = BanditInteraction().run(env, IMED(env.number_arms), horizon=100)
    >>> scores.shape
    (100,)
    """

    def run(self, env: StochasticBanditEnv, learner: BanditAgent, horizon: int) -> np.ndarray:
        """Run one interaction and return its cumulative expected score.

        Parameters
        ----------
        env : ~statrl.settings.bandits.stochastic.anytime.environment.StochasticBanditEnv
            The bandit instance. Reset at the start of the run.
        learner : ~statrl.settings.bandits.stochastic.anytime.agent.BanditAgent
            The agent. Reset at the start of the run, so a single instance can
            be reused across replicates.
        horizon : int
            Number of rounds to play.

        Returns
        -------
        ndarray of shape (horizon,)
            Cumulative sum of the *expected* rewards of the arms played, i.e.
            entry ``t`` is :math:`\\sum_{s \\le t} \\mu_{a_s}`. Accumulating
            means rather than realized rewards removes the reward noise from
            the regret curve.
        """
        env.reset()
        learner.reset()

        steps_scores = np.empty(horizon)

        for t in range(horizon):
            arm = learner.select_arm()

            reward = env.step(arm)

            learner.update(arm, reward)

            steps_scores[t] = env.expected_reward(arm)

        return np.cumsum(steps_scores)

    def renderrun(self, env: StochasticBanditEnv, learner: BanditAgent, horizon: int) -> None:
        """Run one interaction, printing each round to stdout.

        Same loop as :meth:`run`, but a
        :class:`~statrl.settings.bandits.stochastic.anytime.renderers.textrenderer.Textrenderer`
        is attached to the environment and no score is returned. Intended for
        inspecting short runs by eye, not for benchmarking.

        Parameters
        ----------
        env : ~statrl.settings.bandits.stochastic.anytime.environment.StochasticBanditEnv
            The bandit instance. Its ``renderers`` list is overwritten.
        learner : ~statrl.settings.bandits.stochastic.anytime.agent.BanditAgent
            The agent.
        horizon : int
            Number of rounds to play.
        """
        env.renderers= [Textrenderer()]
        env.reset()
        learner.reset()

        env.render()
        for t in range(horizon):
            arm = learner.select_arm()

            reward = env.step(arm)

            learner.update(arm, reward)

            env.render()

        env.close()

    @property
    def plotlabels(self):
        """tuple of (str, str): Axis labels ``(x, y)`` for the regret plots."""
        return ("Time step", "Regret")






