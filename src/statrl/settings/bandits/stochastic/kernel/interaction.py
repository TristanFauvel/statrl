

import numpy as np
from statrl.settings.bandits.stochastic.kernel.environment import KernelBanditEnv
from statrl.settings.bandits.stochastic.kernel.agent import KernelBanditAgent


from statrl.settings.bandits.stochastic.kernel.renderers.textrenderer import Textrenderer
from statrl.settings.bandits.stochastic.kernel.renderers.plotrenderer import  PlotRenderer

from statrl.experiments.onerun import Interaction

class KernelBanditInteraction(Interaction):
    """Interaction loop for the kernel bandit setting.

    Drives ``select_arm`` -> ``step`` -> ``update`` for a fixed number of
    rounds and returns the cumulative *expected* score.

    See Also
    --------
    statrl.settings.bandits.stochastic.anytime.interaction.BanditInteraction :
        The unstructured counterpart.
    """

    def run(self, env: KernelBanditEnv, learner: KernelBanditAgent, horizon: int) -> np.ndarray:
        """Run one interaction and return its cumulative expected score.

        Parameters
        ----------
        env : ~statrl.settings.bandits.stochastic.kernel.environment.KernelBanditEnv
            The bandit instance. Reset at the start of the run.
        learner : ~statrl.settings.bandits.stochastic.kernel.agent.KernelBanditAgent
            The agent. Reset at the start of the run, so a single instance
            can be reused across replicates.
        horizon : int
            Number of rounds to play.

        Returns
        -------
        ndarray of shape (horizon,)
            Cumulative sum of the *expected* rewards of the arms played.
            Accumulating means rather than realized rewards removes the
            reward noise from the regret curve.
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

    def renderrun(self, env: KernelBanditEnv, learner: KernelBanditAgent, horizon: int) -> None:
        """Run one interaction, printing and plotting each round.

        Same loop as :meth:`run`, but a
        :class:`~statrl.settings.bandits.stochastic.kernel.renderers.textrenderer.Textrenderer`
        and a
        :class:`~statrl.settings.bandits.stochastic.kernel.renderers.plotrenderer.PlotRenderer`
        are attached to the environment and no score is returned. Intended
        for inspecting short runs by eye, not for benchmarking.

        Parameters
        ----------
        env : ~statrl.settings.bandits.stochastic.kernel.environment.KernelBanditEnv
            The bandit instance. Its ``renderers`` list is overwritten.
        learner : ~statrl.settings.bandits.stochastic.kernel.agent.KernelBanditAgent
            The agent.
        horizon : int
            Number of rounds to play.
        """
        env.renderers= [Textrenderer(),PlotRenderer()]
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






