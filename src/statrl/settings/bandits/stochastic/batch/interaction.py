

import numpy as np
from statrl.settings.bandits.stochastic.batch.environment import BatchMAB
from statrl.settings.bandits.stochastic.batch.agent import BatchBanditAgent

from statrl.experiments.onerun import Interaction

class BatchBanditInteraction(Interaction):
    """Interaction loop for the batched bandit setting.

    Drives ``batchplay`` -> ``step`` -> ``batchupdate``. One round is one batch, so
    ``horizon`` counts batches rather than pulls.

    See Also
    --------
    statrl.settings.bandits.stochastic.anytime.interaction.BanditInteraction :
        The unbatched counterpart.

    Examples
    --------
    >>> from statrl.settings.bandits.stochastic.anytime.envs.parametric import BernoulliBandit
    >>> from statrl.settings.bandits.stochastic.batch.environment import BatchMAB
    >>> from statrl.settings.bandits.stochastic.batch.agents.BIMED import BIMED
    >>> env = BatchMAB(BernoulliBandit([0.2, 0.9, 0.5]), batchsize=[4] * 20)
    >>> BatchBanditInteraction().run(env, BIMED(3), horizon=20).shape
    (20,)
    """

    def run(self, env: BatchMAB, learner: BatchBanditAgent, horizon: int) -> np.ndarray:
        """Run one interaction and return its cumulative expected score.

        Parameters
        ----------
        env : BatchMAB
            The batched bandit instance.
        learner : BatchBanditAgent
            The agent.
        horizon : int
            Number of **batches** to play. The number of pulls is the sum of
            the batch sizes over those rounds.

        Returns
        -------
        ndarray of shape (horizon,)
            Cumulative sum of each batch's expected score, i.e. the summed
            true means of the arms pulled in it. Entry ``t`` therefore covers
            every pull up to the end of batch ``t``.
        """
        info = env.reset()
        learner.reset()
        B = info["nextbatchsize"]

        steps_scores = np.empty(horizon)

        for t in range(horizon):
            batchaction = learner.batchplay(B)  # Get action
            batchreward, info = env.step(batchaction)  # Get response
            learner.batchupdate(batchaction, batchreward)  # Update learners

            B = info["nextbatchsize"]
            steps_scores[t] = info["mean"]

        return np.cumsum(steps_scores)

    def renderrun(self, env: BatchMAB, learner: BatchBanditAgent, horizon: int) -> None:
        """Run one interaction with rendering enabled.

        Parameters
        ----------
        env : BatchMAB
            The batched bandit instance. 
        learner : BatchBanditAgent
            The agent.
        horizon : int
            Number of batches to play.
        """
        env.renderers= []
        info = env.reset()
        learner.reset()
        B = info["nextbatchsize"]

        env.render()
        for t in range(horizon):
            batchaction = learner.batchplay(B)  # Get action

            batchreward, info = env.step(batchaction)  # Get response
            learner.batchupdate(batchaction, batchreward)  # Update learners

            B = info["nextbatchsize"]
            env.render()

        env.close()


    @property
    def plotlabels(self):
        """tuple of (str, str): Axis labels ``(x, y)`` for the regret plots.

        The x-axis counts batches, not pulls, so it is labelled by episode.
        """
        return (r"Episode $\ell$", "Regret")