
#" Environment interface are the same, only agents interface are different"
import statrl.settings.bandits.stochastic.anytime.agent as agentA
import statrl.settings.bandits.stochastic.knownhorizon.agent as agentK
from typing import Any

class AnytimeToKnownHorizonAgentWrapper(
        agentK.BanditAgent):
    """Run an anytime agent inside the known-horizon setting.

    An anytime agent simply ignores the horizon, so the wrapper drops it.   

    Parameters
    ----------
    learner : statrl.settings.bandits.stochastic.anytime.agent.BanditAgent
        The anytime agent to adapt.

    See Also
    --------
    KnownHorizonToAnytimeAgentWrapper : The converse adaptation.

    Examples
    --------
    >>> from statrl.settings.bandits.stochastic.anytime.agents.IMED import IMED
    >>> AnytimeToKnownHorizonAgentWrapper(IMED(3)).name
    'IMED-anytime'
    """

    def __init__(self, learner: agentA.BanditAgent) -> None:
        self.learner = learner
        super().__init__(self.learner.name+"-anytime")

    def reset(self, horizon: int) -> None:
        """Reset the wrapped agent, discarding the horizon.

        Parameters
        ----------
        horizon : int
            Ignored: an anytime agent must not depend on it.
        """
        self.learner.reset()

    def select_arm(self) -> int:
        """Delegate the choice to the wrapped agent.

        Returns
        -------
        int
            Index of the selected arm.
        """
        return self.learner.select_arm()

    def update(self, arm: int, reward: float) -> None:
        """Forward the observation to the wrapped agent.

        Parameters
        ----------
        arm : int
            Index of the arm that was pulled.
        reward : float
            Reward observed for that arm.
        """
        self.learner.update(arm, reward)


    @property
    def policy(self) -> Any:
        """Policy of the wrapped agent.

        Raises
        ------
        AttributeError
            If the wrapped agent has no ``policy``. Only oracles define one;
            the experiment harness reads it to log the optimal policy.
        """
        return self.learner.policy  # type: ignore[attr-defined]  # policy is oracle-specific, not on the base agent.py interface


class KnownHorizonToAnytimeAgentWrapper(
        agentA.BanditAgent):
    """Run a horizon-aware agent in the anytime setting, at a fixed horizon.

    An anytime agent is never told the horizon, so the wrapper must supply one up front and reuse it for
    every run. The wrapped agent then plans against ``horizon`` regardless of
    how long the interaction actually lasts.  

    Parameters
    ----------
    learner : statrl.settings.bandits.stochastic.knownhorizon.agent.BanditAgent
        The horizon-aware agent to adapt.
    horizon : int
        Horizon announced to the wrapped agent on every reset. Set it to the
        horizon of the experiment you intend to run.
    name : str
        Label for the wrapped agent. Unlike the converse wrapper, no suffix is
        added, so pick a name that records the fixed horizon.

    See Also
    --------
    AnytimeToKnownHorizonAgentWrapper : The converse, lossless adaptation.
    """

    def __init__(self, learner: agentK.BanditAgent, horizon: int, name: str) -> None:
        self.learner = learner
        self.horizon= horizon
        super().__init__(name)

    def reset(self) -> None:
        """Reset the wrapped agent, announcing the fixed horizon."""
        self.learner.reset(self.horizon)

    def select_arm(self) -> int:
        """Delegate the choice to the wrapped agent.

        Returns
        -------
        int
            Index of the selected arm.
        """
        return self.learner.select_arm()

    def update(self, arm: int, reward: float) -> None:
        """Forward the observation to the wrapped agent.

        Parameters
        ----------
        arm : int
            Index of the arm that was pulled.
        reward : float
            Reward observed for that arm.
        """
        self.learner.update(arm, reward)

    @property
    def policy(self) -> Any:
        """Policy of the wrapped agent.

        Raises
        ------
        AttributeError
            If the wrapped agent has no ``policy``; only oracles define one.
        """
        return self.learner.policy  # type: ignore[attr-defined]  # only oracle agents define this
