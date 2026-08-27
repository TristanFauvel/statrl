
from abc import ABC, abstractmethod

from gymnasium.utils import seeding

class BanditAgent(ABC):
    """Base class for horizon-aware stochastic bandit agents.

    Identical to the anytime protocol except that :meth:`reset` receives the
    horizon, letting an agent tune its behaviour to the number of rounds it
    will play.

    Parameters
    ----------
    name : str
        Label used in logfiles and plot legends.
    seed : int, default=1
        Seed for the agent's own randomness.

    Attributes
    ----------
    horizon : int
        Number of rounds of the current run, set by :meth:`reset`.
    np_random : numpy.random.Generator
        Agent-local generator, available after the first :meth:`reset`.

    See Also
    --------
    statrl.settings.bandits.stochastic.anytime.agent.BanditAgent :
        The anytime counterpart.
    statrl.settings.bandits.stochastic.knownhorizon.wrappers.wrapper_anytime_knownhorizon.AnytimeToKnownHorizonAgentWrapper :
        Runs an anytime agent in this setting.
    """

    def __init__(self,name: str,seed: int = 1) -> None:
        self.name = name
        self.seed =seed


    def reset(self, horizon: int) -> None:
        """Start a new independent run of known length.

        Parameters
        ----------
        horizon : int
            Number of rounds that will be played. Stored on
            :attr:`horizon` and free to be used by the selection rule.
        """
        self.np_random, self.seed = seeding.np_random(self.seed)
        self.horizon = horizon

    @abstractmethod
    def select_arm(self) -> int:
        """Choose the arm to pull next.

        Returns
        -------
        int
            Index of the selected arm, in ``range(env.number_arms)``.
        """

    def update(self, arm: int, reward: float) -> None:
        """Learn from the reward observed for the arm just pulled. 

        Parameters
        ----------
        arm : int
            Index of the arm that was pulled.
        reward : float
            Reward observed for that arm.
        """
