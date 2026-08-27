from abc import ABC, abstractmethod
from gymnasium.utils import seeding


class BatchBanditAgent(ABC):
    """Base class for batched bandit agents.

    In the batched setting an agent must commit to a whole block of pulls
    before seeing any of their rewards.  

    Two levels of interface are provided. :meth:`play` and :meth:`update` are
    the per-pull rules; :meth:`batchplay` and :meth:`batchupdate` are what the
    interaction loop actually calls. Subclasses must implement the batch pair,
    which lets them exploit within-batch structure: an agent may update its
    index between the pulls of a batch (using only what it knew when the batch
    began) even though no reward has yet arrived.

    Parameters
    ----------
    name : str, default='BanditAgent'
        Label used in logfiles and plot legends.
    seed : int, default=1
        Seed for the agent's own randomness.

    Attributes
    ----------
    np_random : numpy.random.Generator
        Agent-local generator, available after the first :meth:`reset`.

    See Also
    --------
    statrl.settings.bandits.batch.interaction.BatchBanditInteraction :
        The loop that drives these agents.
    """

    def __init__(self, name: str = "BanditAgent", seed: int = 1) -> None:
        self.name = name
        self.seed =seed

    def reset(self) -> None:
        """Start a new independent run, reseeding the agent's generator."""
        self.np_random, self.seed = seeding.np_random(self.seed)

    def play(self) -> int:
        """Choose a single arm.

        Returns
        -------
        int
            Index of the selected arm.

        Raises
        ------
        NotImplementedError
            If not overridden. Agents whose :meth:`batchplay` builds a batch
            from repeated single pulls must implement this; agents that decide
            a batch as a whole, such as
            :class:`~statrl.settings.bandits.batch.agents.BABA.BABA`, need not.
        """
        raise NotImplementedError


    def update(self, arm: int, reward: float)-> None:
        """Learn from one ``(arm, reward)`` pair.

        Parameters
        ----------
        arm : int
            Index of the arm that was pulled.
        reward : float
            Reward observed for it.
        """
        pass

    @abstractmethod
    def batchplay(self, batchsize: int) -> list[int]:
        """Commit to the arms of a whole batch, before any reward is seen.

        Parameters
        ----------
        batchsize : int
            Number of pulls in this batch, announced by the environment as
            ``info["nextbatchsize"]``. It varies between batches under a
            non-constant schedule.

        Returns
        -------
        list of int
            Exactly ``batchsize`` arm indices. The environment asserts the
            length. The default implementation repeats :meth:`play`.
        """
        return [self.play() for b in range(batchsize)]

    @abstractmethod
    def batchupdate(self, batcharm: list[int], batchreward: list[float]) -> None:
        """Learn from all the rewards of a batch at once.

        Parameters
        ----------
        batcharm : list of int
            The arms that were pulled, as returned by :meth:`batchplay`.
        batchreward : list of float
            The rewards observed for them, in the same order.

        Notes
        -----
        The default implementation replays the pairs through :meth:`update`.
        """
        for arm, reward in zip(batcharm, batchreward):
            self.update(arm, reward)