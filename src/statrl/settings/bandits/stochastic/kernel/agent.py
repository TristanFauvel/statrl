from abc import ABC, abstractmethod
from typing import Callable

from gymnasium.utils import seeding
import numpy as np

class KernelBanditAgent(ABC):
    """Base class for kernel bandit agents.

    Subclasses implement the protocol shared by every agent in this setting:
    :meth:`reset` to start an independent run, :meth:`select_arm` to choose an
    arm, and :meth:`update` to learn from the observed reward.

    Parameters
    ----------
    arms : ndarray of shape (nbArms, d)
        Feature vector of each arm, as in
        :attr:`~statrl.settings.bandits.stochastic.kernel.environment.KernelBanditEnv.arm_features`.
    kernel : callable
        Covariance function ``kernel(x, y) -> float`` over feature vectors,
        e.g. one built by
        :func:`~statrl.settings.bandits.stochastic.kernel.envs.kernels.RBFKernel`.
    noise_std : float, default=0.1
        Standard deviation of the observation noise the agent should assume.
    name : str, default='Kernel Bandit Agent'
        Label used in logfiles and plot legends.
    seed : int, default=1
        Seed for the agent's own randomness. :meth:`reset` re-derives
        ``np_random`` from it, so replicates are reproducible.

    Attributes
    ----------
    np_random : numpy.random.Generator
        Agent-local generator, available after the first :meth:`reset`.

    See Also
    --------
    statrl.settings.bandits.stochastic.anytime.agent.BanditAgent :
        The unstructured counterpart, with no feature vectors or kernel.
    """

    def __init__(
        self,
        arms: np.ndarray,
        kernel: Callable,
        noise_std: float = 0.1,
        name: str = "Kernel Bandit Agent",
        seed: int = 1,
    ) -> None:
        self.arms = np.asarray(arms)
        self.kernel = kernel
        self.noise_std = noise_std
        self.name = name
        self.seed = seed

    def reset(self) -> None:
        """Start a new independent run by reseeding ``np_random``."""
        self.np_random, self.seed = seeding.np_random(self.seed)

    @abstractmethod
    def select_arm(self) -> int:
        """Choose the arm to pull next.

        Returns
        -------
        int
            Index of the selected arm, in ``range(len(arms))``.
        """
        ...

    @abstractmethod
    def update(self, arm: int, reward: float) -> None:
        """Learn from the reward observed for the arm just pulled.

        Parameters
        ----------
        arm : int
            Index of the arm that was pulled.
        reward : float
            Reward sampled by
            :meth:`~statrl.settings.bandits.stochastic.kernel.environment.KernelBanditEnv.step`.
        """
        ...