from typing import Optional

import numpy as np

from gymnasium import Env as Environment
from gymnasium.utils import seeding





class KernelBanditEnv(Environment):
    """Finite kernelized bandit environment.

    A latent reward function is sampled once from a Gaussian Process

    .. math:: f \\sim \\mathrm{GP}(0, k)

    over the arm feature vectors, using ``kernel`` as the covariance function
    :math:`k`. Pulling arm :math:`i` returns :math:`f(x_i) + \\text{noise}`,
    where :math:`x_i` is that arm's feature vector. 

    Parameters
    ----------
    arm_features : array-like of shape (nbArms, d)
        Feature vector of each arm.
    kernel : callable
        Covariance function ``kernel(x, y) -> float`` over feature vectors,
        e.g. one built by
        :func:`~statrl.settings.bandits.stochastic.kernel.envs.kernels.RBFKernel`.
    noise_std : float, default=0.0
        Standard deviation of the observation noise added in :meth:`step`.
    name : str, default='KernelBandit'
        Label used in logfiles, plot titles, and dump filenames.
    function_seed : int, default=1
        Seed for sampling the latent function :math:`f`. Fixed at
        construction, so the arm means are the same across every
        :meth:`reset`.

    Attributes
    ----------
    renderers : list
        Renderers notified on every :meth:`render` call. Empty by default.
    np_random : numpy.random.Generator
        Environment-local generator for observation noise, available after
        :meth:`reset`.

    See Also
    --------
    statrl.settings.bandits.stochastic.kernel.envs.kernels.KernelBandit :
        Factory building the feature grid and kernel from named arguments.

    Examples
    --------
    >>> import numpy as np
    >>> from statrl.settings.bandits.stochastic.kernel.envs.kernels import RBFKernel
    >>> arms = np.linspace(0, 1, 100).reshape(-1, 1)
    >>> env = KernelBanditEnv(arm_features=arms, kernel=RBFKernel(), noise_std=0.1)
    >>> env.number_arms
    100
    """

    def __init__(
        self,
        arm_features,
        kernel,
        noise_std: float = 0.0,
        name: str = "KernelBandit",
        function_seed: int = 1,
    ):

        self.arm_features = np.asarray(arm_features, dtype=float)
        self.kernel = kernel
        self.noise_std = noise_std
        self.name = name
        self.displayname: str = name

        self.function_seed = function_seed
        self.renderers: list = []
        self.last: tuple[Optional[int], float] = (None, 0.0)

        self._sample_function()

    @property
    def number_arms(self) -> int:
        """Number of available arms."""
        return len(self.arm_features)

    @property
    def means(self):
        """ndarray of shape (nbArms,): Mean reward of every arm, :math:`f(x_i)`.

        This is the single realization of the latent GP sampled at
        construction.

        Raises
        ------
        RuntimeError
            If accessed before :meth:`_sample_function` has run (should not
            happen in normal use, since it runs at construction).
        """
        if self._means is None:
            raise RuntimeError(
                "Environment must be reset before accessing means."
            )
        return self._means

    @property
    def optimal_mean(self) -> float:
        """float: Mean reward of the best arm, :math:`\\max_i f(x_i)`."""
        return float(np.max(self.means))

    @property
    def optimal_arm(self) -> int:
        """int: Index of the best arm.

        Ties are broken by :func:`numpy.argmax`, i.e. the lowest index wins.
        """
        return int(np.argmax(self.means))

    def _covariance_matrix(self) -> np.ndarray:
        """Gram matrix of the kernel over every pair of arm features."""
        n = self.number_arms

        K = np.empty((n, n))

        for i in range(n):
            for j in range(n):
                K[i, j] = self.kernel(
                    self.arm_features[i],
                    self.arm_features[j]
                )

        return K

    def _sample_function(self) -> None:
        """Draw the latent GP realization :math:`f` once, at construction.

        Populates :attr:`means`. Uses a dedicated generator seeded from
        :attr:`function_seed`, independent of :attr:`np_random`, so the arm
        means do not change across :meth:`reset` calls.
        """
        # Dedicated RNG for the latent GP realization
        rng = np.random.default_rng(self.function_seed)

        K = self._covariance_matrix()

        K += 1e-10 * np.eye(self.number_arms)

        self._means = rng.multivariate_normal(
            mean=np.zeros(self.number_arms),
            cov=K,
        )

    def reset(self, seed=None, options=None):
        """Start a new run by reseeding the observation-noise generator.

        The latent function :attr:`means` is not resampled; only the noise
        added by :meth:`step` is affected.

        Parameters
        ----------
        seed : int, optional
            Seed for the environment's noise generator. ``None`` draws a
            fresh one.
        options : dict, optional
            Unused; accepted for :class:`gymnasium.Env` compatibility.

        Returns
        -------
        int
            The constant dummy observation ``0`` — a bandit is stateless.
        """
        # RNG used for the interaction noise
        self.np_random, self.seed = seeding.np_random(seed)

        self.last = (None, 0.0)

        return 0

    def expected_reward(self, arm: int) -> float:
        """Expected reward of an arm, for regret accounting only.

        Parameters
        ----------
        arm : int
            Index of the arm.

        Returns
        -------
        float
            That arm's true mean, :math:`f(x_{\\text{arm}})`. Never pass this
            to a learner.
        """
        return float(self._means[arm])

    def step(self, arm: int) -> float:  # type: ignore[override]  # bandit API: reward only, not gym's 5-tuple
        """Sample one reward from the given arm.

        Parameters
        ----------
        arm : int
            Index of the arm to pull, in ``range(number_arms)``.

        Returns
        -------
        float
            :math:`f(x_{\\text{arm}})` plus Gaussian noise of standard
            deviation :attr:`noise_std` (no noise added if it is zero).
        """
        reward = self._means[arm]

        if self.noise_std > 0:
            reward += self.np_random.normal(
                0.0,
                self.noise_std
            )

        self.last = (arm, reward)

        return float(reward)

    def render(self, mode="human"):
        """Forward the last ``(arm, reward)`` pair to every attached renderer.

        Parameters
        ----------
        mode : str, default='human'
            Unused; accepted for :class:`gymnasium.Env` compatibility.
        """
        for renderer in self.renderers:
            renderer.render(self, self.last)

    def close(self):
        """Release every attached renderer at the end of a rendered run."""
        for renderer in self.renderers:
            renderer.stop(self)