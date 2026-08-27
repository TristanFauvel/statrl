from __future__ import annotations

import numpy as np

from statrl.settings.bandits.stochastic.kernel.environment import KernelBanditEnv


# ---------------------------------------------------------------------------
# Kernels
# ---------------------------------------------------------------------------

def RBFKernel(lengthscale: float = 0.2, variance: float = 1.0):
    """Squared-exponential / RBF kernel.

    Parameters
    ----------
    lengthscale : float, default=0.2
        Distance over which correlation decays.
    variance : float, default=1.0
        Kernel variance, :math:`k(x, x)`.

    Returns
    -------
    callable
        ``kernel(x, y) -> float``, infinitely differentiable in the distance
        between ``x`` and ``y``.
    """

    def kernel(x: np.ndarray, y: np.ndarray) -> float:
        """RBF covariance between feature vectors ``x`` and ``y``."""
        d2 = np.sum((x - y) ** 2)
        return variance * np.exp(-d2 / (2.0 * lengthscale ** 2))

    return kernel


def Matern32Kernel(lengthscale: float = 0.2, variance: float = 1.0):
    """Matérn-3/2 kernel.

    Parameters
    ----------
    lengthscale : float, default=0.2
        Distance over which correlation decays.
    variance : float, default=1.0
        Kernel variance, :math:`k(x, x)`.

    Returns
    -------
    callable
        ``kernel(x, y) -> float``, once differentiable — rougher than the RBF
        kernel.
    """

    def kernel(x: np.ndarray, y: np.ndarray) -> float:
        """Matérn-3/2 covariance between feature vectors ``x`` and ``y``."""
        r = np.linalg.norm(x - y)
        z = np.sqrt(3.0) * r / lengthscale
        return variance * (1.0 + z) * np.exp(-z)

    return kernel


def Matern52Kernel(lengthscale: float = 0.2, variance: float = 1.0):
    """Matérn-5/2 kernel.

    Parameters
    ----------
    lengthscale : float, default=0.2
        Distance over which correlation decays.
    variance : float, default=1.0
        Kernel variance, :math:`k(x, x)`.

    Returns
    -------
    callable
        ``kernel(x, y) -> float``, twice differentiable — between the RBF and
        Matérn-3/2 kernels in smoothness.
    """

    def kernel(x: np.ndarray, y: np.ndarray) -> float:
        """Matérn-5/2 covariance between feature vectors ``x`` and ``y``."""
        r = np.linalg.norm(x - y)
        z = np.sqrt(5.0) * r / lengthscale
        return variance * (1.0 + z + z ** 2 / 3.0) * np.exp(-z)

    return kernel


kernel_dictionary = {"RBFKernel": RBFKernel, "Matern32Kernel": Matern32Kernel, "Matern52Kernel":Matern52Kernel}

# ---------------------------------------------------------------------------
# Kernel bandit constructor
# ---------------------------------------------------------------------------
from collections.abc import Sequence

def KernelBandit(
    nb_arms: int | Sequence[int],
    kernel: str,
    lengthscale: float = 0.2,
    variance: float = 1.0,
    noise_std: float = 0.1,
    name: str = "KernelBandit",
    function_seed: int =1,
) -> KernelBanditEnv:
    """Build a :class:`KernelBanditEnv` on a regular grid in :math:`[0, 1]^d`.

    Parameters
    ----------
    nb_arms : int or sequence of int
        Number of arms per dimension. An int gives a 1-d grid; a sequence of
        length ``d`` gives a ``d``-dimensional grid with that many points
        along each axis (total arms is the product).
    kernel : str
        Key of :data:`kernel_dictionary` — ``"RBFKernel"``,
        ``"Matern32Kernel"``, or ``"Matern52Kernel"``.
    lengthscale : float, default=0.2
        Passed to the kernel factory.
    variance : float, default=1.0
        Passed to the kernel factory.
    noise_std : float, default=0.1
        Standard deviation of the observation noise.
    name : str, default='KernelBandit'
        Base label; the returned environment's name also encodes the kernel
        and hyperparameters.
    function_seed : int, default=1
        Seed for sampling the latent function.

    Returns
    -------
    ~statrl.settings.bandits.stochastic.kernel.environment.KernelBanditEnv
        Environment over the grid, with feature vectors flattened in
        row-major order (:func:`numpy.meshgrid` with ``indexing="ij"``).

    Raises
    ------
    KeyError
        If ``kernel`` names no entry of :data:`kernel_dictionary`.
    """

    if isinstance(nb_arms, int):
        nb_arms = [nb_arms]

    axes = [
        np.linspace(0.0, 1.0, n)
        for n in nb_arms
    ]

    mesh = np.meshgrid(*axes, indexing="ij")

    arms = np.stack(
        [m.ravel() for m in mesh],
        axis=-1,
    )

    name = (
        f"{name}"
        f"-{kernel}"
        f"-K{list(nb_arms)}"
        f"-lengthscale{lengthscale}"
        f"-variance{variance}"
        f"-noise{noise_std}"
    )


    return KernelBanditEnv(
        arm_features=arms,
        kernel=kernel_dictionary[kernel](
            lengthscale=lengthscale,
            variance=variance,
        ),
        noise_std=noise_std,
        name=name,
        function_seed=function_seed
    )
