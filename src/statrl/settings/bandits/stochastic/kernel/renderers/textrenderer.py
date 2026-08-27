

import sys
import string
import numpy as np

from typing import Optional

from statrl.settings.bandits.stochastic.kernel.environment import KernelBanditEnv


class Textrenderer():
    """Print each kernel bandit round to stdout, one line per pull.

    Attributes
    ----------
    started : bool
        Whether the header has been printed yet. The header is emitted lazily
        on the first :meth:`render` call.
    """

    def __init__(self) -> None:
        self.started = False


    def start(self, env: KernelBanditEnv) -> None:
        """Print the header naming the environment and its grid shape.

        Parameters
        ----------
        env : ~statrl.settings.bandits.stochastic.kernel.environment.KernelBanditEnv
            The environment being rendered.
        """
        self.outfile = sys.stdout
        self.outfile.write("Environment: " + str(env.displayname) + "\n")
        self.outfile.write("Actions: "+ str(self._grid_shape(env)) + "\n")
        self.outfile.write("-"*30+"\n")

    def stop(self, env: KernelBanditEnv) -> None:
        """Print the closing rule at the end of a rendered run.

        Parameters
        ----------
        env : ~statrl.settings.bandits.stochastic.kernel.environment.KernelBanditEnv
            The environment being rendered.
        """
        self.outfile.write("-"*30+"\n")

    def _grid_shape(self, env: KernelBanditEnv) -> tuple[int, ...]:
        """Number of distinct values along each feature dimension."""
        return tuple(
            len(np.unique(env.arm_features[:, d]))
            for d in range(env.arm_features.shape[1])
        )

    def _nameActions(self, env: KernelBanditEnv) -> list[str]:
        """Name of every arm, in arm-index order."""
        return [
            self._nameAction(env, arm)
            for arm in range(env.number_arms)
        ]

    def _nameAction(self, env: KernelBanditEnv, arm: int) -> str:
        """Name one arm from its grid coordinates, one letter per dimension."""
        grid_shape = self._grid_shape(env)

        coordinates = np.unravel_index(
            arm,
            grid_shape,
        )

        letters = string.ascii_uppercase

        return "".join(
            letters[i%26]+(str(int(i/26)) if i>25 else "")
            for i in coordinates
        )

    def render(self, env: KernelBanditEnv, last: tuple[Optional[int], float]) -> None:
        """Print the arm just played and the reward it returned.

        Parameters
        ----------
        env : ~statrl.settings.bandits.stochastic.kernel.environment.KernelBanditEnv
            The environment being rendered.
        last : tuple of (int or None, float)
            The ``(arm, reward)`` pair recorded by the environment. A ``None``
            arm means no pull has happened yet, and nothing is printed.
        """
        lastaction, lastreward = last

        if not self.started:
            self.start(env)
            self.started = True

        if lastaction is not None:
            action_name = self._nameAction(env, lastaction)
            self.outfile.write(
                f"({action_name})\tr={lastreward:0.2f}\n"
            )
