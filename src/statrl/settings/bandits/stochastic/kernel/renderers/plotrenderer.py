from __future__ import annotations

from typing import Optional

import matplotlib.pyplot as plt

from statrl.settings.bandits.stochastic.kernel.environment import KernelBanditEnv


class PlotRenderer:
    """Matplotlib renderer for kernel bandit environments.

    Plots the latent function and the arm played each round: a curve for a
    1-d feature space, a scatter for 2-d, or a text summary for higher
    dimensions (see :meth:`_plot_1d`, :meth:`_plot_2d`,
    :meth:`_plot_high_dimensional`).

    Parameters
    ----------
    pause : float, default=0.1
        Seconds to pause after each draw, via :func:`matplotlib.pyplot.pause`,
        so the figure has time to update between rounds.

    Attributes
    ----------
    fig : matplotlib.figure.Figure or None
        The renderer's figure, created in :meth:`start` and cleared in
        :meth:`stop`.
    ax : matplotlib.axes.Axes or None
        The single axes the environment is drawn on.
    colorbar : matplotlib.colorbar.Colorbar or None
        Colorbar shown for the 2-d case; ``None`` otherwise.
    started : bool
        Whether :meth:`start` has run yet. Set lazily on the first
        :meth:`render` call.
    """

    def __init__(self, pause: float = 0.1) -> None:
        self.pause = pause
        self.fig: Optional[plt.Figure] = None
        self.ax: Optional[plt.Axes] = None
        self.colorbar: Optional[plt.Colorbar] = None
        self.started = False

    def start(self, env: KernelBanditEnv) -> None:
        """Create the figure and draw the environment's initial state.

        Parameters
        ----------
        env : ~statrl.settings.bandits.stochastic.kernel.environment.KernelBanditEnv
            The environment being rendered.
        """
        self.fig = plt.figure()
        self.ax = self.fig.add_subplot(111)

        self.outfile = None

        self._plot_environment(env)

        plt.ion()
        plt.show(block=False)
        plt.pause(self.pause)

    def stop(self, env: KernelBanditEnv) -> None:
        """Close the figure at the end of a rendered run.

        Parameters
        ----------
        env : ~statrl.settings.bandits.stochastic.kernel.environment.KernelBanditEnv
            The environment being rendered. Unused; accepted to match the
            renderer protocol.
        """
        if self.fig is not None:
            plt.ioff()
            plt.show()
            plt.close(self.fig)

        self.fig = None
        self.ax = None
        self.colorbar = None
        self.started = False

    def render(
        self,
        env: KernelBanditEnv,
        last: tuple[Optional[int], float],
    ) -> None:
        """Draw the current round, creating the figure on the first call.

        Parameters
        ----------
        env : ~statrl.settings.bandits.stochastic.kernel.environment.KernelBanditEnv
            The environment being rendered.
        last : tuple of (int or None, float)
            The ``(arm, reward)`` pair recorded by the environment. A ``None``
            arm means no pull has happened yet.
        """
        if not self.started:
            self.start(env)
            self.started = True
            return

        self._plot_environment(env, last)

        plt.pause(self.pause)

    def _plot_environment(
        self,
        env: KernelBanditEnv,
        last: tuple[Optional[int], float] = (None, 0.0),
    ) -> None:
        """Clear the axes and redraw, dispatching on the feature dimension."""
        assert self.ax is not None and self.fig is not None  # set by start(), which always runs first

        # Remove the previous colorbar before clearing/redrawing.
        if self.colorbar is not None:
            self.colorbar.remove()
            self.colorbar = None

        self.ax.clear()

        dimension = env.arm_features.shape[1]

        if dimension == 1:
            self._plot_1d(env, last)

        elif dimension == 2:
            self._plot_2d(env, last)

        else:
            self._plot_high_dimensional(env, last)

        self.ax.set_title(env.displayname)
        self.fig.canvas.draw_idle()

    def _plot_1d(
        self,
        env: KernelBanditEnv,
        last: tuple[Optional[int], float],
    ) -> None:
        """Draw the latent function as a curve, for a 1-d feature space."""
        assert self.ax is not None  # set by start(), which always runs first
        x = env.arm_features[:, 0]
        f = env.means

        self.ax.plot(x, f, label="latent function")

        optimal = env.optimal_arm

        self.ax.scatter(
            x[optimal],
            f[optimal],
            marker="*",
            s=150,
            label="optimal",
        )

        lastaction, lastreward = last

        if lastaction is not None:
            self.ax.scatter(
                x[lastaction],
                f[lastaction],
                s=80,
                label=f"selected (r={lastreward:.2f})",
            )

        self.ax.set_xlabel("x")
        self.ax.set_ylabel("f(x)")
        self.ax.legend()

    def _plot_2d(
        self,
        env: KernelBanditEnv,
        last: tuple[Optional[int], float],
    ) -> None:
        """Draw the latent function as a colored scatter, for a 2-d feature space."""
        assert self.ax is not None and self.fig is not None  # set by start(), which always runs first
        x = env.arm_features[:, 0]
        y = env.arm_features[:, 1]
        f = env.means

        scatter = self.ax.scatter(
            x,
            y,
            c=f,
            s=40,
        )

        self.colorbar = self.fig.colorbar(
            scatter,
            ax=self.ax,
            label="f(x)",
        )

        optimal = env.optimal_arm

        self.ax.scatter(
            x[optimal],
            y[optimal],
            marker="*",
            s=200,
            edgecolors="black",
            label="optimal",
        )

        lastaction, lastreward = last

        if lastaction is not None:
            self.ax.scatter(
                x[lastaction],
                y[lastaction],
                s=100,
                facecolors="none",
                edgecolors="black",
                linewidths=2,
                label=f"selected (r={lastreward:.2f})",
            )

        self.ax.set_xlabel("$x_1$")
        self.ax.set_ylabel("$x_2$")
        self.ax.legend()

    def _plot_high_dimensional(
        self,
        env: KernelBanditEnv,
        last: tuple[Optional[int], float],
    ) -> None:
        """Draw a text summary, for a feature space of dimension 3 or more."""
        assert self.ax is not None  # set by start(), which always runs first
        lastaction, lastreward = last

        self.ax.axis("off")

        text = (
            f"Kernel Bandit\n\n"
            f"Dimension: {env.arm_features.shape[1]}\n"
            f"Number of arms: {env.number_arms}\n"
            f"Optimal arm: {env.optimal_arm}\n"
            f"Optimal value: {env.optimal_mean:.3f}\n"
        )

        if lastaction is not None:
            text += (
                f"\nLast arm: {lastaction}\n"
                f"Reward: {lastreward:.3f}"
            )

        self.ax.text(
            0.5,
            0.5,
            text,
            ha="center",
            va="center",
            fontsize=14,
        )