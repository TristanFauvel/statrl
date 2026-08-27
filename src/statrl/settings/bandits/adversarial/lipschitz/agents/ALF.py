import numpy as np
from typing import Any, Optional

from statrl.settings.bandits.adversarial.lipschitz.agent import Agent

class ALFLearner(Agent):
    """Adversarial Lipschitz Forecaster

    Learns in a continuous Lipschitz bandit/online optimization setting
    by reducing to a finite expert set.

    Parameters
    ----------
    action_space : object
        Continuous domain. Only :class:`~gymnasium.spaces.Box`-like spaces,
        exposing ``low`` and ``high``, are supported.
    epsilon : float
        Cover resolution. The grid uses ``max(2, int(1 / epsilon))`` points
        per dimension, so the expert count grows exponentially with the
        dimension of the space.
    eta : float
        Learning rate of the exponential weights.
    horizon : int
        Time horizon :math:`T`. Stored for tuning ``epsilon`` and ``eta``;
        this implementation does not tune them for you.
    metric : object, optional
        Metric for building the cover. Unused by the current Box grid.
    sampling : {'argmax', 'sample'}, default='argmax'
        Whether to play the highest-weight cover point deterministically or
        draw one from the weight distribution. 
    Attributes
    ----------
    actions : ndarray of shape (n_actions, n_dims)
        The cover points, i.e. the expert set.
    n_actions : int
        Number of cover points.
    weights : ndarray of shape (n_actions,)
        Current expert weights, normalized to sum to one.
    time : int
        Number of updates performed so far.

    Raises
    ------
    NotImplementedError
        If ``action_space`` exposes no ``low`` / ``high``; a general metric
        cover is not implemented.

    References
    ----------
    .. [1] Maillard, O.-A. and Munos, R. "Online learning in adversarial
           Lipschitz environments." *European Conference on Machine Learning
           and Knowledge Discovery in Databases*, 305-320, 2010.
    """

    def __init__(
        self,
        action_space: Any,
        epsilon: float,
        eta: float,
        horizon: int,
        metric: Optional[Any] = None,
        sampling: str = "argmax",
    ) -> None:
        """
        Parameters
        ----------
        action_space:
            Continuous domain (assumed metric space or Box).

        epsilon:
            Discretization resolution.

        eta:
            Learning rate for exponential weights.

        horizon:
            Time horizon T (may be used for tuning epsilon/eta).

        metric:
            Optional metric for cover construction.

        sampling:
            'argmax' or 'sample'
        """

        self.action_space = action_space
        self.epsilon = epsilon
        self.eta = eta
        self.horizon = horizon
        self.metric = metric
        self.sampling = sampling

        # ------------------------------------------------------------
        # Step 1: build finite discretization (ε-net / grid / particles)
        # ------------------------------------------------------------
        self.actions = self._build_cover(action_space, epsilon)

        self.n_actions = len(self.actions)

        # ------------------------------------------------------------
        # Step 2: initialize uniform weights
        # ------------------------------------------------------------
        self.weights = np.ones(self.n_actions) / self.n_actions

        self.time = 0

    # ================================================================
    # CORE INTERFACE
    # ================================================================

    def select_arm(self, observation: Optional[Any] = None) -> np.ndarray:
        """Pick a cover point according to the current expert weights.

        Parameters
        ----------
        observation : object, optional
            Ignored; ALF plays from its weights alone.

        Returns
        -------
        ndarray
            The chosen cover point. Its index is remembered so that the next
            :meth:`update` credits the right expert.
        """

        probs = self._get_probabilities()

        if self.sampling == "sample":
            idx = np.random.choice(self.n_actions, p=probs)
        else:
            idx = int(np.argmax(probs))

        self.last_idx = idx
        return self.actions[idx]

    def update(self, action: np.ndarray, reward: float, observation: Optional[Any] = None) -> None:
        """Apply the Hedge update to the expert that was played.

        Multiplies that expert's weight by :math:`e^{\\eta r}` and
        renormalizes.

        Parameters
        ----------
        action : ndarray
            The action played. Ignored — the expert index recorded by
            :meth:`select_arm` is used instead.
        reward : float
            Observed reward :math:`f_t(x_t)`.
        observation : object, optional
            Ignored.
 
        """

        idx = self.last_idx

        # importance-weighted or direct reward (bandit/full-info abstraction)
        r = float(reward)

        # ------------------------------------------------------------
        # Exponential weights update
        # ------------------------------------------------------------
        self.weights[idx] *= np.exp(self.eta * r)

        # normalize
        self.weights /= np.sum(self.weights)

        self.time += 1

    # ================================================================
    # INTERNALS
    # ================================================================

    def _get_probabilities(self) -> np.ndarray:
        """
        Convert weights to probability distribution.
        """
        w = np.array(self.weights)
        return w / np.sum(w)

    def _build_cover(self, action_space: Any, epsilon: float) -> np.ndarray:
        """
        Constructs ε-discretization of the action space.

        NOTE: In the paper this is abstract (metric cover).
        Here we instantiate a simple version for Box spaces.
        """

        if hasattr(action_space, "low") and hasattr(action_space, "high"):
            low, high = action_space.low, action_space.high

            # grid resolution based on epsilon
            dims = len(low)
            steps = max(2, int(1.0 / epsilon))

            grids = [np.linspace(lo, hi, steps) for lo, hi in zip(low, high)]

            mesh = np.meshgrid(*grids)
            points = np.stack(mesh, axis=-1).reshape(-1, dims)

            return points

        raise NotImplementedError("General metric cover not implemented.")