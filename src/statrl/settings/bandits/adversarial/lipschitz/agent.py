from typing import Any, Optional


class Agent:
    """Base class for agents in the adversarial Lipschitz setting.

    The protocol differs from the stochastic bandit one in two ways: actions
    are points of a continuous metric space rather than arm indices, and
    :meth:`select_arm` receives an observation, since the adversary's choice
    at round ``t`` may be partly revealed before the action is committed.
 
    """

    def select_arm(self, observation: Any) -> Any:
        """Choose the action :math:`x_t` to play this round.

        Parameters
        ----------
        observation : object
            Observation for the current round, or ``None`` in the pure bandit
            case where the environment reveals nothing in advance.

        Returns
        -------
        object
            A point of the action space, typically an
            :class:`~numpy.ndarray`. The interaction loop clips it to the
            bounds of a :class:`~gymnasium.spaces.Box` action space.

        Raises
        ------
        NotImplementedError
            Always, in the base class; subclasses must override.
        """
        raise NotImplementedError

    def update(self, action: Any, reward: float, observation: Optional[Any] = None) -> None:
        """Learn from the reward observed for the action just played.

        Optional: the default does nothing, so a fixed-action baseline needs
        to implement :meth:`select_arm` only.

        Parameters
        ----------
        action : object
            The action that was played.
        reward : float
            Value :math:`f_t(x_t)` returned by the adversary. Only this scalar
            is observed — never the whole function :math:`f_t`.
        observation : object, optional
            Observation the action was chosen from.
        """
        pass