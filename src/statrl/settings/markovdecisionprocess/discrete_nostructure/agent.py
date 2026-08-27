import numpy as np
from gymnasium.utils import seeding

class MDPAgent:
    """Base class for agents in the discrete MDP setting.

    The protocol differs from the bandit one in that both :meth:`play` and
    :meth:`update` are state-aware: an action is chosen *for a state*, and
    learning must account for where the process went next.  

    Parameters
    ----------
    nS : int
        Number of states.
    nA : int
        Number of actions.
    name : str, default='Agent'
        Label used in logfiles and plot legends.
    seed : int, optional
        Seed for the agent's own randomness.

    Attributes
    ----------
    np_random : numpy.random.Generator
        Agent-local generator, available after the first :meth:`reset`.

    See Also
    --------
    statrl.settings.markovdecisionprocess.discrete_nostructure.agents.IMED_RL.IMEDRL
    statrl.settings.markovdecisionprocess.discrete_nostructure.agents.PSRL.PSRL
    """

    def __init__(self, nS, nA, name="Agent",seed=None):
        self.nS = nS
        self.nA = nA
        self.name= name
        self.seed = seed


    def reset(self,inistate) -> None:
        """Start a new independent run from a given initial state.

        Parameters
        ----------
        inistate : int
            State the environment was reset to. Agents that track a current
            state need it; the default only reseeds.
        """
        self.np_random, self.seed = seeding.np_random(self.seed)


    def play(self,state):
        """Choose an action for the given state.

        Parameters
        ----------
        state : int
            Current state.

        Returns
        -------
        int
            The chosen action. The default picks uniformly at random.
        """
        return np.random.randint(self.nA)

    def update(self, state, action, reward, observation):
        """Learn from one transition.

        Parameters
        ----------
        state : int
            State the action was taken in.
        action : int
            Action taken.
        reward : float
            Reward observed.
        observation : int
            State reached. Unlike a bandit, this is part of the feedback —
            learning the transition kernel is half the problem.
        """
        ()