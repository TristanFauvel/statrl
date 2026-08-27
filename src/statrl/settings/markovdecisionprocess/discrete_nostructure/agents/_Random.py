
from statrl.settings.markovdecisionprocess.discrete_nostructure.agent import MDPAgent
class Random(MDPAgent):
    """Uniform exploration: take a random action in every state. 
    
    Parameters
    ----------
    env : DiscreteMDP
        The environment, whose ``action_space`` supplies the draws.
    """

    def __init__(self,env):
        self.env=env
        self.name= "Random"
        super(Random, self).__init__(env.nS, env.nA, self.name)

    def reset(self,inistate):
        """Start a new run. No statistics are kept, so this does nothing.

        Parameters
        ----------
        inistate : int
            Initial state; ignored.
        """
        pass

    def play(self,state):
        """Draw an action uniformly at random, ignoring the state.

        Parameters
        ----------
        state : int
            Current state; ignored.

        Returns
        -------
        int
            An action drawn from the environment's action space.
        """
        return self.env.action_space.sample()

    def update(self, state, action, reward, observation):
        """Ignore the transition — this agent does not learn.

        Parameters
        ----------
        state : int
            State the action was taken in.
        action : int
            Action taken.
        reward : float
            Reward observed.
        observation : int
            State reached.
        """
        pass