def keyboard_waitfor(list):
    """Block on stdin until the user types one of the allowed strings.

    Parameters
    ----------
    list : sequence of str
        Accepted inputs. Anything else is silently re-prompted.

    Returns
    -------
    tuple of (str, int)
        The string entered and its position in ``list``.
    """
    while True:
        x = input()
        if x in list:
            i = list.index(x)
            return x,i

from statrl.settings.markovdecisionprocess.discrete_nostructure.agent import MDPAgent
class Human(MDPAgent):
    """Interactive agent that asks the user for each action.

    Prompts on stdin at every step and blocks until a valid action name is
    entered.  

    Parameters
    ----------
    env : object
        A wrapped environment; ``env.env`` must expose ``nameActions``, so a
        raw :class:`~statrl.settings.markovdecisionprocess.discrete_nostructure.environment.DiscreteMDP`
        will not do.
    """

    def __init__(self,env):
        self.env=env.env
        self.name="Human"
        super(Human, self).__init__(env.nS, env.nA, self.name)

    def reset(self,inistate):
        """Start a new run. Nothing is kept between runs.

        Parameters
        ----------
        inistate : int
            Initial state; ignored.
        """
        pass

    def play(self,state):
        """Print the state and block until the user names an action.

        Parameters
        ----------
        state : int
            Current state, shown to the user.

        Returns
        -------
        int
            Index of the action the user chose.
        """
        print("Current state is: ",state)
        print("Please choose an action in the following list (confirm with entry): ", self.env.nameActions)
        (name_action,action) = keyboard_waitfor(list(self.env.nameActions))
        print("Chosen action is: ",name_action)#, "(number ",action,")")

        return action

    def update(self, state, action, reward, observation):
        """Ignore the transition, the user is the policy.

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