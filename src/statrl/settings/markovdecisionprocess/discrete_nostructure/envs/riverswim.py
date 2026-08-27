
from statrl.settings.markovdecisionprocess.discrete_nostructure.environment import DiscreteMDP
from statrl.settings.utils import Dirac
import numpy as np

class RiverSwim(DiscreteMDP):
    """The RiverSwim hard-exploration benchmark MDP.

    The agent starts at the left bank of a river of ``nbStates`` states. Going
    left always succeeds and pays a small reward at the leftmost state. Going
    right pays a large reward at the rightmost state but succeeds only with
    probability ``rightProbaright``, and may even drift back. An agent must
    therefore give up a certain small reward for many steps to reach an
    uncertain large one.

    Parameters
    ----------
    nbStates : int
        Number of states, i.e. the length of the river. The longer it is, the
        harder exploration becomes.
    rightProbaright : float, default=0.6
        Probability that swimming right moves right.
    rightProbaLeft : float, default=0.05
        Probability that swimming right drifts left instead.
    rewardL : float, default=0.1
        Reward for going left at the leftmost state.
    rewardR : float, default=0.99
        Reward for going right at the rightmost state.
    name : str, default='RiverSwim'
        Label used in logfiles, plot titles, and dump filenames.

    Attributes
    ----------
    nameActions : list of str
        ``["R", "L"]`` — action 0 is right, action 1 is left.

    See Also
    --------
    ErgodicRiverSwim : A variant where every state stays reachable under both actions.

    Examples
    --------
    >>> env = RiverSwim(6)
    >>> env.nS, env.nA
    (6, 2)
    >>> env.getMeanReward(5, 0)      # large reward at the far bank
    0.99
    """

    def __init__(self, nbStates, rightProbaright=0.6, rightProbaLeft=0.05, rewardL=0.1,
                 rewardR=0.99,name="RiverSwim"):  # , ergodic=False):
        self.nS = nbStates
        self.nA = 2
        self.states = range(0, self.nS)
        self.actions = range(0, self.nA)
        self.nameActions = ["R", "L"]

        self.startdistribution = np.zeros((self.nS))
        self.startdistribution[0] = 1.
        self.rewards = {}
        self.P = {}
        self.transitions = {}
        # Initialize a RiverSwim MDP
        for s in self.states:
            self.P[s] = {}
            self.transitions[s] = {}
            # GOING RIGHT
            self.transitions[s][0] = {}
            self.P[s][0] = []  # 0=right", 1=left
            li = self.P[s][0]
            prr = 0.
            if (s < self.nS - 1):
                li.append((rightProbaright, s + 1, False))
                self.transitions[s][0][s + 1] = rightProbaright
                prr = rightProbaright
            prl = 0.
            if (s > 0):
                li.append((rightProbaLeft, s - 1, False))
                self.transitions[s][0][s - 1] = rightProbaLeft
                prl = rightProbaLeft
            li.append((1. - prr - prl, s, False))
            self.transitions[s][0][s] = 1. - prr - prl

            self.P[s][1] = []  # 0=right", 1=left
            self.transitions[s][1] = {}
            li = self.P[s][1]
            if (s > 0):
                li.append((1., s - 1, False))
                self.transitions[s][1][s - 1] = 1.
            else:
                li.append((1., s, False))
                self.transitions[s][1][s] = 1.

            self.rewards[s] = {}
            if (s == self.nS - 1):
                self.rewards[s][0] = Dirac(rewardR)
            else:
                self.rewards[s][0] = Dirac(0.)
            if (s == 0):
                self.rewards[s][1] = Dirac(rewardL)
            else:
                self.rewards[s][1] = Dirac(0.)

        # print("Rewards : ", self.rewards, "\nTransitions : ", self.transitions)

        super(RiverSwim, self).__init__(self.nS, self.nA, self.P, self.rewards, self.startdistribution,
                                        self.nameActions,name=name)



class ErgodicRiverSwim(DiscreteMDP):
    """RiverSwim variant in which swimming left may still drift right.

    Adds a small ``ergodic`` leak to the left action, so every state remains
    reachable under every policy. Algorithms whose guarantees assume an
    ergodic MDP need this variant rather than plain :class:`RiverSwim`.

    Parameters
    ----------
    nbStates : int
        Number of states.
    rightProbaright : float, default=0.6
        Probability that swimming right moves right.
    rightProbaLeft : float, default=0.05
        Probability that swimming right drifts left instead.
    rewardL : float, default=0.1
        Reward for going left at the leftmost state.
    rewardR : float, default=1.0
        Reward for going right at the rightmost state.
    ergodic : float, default=0.001
        Leak probability on the left action; half of it moves right and the
        rest stays put. Larger values make exploration easier and the instance
        less discriminating.
    name : str, default='RiverSwim'
        Label used in logfiles and figures. Shares
        :class:`RiverSwim`'s default, so pass a distinct name when comparing
        the two in one results folder.

    See Also
    --------
    RiverSwim : The non-ergodic original.
    """

    def __init__(self, nbStates, rightProbaright=0.6, rightProbaLeft=0.05, rewardL=0.1,
                 rewardR=1., ergodic=0.001, name="RiverSwim"):  # , ergodic=False):
        self.nS = nbStates
        self.nA = 2
        self.states = range(0, self.nS)
        self.actions = range(0, self.nA)
        self.nameActions = ["R", "L"]

        self.startdistribution = np.zeros((self.nS))
        self.startdistribution[0] = 1.
        self.rewards = {}
        self.P = {}
        self.transitions = {}
        # Initialize a RiverSwim MDP
        for s in self.states:
            self.P[s] = {}
            self.transitions[s] = {}
            # GOING RIGHT
            self.transitions[s][0] = {}
            self.P[s][0] = []  # 0=right", 1=left
            li = self.P[s][0]
            prr = 0.
            if (s < self.nS - 1):
                li.append((rightProbaright, s + 1, False))
                self.transitions[s][0][s + 1] = rightProbaright
                prr = rightProbaright
            prl = 0.
            if (s > 0):
                li.append((rightProbaLeft, s - 1, False))
                self.transitions[s][0][s - 1] = rightProbaLeft
                prl = rightProbaLeft
            li.append((1. - prr - prl, s, False))
            self.transitions[s][0][s] = 1. - prr - prl

            self.P[s][1] = []  # 0=right", 1=left
            self.transitions[s][1] = {}
            li = self.P[s][1]
            plr = 0.
            pll = 0.
            if (s > 0):
                li.append((1.-ergodic, s - 1, False))
                self.transitions[s][1][s - 1] = 1.-ergodic
                pll = 1.-ergodic
            if (s < self.nS - 1):
                li.append((ergodic/2, s + 1, False))
                self.transitions[s][1][s + 1] = ergodic/2
                plr = ergodic/2
            li.append((1. - plr - pll, s, False))
            self.transitions[s][1][s] = 1. - plr - pll

            self.rewards[s] = {}
            if (s == self.nS - 1):
                self.rewards[s][0] = Dirac(rewardR)
            else:
                self.rewards[s][0] = Dirac(0.)
            if (s == 0):
                self.rewards[s][1] = Dirac(rewardL)
            else:
                self.rewards[s][1] = Dirac(0.)

        # print("Rewards : ", self.rewards, "\nTransitions : ", self.transitions)

        super(ErgodicRiverSwim, self).__init__(self.nS, self.nA, self.P, self.rewards, self.startdistribution,
                                        self.nameActions,name=name)