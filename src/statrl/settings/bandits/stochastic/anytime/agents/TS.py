import numpy as np

from statrl.settings.bandits.stochastic.anytime.agent import BanditAgent
from statrl.settings.utils import randmax


class TS(BanditAgent):
    """
    Thompson Sampling for Bernoulli multi-armed bandits.

    Thompson Sampling is a Bayesian algorithm that maintains a posterior
    probability distribution over the mean reward of each arm. At every
    decision step, one sample is drawn from each posterior distribution,
    and the arm with the largest sampled value is selected.

    For Bernoulli rewards, the Beta distribution is conjugate to the
    Bernoulli likelihood. Assuming an initial Beta(1,1) prior, the
    posterior after observing S successes and F failures is

        Beta(S + 1, F + 1).

    This implementation maintains the posterior parameters through the
    cumulative rewards and pull counts of each arm.

    Attributes
    ----------
    nbDraws : numpy.ndarray
        Number of times each arm has been selected.

    cumRewards : numpy.ndarray
        Cumulative rewards (number of successes for Bernoulli rewards)
        obtained from each arm.

    theta : numpy.ndarray
        Random samples drawn from each arm's posterior distribution.
        These values constitute the Thompson indices used for arm
        selection.
    """

    def __init__(self, nbArms, name="TS"):
        """
        Construct a Thompson Sampling learner.

        Parameters
        ----------
        nbArms : int
            Number of arms.
        """
        self.nA = nbArms
        if name==None:
            BanditAgent.__init__(self, name="TS")
        else:
            BanditAgent.__init__(self, name=name)

    def reset(self):
        """
        Reset the learner before a new experiment.

        Initializes the sufficient statistics defining the Beta posterior
        distributions of every arm.
        """
        self.nbDraws = np.zeros(self.nA)
        self.cumRewards = np.zeros(self.nA)
        self.theta = np.zeros(self.nA)

    def select_arm(self):
        """
        Select the arm to play.

        Returns
        -------
        int
            Index of the arm whose sampled posterior mean is largest.

        Notes
        -----
        One Thompson sample is drawn for every arm during the previous
        update step. The arm associated with the largest sample is played.
        """
        return randmax(self.theta)

    def update(self, arm, reward):
        """
        Update the posterior distribution after observing a reward.

        Parameters
        ----------
        arm : int
            Index of the selected arm.

        reward : float
            Observed Bernoulli reward (typically 0 or 1).

        Notes
        -----
        Assuming a Beta(1,1) prior, the posterior distribution of arm
        ``a`` after observing

        * S_a successes
        * F_a failures

        is

            Beta(S_a + 1, F_a + 1).

        After updating the sufficient statistics, one posterior sample is
        generated independently for every arm. These samples are used by
        :meth:`play` to choose the next action.
        """
        self.cumRewards[arm] += reward
        self.nbDraws[arm] += 1

        self.theta = np.array([
            np.random.beta(
                max(self.cumRewards[a], 0) + 1,
                max(self.nbDraws[a] - self.cumRewards[a], 0) + 1,
            )
            for a in range(self.nA)
        ])