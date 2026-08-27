from statrl.settings.bandits.batch.agent import BatchBanditAgent
from statrl.settings.utils import randmax

import numpy as np

"""
BCB — Bounded CVaR Bandit (Gautron et al., 2024)
=================================================
We implement the parameter choice CVaR = Expectation (alpha -> 1), which
corresponds to Non-Parametric Thompson Sampling with a Dirichlet prior
anchored at the upper bound B of the reward support.

References
----------
Gautron et al. (2024) "Bandits with Bounded CVaR Constraints".
"""


class BCB(BatchBanditAgent):
    """BCB with CVaR = Expectation (adapted batch version).

    In each batch, the arm counts are updated *sequentially* inside the batch
    (optimistic within-batch exploration), but reward histories (used for the
    Dirichlet draw) are only updated at the end of the batch via batchupdate.

    Parameters
    ----------
    nbArms : int
    bound : float
        Upper bound B of the reward support.  The Dirichlet prior is
        initialised with a single pseudo-observation at B.
    """

    def __init__(self, nbArms, bound=1.0):
        self.nbArms = nbArms
        self.bound = bound
        BatchBanditAgent.__init__(self, name="BCB-adapted")

    def reset(self):
        """Clear every statistic and restore the prior.

        Each arm's reward history restarts with a single pseudo-observation at
        the upper bound ``B``. That optimistic anchor is what drives
        exploration: an arm with few observations still has appreciable
        posterior mass near ``B``.
        """
        self.nbDraws = np.zeros(self.nbArms)
        self.cumRewards = np.zeros(self.nbArms)
        self.meanRewards = np.zeros(self.nbArms, dtype=float)
        # Each arm starts with one pseudo-observation at the upper bound B
        self.rewardHistory = [[self.bound] for _ in range(self.nbArms)]

    # ------------------------------------------------------------------
    # Core Dirichlet sampling
    # ------------------------------------------------------------------
    def _dirichletmean(self, rewards):
        w = np.random.dirichlet(np.ones(len(rewards)))
        return float(np.dot(w, rewards))

    def play(self):
        """Draw one posterior mean per arm and play the best.

        Returns
        -------
        int
            Arm with the highest Dirichlet-reweighted mean this draw. Ties are
            broken uniformly at random.
        """
        return randmax([self._dirichletmean(self.rewardHistory[a])
                        for a in range(self.nbArms)])

    # ------------------------------------------------------------------
    # Online (non-batch) interface
    # ------------------------------------------------------------------
    def update(self, arm, reward):
        """Record one ``(arm, reward)`` pair in the counts and the history.

        Parameters
        ----------
        arm : int
            Index of the arm that was pulled.
        reward : float
            Reward observed for it; appended to that arm's history, which is
            the empirical measure the Dirichlet draw reweights.
        """
        self.cumRewards[arm] += reward
        self.nbDraws[arm] += 1
        self.meanRewards[arm] = self.cumRewards[arm] / self.nbDraws[arm]
        self.rewardHistory[arm].append(reward)

    # ------------------------------------------------------------------
    # Batch interface (adapted: counts updated during play)
    # ------------------------------------------------------------------
    def batchplay(self, batchsize):
        """Fill the whole batch with a single posterior draw's winner.

        Parameters
        ----------
        batchsize : int
            Number of pulls in this batch.

        Returns
        -------
        list of int
            ``batchsize`` copies of one arm.

        Notes
        -----
        The scores depend only on the reward histories, which do not change
        during a batch, so every draw within the batch would select the same
        arm. The winner is computed once instead of ``batchsize`` times. Its
        count is incremented optimistically up front, keeping ``nbDraws``
        consistent with what :meth:`batchupdate` assumes.
        """ 
        scores = np.array([self._dirichletmean(self.rewardHistory[a])
                  for a in range(self.nbArms)])
        a = randmax(scores)
        self.nbDraws[a] += batchsize   # optimistic count increment
        return [a] * batchsize

    def batchupdate(self, batcharm, batchreward):
        """Append the batch's rewards to the arm histories and refresh the means.

        Parameters
        ----------
        batcharm : list of int
            The arms that were pulled.
        batchreward : list of float
            The rewards observed for them.

        Notes
        -----
        ``nbDraws`` is *not* incremented here: :meth:`batchplay` already did so
        optimistically when it committed the batch.
        """
        arm_arr = np.asarray(batcharm)
        rew_arr = np.asarray(batchreward)
        for a in range(self.nbArms):
            mask = arm_arr == a
            if mask.any():
                rewards_a = rew_arr[mask]
                self.cumRewards[a] += rewards_a.sum()
                self.rewardHistory[a].extend(rewards_a.tolist())
        # Recompute means from cumRewards and actual nbDraws
        # (nbDraws was pre-incremented in batchplay, stays consistent)
        for a in range(self.nbArms):
            if self.nbDraws[a] > 0:
                self.meanRewards[a] = self.cumRewards[a] / self.nbDraws[a]


class BCBnaif(BatchBanditAgent):
    """BCB without the optimistic within-batch count increment.

    Differs from :class:`BCB` in one respect: the pull counts are left
    untouched during :meth:`batchplay` and updated only at the end of the
    batch. Equivalent to drawing ``batchsize`` i.i.d. actions from the current
    policy and updating afterwards. Kept as the reference point that isolates
    what the optimistic increment buys.

    Parameters
    ----------
    nbArms : int
        Number of arms.
    bound : float, default=1.0
        Upper bound ``B`` of the reward support; the Dirichlet prior is
        anchored on a single pseudo-observation there.

    See Also
    --------
    BCB : The adapted version, with the within-batch increment.
    """

    def __init__(self, nbArms, bound=1.0):
        self.nbArms = nbArms
        self.bound = bound
        BatchBanditAgent.__init__(self, name="BCB")

    def reset(self):
        """Clear every statistic and restore the anchored Dirichlet prior."""
        self.nbDraws = np.zeros(self.nbArms)
        self.cumRewards = np.zeros(self.nbArms)
        self.meanRewards = np.zeros(self.nbArms, dtype=float)
        self.rewardHistory = [[self.bound] for _ in range(self.nbArms)]

    def _dirichletmean(self, rewards):
        w = np.random.dirichlet(np.ones(len(rewards)))
        return float(np.dot(w, rewards))

    def play(self):
        """Draw one posterior mean per arm and play the best.

        Returns
        -------
        int
            Arm with the highest Dirichlet-reweighted mean this draw.
        """
        return randmax([self._dirichletmean(self.rewardHistory[a])
                        for a in range(self.nbArms)])

    def update(self, arm, reward):
        """Record one ``(arm, reward)`` pair in the counts and the history.

        Parameters
        ----------
        arm : int
            Index of the arm that was pulled.
        reward : float
            Reward observed for it.
        """
        self.cumRewards[arm] += reward
        self.nbDraws[arm] += 1
        self.meanRewards[arm] = self.cumRewards[arm] / self.nbDraws[arm]
        self.rewardHistory[arm].append(reward)

    def batchplay(self, batchsize):
        """Fill the whole batch with a single posterior draw's winner.

        Parameters
        ----------
        batchsize : int
            Number of pulls in this batch.

        Returns
        -------
        list of int
            ``batchsize`` copies of one arm. Unlike :meth:`BCB.batchplay`, no
            count is incremented here.
        """
        scores = [self._dirichletmean(self.rewardHistory[a])
                  for a in range(self.nbArms)]
        a = randmax(np.array(scores))
        return [a] * batchsize

    def batchupdate(self, batcharm, batchreward):
        """Fold the batch's rewards into the counts, histories, and means.

        Parameters
        ----------
        batcharm : list of int
            The arms that were pulled.
        batchreward : list of float
            The rewards observed for them.
        """
        arm_arr = np.asarray(batcharm)
        rew_arr = np.asarray(batchreward)
        for a in range(self.nbArms):
            mask = arm_arr == a
            if mask.any():
                rewards_a = rew_arr[mask]
                self.cumRewards[a] += rewards_a.sum()
                self.nbDraws[a] += mask.sum()
                self.rewardHistory[a].extend(rewards_a.tolist())
        for a in range(self.nbArms):
            if self.nbDraws[a] > 0:
                self.meanRewards[a] = self.cumRewards[a] / self.nbDraws[a]