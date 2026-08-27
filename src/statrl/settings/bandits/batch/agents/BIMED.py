from statrl.settings.bandits.batch.agent import BatchBanditAgent
from statrl.settings.utils import randmin,randmax, KLinf_threshold

import numpy as np
from math import log

"""
IMED variants: non-parametric version using KLinf_threshold
============================================================
All three classes use the empirical KL-divergence  Kinf(hat_F_a, mu*)
(computed via KLinf_threshold) instead of a parametric kl(mu_a, mu*).

This makes the algorithms distribution-free for any bounded reward
distribution with known upper bound `bound`.

Index formula (all variants):
    I_a = N_a * Kinf(hat_F_a, mu*) + log(N_a)

with the difference lying in *when* N_a and hat_F_a are updated:

  IMED        (3b adapted)  — N_a incremented & index refreshed at each
                              step inside the batch; hat_F_a updated at
                              end of batch.  The index only changes when
                              the *selected arm* changes (only N_a of the
                              pulled arm changes), so we skip redundant
                              recomputation of other arms.
  IMEDnaive   (3a naive)    — N_a *not* incremented during the batch; all
                              arms are drawn from the same index computed
                              before the batch starts → always returns the
                              same arm, so batchplay is O(1).
"""


class BIMED(BatchBanditAgent):
    """Batched IMED with within-batch sequential index updates.

    Uses the non-parametric index [1]_
    :math:`I_a = N_a K_{\\inf}(\\hat{F}_a, \\hat{\\mu}^\\star) + \\log N_a`,
    where :math:`K_{\\inf}` is computed by
    :func:`~statrl.settings.utils.KLinf_threshold`. Being non-parametric, it
    needs no assumption on the reward family beyond a known upper bound.

    Within a batch only :math:`N_a` moves — the reward histories, and so
    :math:`\\hat{F}_a`, stay frozen until the batch ends.

    Parameters
    ----------
    nbArms : int
        Number of arms.
    bound : float, default=1.0
        Known upper bound of the reward support, passed to
        :func:`~statrl.settings.utils.KLinf_threshold`. An understated bound
        invalidates the divergence.
    **kwargs : dict
        Ignored; accepted so the batched agents share a constructor signature.

    Attributes
    ----------
    nbDraws : ndarray of shape (nbArms,)
        Running pull count of each arm, updated within a batch.
    nbDraws_start : ndarray of shape (nbArms,)
        Pull count frozen at the last batch boundary.
    rewardHistory : list of list of float
        Rewards observed per arm; the empirical measure :math:`\\hat{F}_a`.
    indexes : ndarray of shape (nbArms,)
        Current index of each arm.
    x_threshold : float
        Within-batch inflation factor, ``log(t + B) / log(t)``, recomputed at
        the start of each batch. It grows with the batch size, letting a
        larger batch spread further from the frozen statistics.

    See Also
    --------
    statrl.settings.bandits.batch.agents.BatchIMED.BatchIMED :
        Caches :math:`K_{\\inf}` per arm instead of recomputing it.
    statrl.settings.bandits.stochastic.anytime.agents.IMED.IMED :
        The unbatched, parametric original.

    References
    ----------
    .. [1] Honda, J. and Takemura, A. "Non-asymptotic analysis of a new bandit
           algorithm for semi-bounded rewards." *Journal of Machine Learning
           Research*, 16(113):3721-3756, 2015.
    """

    def __init__(self, nbArms, bound=1.0, **kwargs):
        self.nbArms = nbArms
        self.bound = bound
        BatchBanditAgent.__init__(self, name="B-IMED")

    def reset(self):
        """Clear every statistic before a new independent run.

        Two pull counters are kept: ``nbDraws`` is the running total, while
        ``nbDraws_start`` freezes it at the last batch boundary. The index uses
        both, which is what lets an arm's index move within a batch while the
        empirical distribution it is based on stays fixed.
        """
        self.nbDraws = np.zeros(self.nbArms) # total number of draws of each arm
        self.nbDraws_start = np.zeros(self.nbArms) # number of draws of each arm at the start of an episode
        self.cumRewards = np.zeros(self.nbArms)
        self.meanRewards = np.zeros(self.nbArms, dtype=float)
        self.maxMeans = 0.0
        self.indexes = np.zeros(self.nbArms)
        self.rewardHistory = [[] for _ in range(self.nbArms)]

    def play(self):
        """Pick the minimal-index arm, unless it has outrun its batch budget.

        Returns
        -------
        int
            The minimal-index arm when it is still under the within-batch
            budget ``x_threshold * nbDraws_start``, or is itself the empirical
            leader, or has never been pulled. Otherwise the least-saturated
            arm by that same ratio, and failing that the empirical leader.
            The fallbacks stop a single batch from pouring all its pulls into
            one arm on the strength of statistics that cannot update until the
            batch ends.
        """
        a0=randmin(self.indexes)
        a1 = randmax(self.meanRewards)
        if (self.nbDraws_start[a0]==0) or (a0==a1) or (self.nbDraws[a0]<= np.ceil(self.x_threshold*self.nbDraws_start[a0])+1):
            return a0
        else:
            a2 = np.argmin([self.nbDraws[b]/max(self.nbDraws_start[b],1) for b in range(self.nbArms)])
            if (self.nbDraws_start[a2]==0) or (self.nbDraws[a2]<= np.ceil(self.x_threshold*self.nbDraws_start[a2])+1):
                return a2
            else:
                return a1

    def _update_index(self):
        for a in range(self.nbArms):
            self._update_index_arm(a)

    def _update_index_arm(self, a):
        """Recompute the index for a single arm (used in batchplay)."""
        n = self.nbDraws[a]
        nn = self.nbDraws_start[a]
        if (self.meanRewards[a] >= self.maxMeans):
            self.indexes[a] = self.x_threshold * log(max(nn, 1))
        else:
            if n > 0 and len(self.rewardHistory[a]) > 0:
                kinf = KLinf_threshold(self.rewardHistory[a], self.maxMeans,
                                       upper_bound=self.bound)
                self.indexes[a] = n * kinf + self.x_threshold * log(max(nn, 1))
            else:
                self.indexes[a] = self.x_threshold * log(max(nn, 1))

    # ------------------------------------------------------------------
    # Online interface
    # ------------------------------------------------------------------
    def update(self, arm, reward):
        """Record one ``(arm, reward)`` pair and recompute every index.

        The single-pull path, used outside batched interaction. Within a
        batch, :meth:`batchupdate` is called instead.

        Parameters
        ----------
        arm : int
            Index of the arm that was pulled.
        reward : float
            Reward observed for it.
        """
        self.rewardHistory[arm].append(reward)
        self.cumRewards[arm] += reward
        self.nbDraws[arm] += 1
        self.meanRewards[arm] = self.cumRewards[arm] / self.nbDraws[arm]
        self.maxMeans = float(np.max(self.meanRewards))
        self._update_index()

    # ------------------------------------------------------------------
    # Batch interface (adapted)
    # ------------------------------------------------------------------
    def batchplay(self, batchsize):
        """Commit a batch, refreshing only the pulled arm's index at each step.

        The reward histories (and each arm's empirical distribution
        :math:`\\hat{F}_a`) do not change during a batch, so a pull only moves
        :math:`N_a` for the arm chosen. Recomputing that one index instead of
        all of them brings the cost of a batch down from
        :math:`O(B \\cdot K \\cdot K_{\\inf})` to :math:`O(B \\cdot K_{\\inf})`.

        Parameters
        ----------
        batchsize : int
            Number of pulls in this batch.

        Returns
        -------
        list of int
            Exactly ``batchsize`` arm indices.
        """
        batcharms = []
        t = sum(self.nbDraws)
        self.x_threshold = np.log(t + batchsize) / np.log(t) if (t > 1) else 1.
        self._update_index()
        for _ in range(batchsize):
            a = self.play()
            batcharms.append(a)
            self.nbDraws[a] += 1
            self._update_index_arm(a)
        return batcharms

    def batchupdate(self, batcharm, batchreward):
        """Fold in the batch's rewards and re-freeze the pull counts.

        Extends each arm's reward history, refreshes its empirical mean, and
        snapshots ``nbDraws`` into ``nbDraws_start`` to open the next batch.

        Parameters
        ----------
        batcharm : list of int
            The arms that were pulled.
        batchreward : list of float
            The rewards observed for them.

        Notes
        -----
        ``nbDraws`` is not incremented here: :meth:`batchplay` already did so
        as it committed each pull.
        """
        arm_arr = np.asarray(batcharm)
        rew_arr = np.asarray(batchreward)
        for a in range(self.nbArms):
            mask = arm_arr == a
            if mask.any():
                rewards_a = rew_arr[mask]
                self.rewardHistory[a].extend(rewards_a.tolist())
                self.cumRewards[a] += rewards_a.sum()
                # nbDraws already incremented during batchplay
                self.meanRewards[a] = self.cumRewards[a] / self.nbDraws[a]
                self.nbDraws_start[a] = self.nbDraws[a]
        self.maxMeans = float(np.max(self.meanRewards))
        #self._update_index()

