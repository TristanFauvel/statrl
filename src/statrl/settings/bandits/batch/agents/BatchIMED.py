from statrl.settings.bandits.batch.agent import BatchBanditAgent
from statrl.settings.utils import randmin,randmax, KLinf_threshold

import numpy as np
from math import log

"""
IMED variants — non-parametric version using KLinf_threshold
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


class BatchIMED(BatchBanditAgent):
    """Batched IMED caching :math:`K_{\\inf}` across each batch.

    Same non-parametric index [1]_ as
    :class:`~statrl.settings.bandits.batch.agents.BIMED.BIMED`, but
    :math:`K_{\\inf}` is evaluated once per arm in :meth:`batchupdate` and held
    in ``kinfs`` for the whole batch. Since the reward histories cannot change
    mid-batch, that value is exactly what a recomputation would return — so
    the caching is free of approximation, and makes a batch's cost independent
    of its size.

    Parameters
    ----------
    nbArms : int
        Number of arms.
    bound : float, default=1.0
        Known upper bound of the reward support.
    batchagnostic : bool, default=False
        If True, ``x_threshold`` is advanced pull by pull as though the agent
        were unbatched, rather than set once from the batch size. The agent
        then behaves the same whatever the batch schedule — useful as a
        control, and reflected in its name (``ABatchIMED``).
    **kwargs : dict
        Ignored; accepted so the batched agents share a constructor signature.

    Attributes
    ----------
    kinfs : ndarray of shape (nbArms,)
        Cached :math:`K_{\\inf}` per arm, refreshed once per batch. Zero for
        the empirical leader and for arms with no observations.
    nbDraws, nbDraws_start : ndarray of shape (nbArms,)
        Running and batch-boundary pull counts.
    x_threshold : float
        Within-batch inflation factor.

    See Also
    --------
    statrl.settings.bandits.batch.agents.BIMED.BIMED :
        Recomputes :math:`K_{\\inf}` inside the batch.
    BatchIMED2 : A variant whose eligibility test weighs information rather than counts.

    References
    ----------
    .. [1] Honda, J. and Takemura, A. "Non-asymptotic analysis of a new bandit
           algorithm for semi-bounded rewards." *Journal of Machine Learning
           Research*, 16(113):3721-3756, 2015.
    """

    def __init__(self, nbArms, bound=1.0, batchagnostic=False, **kwargs):
        self.nbArms = nbArms
        self.bound = bound
        self.batchagnostic = batchagnostic
        name= "ABatchIMED" if self.batchagnostic else "BatchIMED"
        BatchBanditAgent.__init__(self, name=name)

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
        self.kinfs = np.zeros(self.nbArms)
        self.rewardHistory = [[] for _ in range(self.nbArms)]

    def play(self):
        """Minimize the index over the arms still eligible within this batch.

        An arm is eligible if it ties the empirical leader, or has not yet
        exhausted its within-batch budget ``x_threshold * nbDraws_start``.
        Restricting the minimization keeps a batch from over-committing to one
        arm on statistics frozen at the batch boundary.

        Returns
        -------
        int
            The eligible arm of minimal index.
        """
        a1 = randmax(self.meanRewards)
        a0 = randmin(np.array([self.indexes[a] for a in range(self.nbArms) if (self.meanRewards[a]==self.meanRewards[a1]) or (self.nbDraws[a]<=np.floor(self.x_threshold*self.nbDraws_start[a])+1)]))
        return a0

    def _update_index(self):
        for a in range(self.nbArms):
            self._update_index_arm(a)

    def _update_index_arm(self, a):
        """Recompute the index for a single arm (used in batchplay)."""
        n = self.nbDraws[a]
        nn = self.nbDraws_start[a]
        self.indexes[a] = n * self.kinfs[a] + self.x_threshold * log(max(nn, 1))

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

        The reward histories — and hence each arm's empirical distribution
        :math:`\\hat{F}_a` — do not change during a batch, so a pull only moves
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
        if self.batchagnostic:
            self.x_threshold = np.log(t + 1) / np.log(t) if (t > 1) else 1.
        else:
            self.x_threshold = np.log(t + batchsize) / np.log(t) if (t > 1) else 1.
        self._update_index()
        for b in range(batchsize):
            a = self.play()
            batcharms.append(a)
            self.nbDraws[a] += 1
            if self.batchagnostic:
                self.x_threshold = np.log(t + b+2) / np.log(t) if (t > 1) else 1.
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
        for a in range(self.nbArms):
            if (self.meanRewards[a] < self.maxMeans) and (len(self.rewardHistory[a]) > 0):
                self.kinfs[a] = KLinf_threshold(self.rewardHistory[a], self.maxMeans,
                                upper_bound=self.bound)
            else:
                self.kinfs[a]=0


class BatchIMED2(BatchBanditAgent):
    """Batched IMED whose within-batch budget is measured in information.

    Differs from :class:`BatchIMED` only in :meth:`play`: the eligibility test
    compares :math:`N_a K_{\\inf}` against its value at the batch boundary,
    throttling an arm by the information it has accumulated rather than by its
    pull count. Arms whose :math:`K_{\\inf}` is near zero — those close to the
    leader — are therefore barely throttled at all.

    Parameters
    ----------
    nbArms : int
        Number of arms.
    bound : float, default=1.0
        Known upper bound of the reward support.
    batchagnostic : bool, default=False
        If True, advance ``x_threshold`` pull by pull as an unbatched agent
        would; the instance is then named ``B-IMED (doubly)``.
    **kwargs : dict
        Ignored; accepted so the batched agents share a constructor signature.

    Attributes
    ----------
    kinfs : ndarray of shape (nbArms,)
        Cached :math:`K_{\\inf}` per arm, refreshed once per batch.
    batchsize : int
        Size of the batch currently being committed.

    See Also
    --------
    BatchIMED : The count-based variant.
    """

    def __init__(self, nbArms, bound=1.0, batchagnostic=False, **kwargs):
        self.nbArms = nbArms
        self.bound = bound
        self.batchagnostic = batchagnostic
        name= "B-IMED (doubly)" if self.batchagnostic else "B-IMED"
        BatchBanditAgent.__init__(self, name=name)

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
        self.kinfs = np.zeros(self.nbArms)
        self.rewardHistory = [[] for _ in range(self.nbArms)]

    def play(self):
        """Take the minimal-index arm while its budget holds, else the leader.

        The budget test compares :math:`N_a K_{\\inf}` against
        ``x_threshold`` times its value at the batch boundary, so it throttles
        an arm by the *information* already accrued rather than by pull count.

        Returns
        -------
        int
            The minimal-index arm when it is still within budget, otherwise
            the current empirical leader.
        """
        a1 = randmax(self.meanRewards)
        # THE FOLLOWING RULE fails:
        #a = randmin([self.indexes[a] for a in range(self.nbArms) if (self.meanRewards[a]==self.meanRewards[a1]) or (self.nbDraws[a]<=np.floor(self.x_threshold*self.nbDraws_start[a])+1)])
        #return a

        a0 =  randmin(self.indexes)
        if (self.nbDraws[a0]* self.kinfs[a0] <= self.x_threshold*self.nbDraws_start[a0]* self.kinfs[a0]):
        #if (self.meanRewards[a0]==self.meanRewards[a1]) or (self.nbDraws[a0]<=np.floor(self.x_threshold*self.nbDraws_start[a0])+1):
            return a0
        #a2 = randmin([self.indexes[a] for a in range(self.nbArms) if self.indexes[a]<=self.indexes[a1]])
        #if (self.meanRewards[a2] == self.meanRewards[a1]) or (self.nbDraws[a2] <= np.floor(self.x_threshold * self.nbDraws_start[a2]) + 1):
        #    return a2
        return a1

    def play_alter(self):
       """Alternative selection rule, kept for comparison and unused by default.

       Minimizes the *pull count* over arms whose index does not exceed the
       leader's, instead of minimizing the index itself. Nothing in the class
       calls this; :meth:`play` is the rule in force.

       Returns
       -------
       int
           The selected arm.
       """
       # Alternative version:
       a1 = randmax(self.meanRewards)

       a0 = randmin(np.array([self.nbDraws[a] for a in range(self.nbArms) if self.indexes[a]<=self.indexes[a1]]))
       if (self.nbDraws[a0] * self.kinfs[a0] <= self.x_threshold * self.nbDraws_start[a0] * self.kinfs[a0]):
           return a0
       return a1


    def _update_index(self):
        for a in range(self.nbArms):
            self._update_index_arm(a)

    def _update_index_arm(self, a):
        """Recompute the index for a single arm (used in batchplay)."""
        n = self.nbDraws[a]
        nn = self.nbDraws_start[a]
        self.indexes[a] = n * self.kinfs[a] + self.x_threshold * log(max(nn, 1))
       # Alternative version:
       # if self.meanRewards[a]==max(self.meanRewards):
       #     self.indexes[a] = log(sum(self.nbDraws_start)+ self.batchsize)
       # else:
       #     self.indexes[a] = n * self.kinfs[a] + self.x_threshold * log(max(nn, 1))

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

        The reward histories — and hence each arm's empirical distribution
        :math:`\\hat{F}_a` — do not change during a batch, so a pull only moves
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
        self.batchsize=batchsize
        t = sum(self.nbDraws)
        if self.batchagnostic:
            self.x_threshold =np.log(t + 1) / np.log(t) if (t > 1) else 1.
        else:
            self.x_threshold =  np.log(t + batchsize) / np.log(t) if (t > 1) else 1.
        self._update_index()
        for b in range(batchsize):
            a = self.play()
            batcharms.append(a)
            self.nbDraws[a] += 1
            if  self.batchagnostic:
                self.x_threshold = np.log(t + b+2) / np.log(t) if (t > 1) else 1.
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
        for a in range(self.nbArms):
            if (self.meanRewards[a] < self.maxMeans) and (len(self.rewardHistory[a]) > 0):
                self.kinfs[a] = KLinf_threshold(self.rewardHistory[a], self.maxMeans,
                                upper_bound=self.bound)
            else:
                self.kinfs[a]=0
        #self._update_index()