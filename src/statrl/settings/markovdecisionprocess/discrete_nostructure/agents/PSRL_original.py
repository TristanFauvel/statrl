"""
Posterior Sampling Reinforcement Learning (PSRL) - original reference version.

This module preserves, verbatim, the PSRL implementation as it stood
before the bug fixe. It is kept for reference only.

Relative to the fixed version, this implementation has three known bugs
in :meth:`PSRLOriginal.VI`, each marked inline below:

1. The span-seminorm convergence test took ``abs(u1 - u0)`` before
   computing the span, which can cause premature termination (the span
   seminorm must be applied to the *signed* difference; Puterman & Chan,
   "Markov Decision Processes and Reinforcement Learning", Ch. 7,
   Algorithm 7.1).
2. The iteration cap check (``itera > max_iter``, with ``itera``
   starting at 0) let the loop run ``max_iter + 2`` sweeps before
   warning, instead of ``max_iter``.
3. The greedy policy was written on every sweep from ``u0`` (the
   not-yet-converged iterate), while ``self.u`` ends up storing ``u1``
   (the newer, converged iterate) - so the policy that is kept is the
   one computed against the previous (stale) VI iterate, not the
   converged one.

    Reference : Osband, Russo and Van Roy,"(More) Efficient Reinforcement Learning via Posterior Sampling",
    NeurIPS 2013. 
"""


import numpy as np
import scipy.stats as stat
from statrl.settings.markovdecisionprocess.discrete_nostructure.agent import MDPAgent
from statrl.settings.utils import allmax, categorical_sample


class PSRLOriginal(MDPAgent):
    """
    Posterior Sampling Reinforcement Learning (pre-fix reference version).

    PSRL is a Bayesian model-based reinforcement learning algorithm for
    finite Markov Decision Processes.

    Rather than constructing optimistic confidence sets, PSRL maintains
    posterior distributions over the unknown reward function and
    transition probabilities.

    At the beginning of each episode, a complete MDP is sampled from the
    posterior. The optimal policy of this sampled MDP is computed and
    executed until a stopping criterion triggers the beginning of a new
    episode.

    This implementation assumes

    * finite state and action spaces;
    * Bernoulli rewards;
    * Beta priors over rewards;
    * Dirichlet priors over transitions.

    Parameters
    ----------
    nS : int
        Number of states.

    nA : int
        Number of actions.

    delta : float, optional
        Confidence parameter. Present for compatibility with the common
        learner interface but not explicitly used by the PSRL algorithm.

    Attributes
    ----------
    t : int
        Global interaction time.

    policy : ndarray of shape (nS, nA)
        Current stochastic policy computed on the sampled MDP.

    u : ndarray of shape (nS,)
        Bias (relative value) function obtained by Value Iteration.

    Nk : ndarray of shape (nS, nA)
        Cumulative number of visits to every state-action pair over all
        completed episodes.

    vk : ndarray of shape (nS, nA)
        State-action visit counts during the current episode.

    r_successCounts :
        Beta posterior α parameters.

    r_failureCounts :
        Beta posterior β parameters.

    p_pseudoCounts :
        Dirichlet pseudo-counts defining the posterior transition model.

    r_sampled :
        Reward function sampled from the posterior.

    p_sampled :
        Transition kernel sampled from the posterior.
    """

    def __init__(self, nS, nA, delta=0.05):
        """
        Construct a Posterior Sampling Reinforcement Learning learner.

        The learner starts with independent conjugate priors for every
        state-action pair.

        Rewards are modeled using Beta distributions

            Beta(1,1),

        corresponding to a uniform prior over Bernoulli means.

        Transition probabilities are modeled using symmetric Dirichlet
        priors

            Dirichlet(1,...,1),

        corresponding to a uniform prior over successor states.
        """
        MDPAgent.__init__(self, nS, nA, name="PSRL")
        # ---------------------------------------------------------
        # Problem dimensions
        # ---------------------------------------------------------

        self.nS = nS
        self.nA = nA

        # ---------------------------------------------------------
        # Global interaction statistics
        # ---------------------------------------------------------

        self.t = 1
        self.delta = delta

        # Complete interaction history
        self.observations = [[], [], []]

        # ---------------------------------------------------------
        # Episode statistics
        # ---------------------------------------------------------

        # Number of visits during the current episode.
        self.vk = np.zeros((self.nS, self.nA))

        # Cumulative visits over previous episodes.
        self.Nk = np.zeros((self.nS, self.nA))

        self.Nkmax = 0

        # ---------------------------------------------------------
        # Planning variables
        # ---------------------------------------------------------

        # Current policy computed on the sampled MDP.
        self.policy = np.zeros((self.nS, self.nA))

        # Relative value (bias) function.
        self.u = np.zeros(self.nS)

        # ---------------------------------------------------------
        # Bayesian posterior over rewards
        # ---------------------------------------------------------

        # Alpha parameters of Beta posteriors.
        self.r_successCounts = np.ones((self.nS, self.nA))

        # Beta parameters of Beta posteriors.
        self.r_failureCounts = np.ones((self.nS, self.nA))

        # ---------------------------------------------------------
        # Bayesian posterior over transitions
        # ---------------------------------------------------------

        self.p_pseudoCounts = np.ones((self.nS, self.nA, self.nS))

        # ---------------------------------------------------------
        # Sampled MDP
        # ---------------------------------------------------------

        # Reward function sampled from the posterior.
        self.r_sampled = np.zeros((self.nS, self.nA))

        # Transition kernel sampled from the posterior.
        self.p_sampled = np.zeros((self.nS, self.nA, self.nS))


    # To reinitialize the learner with a given initial state inistate.
    def reset(self, inistate):
        self.t = 1
        self.observations = [[inistate], [], []]
        self.vk = np.zeros((self.nS, self.nA))
        self.Nk = np.zeros((self.nS, self.nA))
        self.Nkmax = 0
        self.u = np.zeros(self.nS)
        self.policy = np.zeros((self.nS, self.nA))

        self.r_successCounts = np.ones((self.nS, self.nA))
        self.r_failureCounts = np.ones((self.nS, self.nA))
        self.p_pseudoCounts = np.ones((self.nS, self.nA, self.nS))


        self.r_sampled = np.zeros((self.nS, self.nA))
        self.p_sampled = np.zeros((self.nS, self.nA, self.nS))

        self.new_episode()


    # The Extend Value Iteration algorithm (approximated with precision epsilon), in parallel policy updated with the greedy one.
    def VI(self, epsilon=0.01, max_iter=1000):

        u0 = self.u - min(self.u)
        u1 = np.zeros(self.nS)
        itera = 0

        while True:
            for s in range(self.nS):
                temp = np.zeros(self.nA)
                for a in range(self.nA):
                    # print("Support of ", s,a," : ", self.supports[s, a], ", ", support)
                    p = self.p_sampled[s, a]  # Allowed to sum  to <=1
                    # print("Max_p of ",s,a, " : ", max_p)
                    temp[a] = self.r_sampled[s, a] + sum([u0[ns] * p[ns] for ns in range(self.nS)])

                # BUG (fixed in the current PSRL by computing the policy once,
                # after VI has converged): this writes self.policy on every
                # sweep from u0, the not-yet-converged iterate, while
                # self.u below ends up storing u1 (the newer iterate) -- so
                # the kept policy is greedy w.r.t. a stale VI iterate.
                # This implements a tie-breaking rule by choosing:  Uniform(Argmmin(Nk))
                (u1[s], arg) = allmax(temp)
                nn = [-self.Nk[s, a] for a in arg]
                (nmax, arg2) = allmax(nn)
                choice = [arg[a] for a in arg2]
                self.policy[s] = [1. / len(choice) if x in choice else 0 for x in range(self.nA)]

            # BUG (fixed in the current PSRL): the span seminorm must be
            # applied to the signed difference u1 - u0, not abs(u1 - u0);
            # taking abs() here can trigger premature convergence.
            diff = [abs(x - y) for (x, y) in zip(u1, u0)]
            if (max(diff) - min(diff)) < epsilon:
                self.u = u1 - min(u1)
                break
            elif itera > max_iter:
                # BUG (fixed in the current PSRL): itera starts at 0 and this
                # check is `itera > max_iter`, so the loop runs
                # max_iter + 2 sweeps before warning instead of max_iter.
                self.u = u1 - min(u1)
                print("[PSRL] No convergence in the VI at time ", self.t, " before ", max_iter, " iterations.")
                break
            else:
                u0 = u1 - min(u1)
                u1 = np.zeros(self.nS)
                itera += 1

    def new_episode(self):
        self.sumratios = 0.
        self.updateN()

        for s in range(self.nS):
            for a in range(self.nA):
                self.r_sampled[s,a] = stat.beta.rvs(self.r_successCounts[s,a],self.r_failureCounts[s,a])
                p = stat.dirichlet.rvs(alpha = self.p_pseudoCounts[s,a])
                p=p[0]
                self.p_sampled[s,a] = [p[ns] for ns in range(self.nS)]


        self.VI(epsilon=1. / max(1, self.t))

    ###### Steps and updates functions ######

    # Auxiliary function to update N the current state-action count.
    def updateN(self):
        self.Nkmax = 0.
        for s in range(self.nS):
            for a in range(self.nA):
                self.Nk[s, a] += self.vk[s, a]
                self.Nkmax = max(self.Nkmax, self.Nk[s, a])
                self.vk[s, a] = 0

    # To chose an action for a given state (and start a new episode if necessary -> stopping criterion defined here).
    def play(self, state):
        action = categorical_sample([self.policy[state, a] for a in range(self.nA)], np.random)
        # if self.sumratios >= 1.:  # Stoppping criterion
        if self.vk[state, action] >= max([1, self.Nk[state, action]]):  # Stopping criterion
            self.new_episode()
            action = categorical_sample([self.policy[state, a] for a in range(self.nA)], np.random)
        return action

    # To update the learner after one step of the current policy.
    def update(self, state, action, reward, observation):
        self.vk[state, action] += 1
        self.observations[0].append(observation)
        self.observations[1].append(action)
        self.observations[2].append(reward)

        self.r_successCounts[state,action] += reward
        self.r_failureCounts[state,action] += 1.-reward

        #print(state,action,observation)
        self.p_pseudoCounts[state,action,observation] +=1

        self.t += 1
