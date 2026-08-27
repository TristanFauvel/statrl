


from typing import Sequence, Union

import numpy as np
from statrl.settings.bandits.stochastic.anytime.environment import StochasticBanditEnv
import statrl.settings.bandits.stochastic.anytime.envs.distributions as distributions

## some functions that create specific MABs

Means = Union[Sequence[float], np.ndarray]


def BernoulliBandit(means: Means, name: str = "MAB-Bernoulli") -> StochasticBanditEnv:
    """Build a Bernoulli bandit from a vector of arm means.

    Parameters
    ----------
    means : array-like of float
        One success probability per arm, each in :math:`[0, 1]`.
    name : str, default='MAB-Bernoulli'
        Prefix of the instance name; the means are appended to it so that
        different instances produce different dump filenames.

    Returns
    -------
    ~statrl.settings.bandits.stochastic.anytime.environment.StochasticBanditEnv
        A bandit whose rewards are in ``{0, 1}``. Pair it with
        :class:`~statrl.settings.bandits.stochastic.anytime.agents.IMED.IMED`
        using :func:`~statrl.settings.utils.klBern`.

    Examples
    --------
    >>> env = BernoulliBandit([0.2, 0.9, 0.5])
    >>> env.number_arms, env.optimal_arm
    (3, 1)
    """
    s="-".join(str(m) for m in means)
    name = f'{name}-means-{s}'
    return StochasticBanditEnv([distributions.Bernoulli(p) for p in means], name=name)

def BinomialBandit(means: Means, repetitions: int = 200, name: str = "MAB-Binomial") -> StochasticBanditEnv:
    """Build a Binomial bandit from a vector of per-trial success probabilities.

    Parameters
    ----------
    means : array-like of float
        One success probability per arm. Note each arm's *reward* mean is
        ``repetitions * means[a]``, not ``means[a]``.
    repetitions : int, default=200
        Number of Bernoulli trials summed into one reward.
    name : str, default='MAB-Binomial'
        Prefix of the instance name.

    Returns
    -------
    ~statrl.settings.bandits.stochastic.anytime.environment.StochasticBanditEnv
        A bandit with integer rewards in ``range(repetitions + 1)``.
    """
    s="-".join(str(m) for m in means)
    name = f'{name}{repetitions}-means-{s}'
    return StochasticBanditEnv([distributions.Binomial(repetitions, p) for p in means], name=name)

def GaussianBandit(means: Means, vars: Means, name: str = "MAB-Gaussian") -> StochasticBanditEnv:
    """Build a Gaussian bandit from vectors of means and variances.

    Parameters
    ----------
    means : array-like of float
        Mean reward of each arm.
    vars : array-like of float
        Variance of each arm; must have the same length as ``means``.
    name : str, default='MAB-Gaussian'
        Prefix of the instance name. Only the means are appended to it, so two
        instances differing solely in their variances share a name — pass a
        distinct ``name`` when comparing those.

    Returns
    -------
    ~statrl.settings.bandits.stochastic.anytime.environment.StochasticBanditEnv
        A bandit with unbounded rewards. Pair it with
        :class:`~statrl.settings.bandits.stochastic.anytime.agents.IMED.IMED`
        using :func:`~statrl.settings.utils.klGauss`.
    """
    s="-".join(str(m) for m in means)
    name = f'{name}-means-{s}'
    return StochasticBanditEnv([distributions.Gaussian(m, v) for m,v in zip(means, vars)], name=name)

def TruncatedGaussianBandit(means: Means, sigma: float = 0.5, low: float = -1.0, high: float = 1.0, name: str = "MAB-TruncGaussian") -> StochasticBanditEnv:
    """Build a bandit whose arms are Gaussians truncated to ``[low, high]``. 

    Parameters
    ----------
    means : array-like of float
        Mean of each *untruncated* Gaussian. Each arm's true mean is the
        truncated one, which differs whenever the bounds are not symmetric
        around ``means[a]``.
    sigma : float, default=0.5
        Common standard deviation before truncation.
    low, high : float, default=-1.0, 1.0
        Bounds of the reward support.
    name : str, default='MAB-TruncGaussian'
        Prefix of the instance name; ``sigma`` is appended to it.

    Returns
    -------
    ~statrl.settings.bandits.stochastic.anytime.environment.StochasticBanditEnv
        A bandit with rewards in ``[low, high]``.
    """
    s = "-".join(str(m) for m in means)
    name = f'{name}-means-{s}-sigma{sigma}'
    return StochasticBanditEnv([distributions.TruncatedGaussian(p, sigma=sigma, low=low, high=high) for p in means], name=name)


def RandomBernoulliBandit(Delta: float, K: int, name: str = "MAB-RandomBernoulli") -> StochasticBanditEnv:
    """Draw a random Bernoulli instance with a prescribed optimality gap.

    Useful for studying how regret scales with the gap: the difficulty of a
    bandit instance is governed by :math:`\\Delta`, so sweeping it while
    holding ``K`` fixed isolates that dependence.

    Parameters
    ----------
    Delta : float
        Gap between the best and second-best arm. The remaining arms get means
        drawn uniformly below the second-best one.
    K : int
        Number of arms.
    name : str, default='MAB-RandomBernoulli'
        Prefix of the instance name.

    Returns
    -------
    ~statrl.settings.bandits.stochastic.anytime.environment.StochasticBanditEnv
        A Bernoulli bandit whose two leading means differ by exactly ``Delta``.
 
    """
    maxMean = Delta + np.random.rand() * (1. - Delta)
    secondmaxMean = maxMean - Delta
    means = secondmaxMean * np.random.random(K)
    bestarm, secondbestarm = np.random.choice(K, 2, replace=False)
    means[bestarm] = maxMean
    means[secondbestarm] = secondmaxMean
    return BernoulliBandit(means, name=name)
