from statrl.settings.bandits.stochastic.batch.environment import BatchMAB

from statrl.settings.bandits.stochastic.anytime.envs.parametric import BernoulliBandit, GaussianBandit, TruncatedGaussianBandit
from statrl.settings.bandits.stochastic.batch.agents.baba_schedule import compute_baba_grid


import math

# Batch-size schedules — ell is the round index (0-based)
def _B_CONST(ell):
    return 10


def _B_LINEAR(ell):
    return int(ell + 1)


def _B_QUADRATIC(ell):
    return int((ell + 1) ** 2)


def _B_CUBIC(ell):
    return int((ell + 1) ** 3)


def _B_EXP(ell):
    return int(2 ** ell)


def _B_SEXP(ell):
    return int(math.exp(ell**1.5))


def _B_DOUBLE_EXP(ell):
    return int(math.exp(2 ** ell))


def _B_ABRUPT(ell):
    return 100 if (ell % 4 == 1) else int((ell + 1) ** 3)

def exotic_schedule(t):
    """Batch schedule cycling through linear, exponential, constant, and cubic.

    Parameters
    ----------
    t : int
        Round index.

    Returns
    -------
    int
        Batch size for round ``t``.
    """
    schedule= {0:_B_LINEAR, 1: _B_EXP, 2:_B_CONST, 3:_B_CUBIC}
    return schedule[(t % 4)](t)

_B_EXOTIC = exotic_schedule


def baba_schedule(horizon, nbArms):
    """Batch sizes of the BABA epoch grid, as a plain list.

    Parameters
    ----------
    horizon : int
        Number of rounds the grid is built for.
    nbArms : int
        Number of arms.

    Returns
    -------
    list of int
        One batch size per round. A list rather than a callable, so it stays
        picklable across the process pool used by
        :func:`~statrl.experiments.parallelruns.multicoreRuns`.

    See Also
    --------
    statrl.settings.bandits.stochastic.batch.agents.baba_schedule.compute_baba_grid :
        Returns the phase and epoch labels alongside these sizes.
    """
    batch_sizes, _, _, _ = compute_baba_grid(horizon, nbArms, None, alpha=3)
    return batch_sizes


schedule_catalogue= {"constant": _B_CONST, "linear": _B_LINEAR, "quadratic": _B_QUADRATIC, "cubic":_B_CUBIC,
            "exp": _B_EXP, "surexp": _B_SEXP,"doubleexp":_B_DOUBLE_EXP, "abrupt":_B_ABRUPT, "exotic":_B_EXOTIC
                     }

mean_catalogue = {"simple4": [0.1, 0.4, 0.7, 0.9],
                  "simple6": [0.2, 0.6, 0.8, 0.8, 0.95, 0.9]
                  }

class BatchGaussianBandit(BatchMAB):
    """Batched Gaussian bandit, built from means and a named batch schedule.

    Parameters
    ----------
    means : array-like of float or str
        Arm means, or a key of ``mean_catalogue`` (``"simple4"``,
        ``"simple6"``) naming a stock instance.
    vars : array-like of float
        Variance of each arm.
    batchschedule : str, default='constant'
        Key of ``schedule_catalogue`` — ``"constant"``, ``"linear"``,
        ``"quadratic"``, ``"cubic"``, ``"exp"``, ``"doubleexp"``,
        ``"abrupt"``, ``"exotic"`` — or ``"baba,<horizon>"``
        to use the BABA grid for that horizon.
    name : str, default='BMAB-Gaussian'
        Accepted but unused: the instance name comes from
        :class:`~statrl.settings.bandits.stochastic.batch.environment.BatchMAB`, which
        derives it from the wrapped bandit and the batch sizes.

    Raises
    ------
    KeyError
        If ``batchschedule`` or a string ``means`` names no catalogue entry.
    """

    def __init__(self, means, vars, batchschedule="constant", name="BMAB-Gaussian"):
        if (type(batchschedule) is str):
            if ("," in batchschedule):
                fct, horiz = batchschedule.split(",")
                horizon = int(horiz)
                schedule = baba_schedule(horizon, len(means))
            else:
                schedule = schedule_catalogue[batchschedule]
        if (type(means) is str):
            super(BatchGaussianBandit, self).__init__(
                GaussianBandit(means= mean_catalogue[means], vars=vars), schedule)
        else:
            super(BatchGaussianBandit, self).__init__(
            GaussianBandit(means=means, vars=vars),schedule)


class BatchBernoulliBandit(BatchMAB):
    """Batched Bernoulli bandit, built from means and a named batch schedule.

    Parameters
    ----------
    means : array-like of float or str
        Arm success probabilities, or a key of ``mean_catalogue``.
    batchschedule : str, default='constant'
        Key of ``schedule_catalogue``, or ``"baba,<horizon>"``.
    name : str, default='BMAB-Bernoulli'
        Accepted but unused; see :class:`BatchGaussianBandit`.

    Raises
    ------
    KeyError
        If ``batchschedule`` or a string ``means`` names no catalogue entry.
    """

    def __init__(self, means, batchschedule="constant", name="BMAB-Bernoulli"):
        if (type(batchschedule) is str):
            if ("," in batchschedule):
                fct, horiz = batchschedule.split(",")
                horizon = int(horiz)
                schedule = baba_schedule(horizon, len(means))
            else:
                schedule = schedule_catalogue[batchschedule]
        if (type(means) is str):
            super(BatchBernoulliBandit, self).__init__(
                BernoulliBandit(means=mean_catalogue[means]), schedule)
        else:
            super(BatchBernoulliBandit, self).__init__(
            BernoulliBandit(means=means),schedule)


class BatchTruncatedGaussianBandit(BatchMAB):
    """Batched bandit with Gaussian arms truncated to ``[low, high]``.

    Parameters
    ----------
    means : array-like of float or str
        Means of the untruncated Gaussians, or a key of ``mean_catalogue``.
    sigma : float, default=0.3
        Common standard deviation before truncation.
    low, high : float, default=-1.0, 1.0
        Bounds of the reward support.
    batchschedule : str, default='constant'
        Key of ``schedule_catalogue``, or ``"baba,<horizon>"``.
    name : str, default='BMAB-TGaussian'
        Accepted but unused; see :class:`BatchGaussianBandit`.

    Raises
    ------
    KeyError
        If ``batchschedule`` or a string ``means`` names no catalogue entry.
    """

    def __init__(self, means, sigma: float = 0.3, low: float = -1.0, high: float = 1.0, batchschedule="constant", name="BMAB-TGaussian"):

        if (type(batchschedule) is str):
            if ("," in batchschedule):
                fct, horiz = batchschedule.split(",")
                horizon = int(horiz)
                schedule = baba_schedule(horizon, len(means))
            else:
                schedule = schedule_catalogue[batchschedule]
        if (type(means) is str):
            super(BatchTruncatedGaussianBandit, self).__init__(
                TruncatedGaussianBandit(means=mean_catalogue[means],sigma= sigma, low=low, high=high), schedule)
        else:
            super(BatchTruncatedGaussianBandit, self).__init__(
            TruncatedGaussianBandit(means=means, sigma= sigma, low=low, high=high),schedule)