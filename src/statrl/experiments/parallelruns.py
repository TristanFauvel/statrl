import time
import copy
from joblib import Parallel, delayed
from typing import Any, Callable


## Parallelization
def multicoreRuns(env: Any, learner: Any, interact: Any, nbReplicates: int, timeHorizon: int, oneRunFunction: Callable[..., Any], root_folder: str) -> tuple[Any, float]:
    """Run one agent for many independent replicates, spread across CPU cores.

    Each replicate gets its own deep copy of the environment, agent, and
    interaction.

    Parameters
    ----------
    env : object
        Environment to replicate.
    learner : object
        Agent to replicate.
    interact : statrl.experiments.onerun.Interaction
        Interaction loop of the setting.
    nbReplicates : int
        Number of independent runs.
    timeHorizon : int
        Number of rounds per run.
    oneRunFunction : callable
        Function executing one replicate, called as
        ``oneRunFunction(env, learner, interact, timeHorizon, root_folder)``.
        In practice :func:`~statrl.experiments.onerun.oneRunWithDump`.
    root_folder : str
        Directory the per-replicate dumps are written to.

    Returns
    -------
    scores : list of str
        One dump filename per replicate, in the order the jobs were created.
    elapsed : float
        Mean wall-clock seconds per replicate. Since the runs are concurrent
        this is total elapsed time divided by ``nbReplicates``, so it measures
        throughput rather than the cost of a single run.

    Notes
    -----
    Uses all available cores (``n_jobs=-1``). Everything passed in must be
    picklable, which is why
    :class:`~statrl.settings.bandits.stochastic.batch.environment.BatchMAB` accepts a
    plain list of batch sizes rather than only a callable.
    """
    #FIXME  Should be made more general? indep of gymnasium?
        #envs.append(gymnasium.make(envRegisterName).unwrapped)
    jobs = [
        (copy.deepcopy(env), copy.deepcopy(learner), copy.deepcopy(interact), timeHorizon, root_folder)
        for _ in range(nbReplicates)
    ]

    t0 = time.time()
    scores = Parallel(n_jobs=-1)(delayed(oneRunFunction)(*job) for job in jobs)
    elapsed = time.time() - t0

    return scores, elapsed / nbReplicates