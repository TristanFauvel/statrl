
from statrl.experiments.utils import dump

import time

from abc import ABC, abstractmethod
from typing import Any
import numpy as np

class Interaction(ABC):
    """Base class for the interaction loop of a setting. 
    """

    @abstractmethod
    def run(self, env: Any, learner: Any, horizon: int)-> np.ndarray:
        """Run one interaction and return its cumulative expected score.

        Parameters
        ----------
        env : object
            Environment of the setting. Reset by the implementation.
        learner : object
            Agent of the setting. Reset by the implementation, so one instance
            can serve many replicates.
        horizon : int
            Number of rounds to play.

        Returns
        -------
        ndarray of shape (horizon,)
            Cumulative *expected* reward. Implementations must return exactly
            ``horizon`` entries — :func:`oneRunWithDump` asserts it.
        """
        return np.array([])


    @abstractmethod
    def renderrun(self, env: Any, learner: Any, horizon: int) -> None:
        """Run one interaction with rendering enabled, returning no score.

        Parameters
        ----------
        env : object
            Environment of the setting; the implementation attaches renderers.
        learner : object
            Agent of the setting.
        horizon : int
            Number of rounds to play.
        """
        ...

    @property
    @abstractmethod
    def plotlabels(self) -> tuple[str, str]:
        """tuple of (str, str): Axis labels ``(x, y)`` for the regret plots.

        Read by
        :func:`~statrl.experiments.massiveruns.runLargeMulticoreExperiment`
        and forwarded to
        :func:`~statrl.experiments.plotruns.plotScoreDiffs`. Settings whose
        rounds are not time steps override it — the batch setting labels its
        x-axis by episode.
        """
        raise NotImplementedError




def oneRunWithDump(env: Any, learner: Any, interact: Any, timeHorizon: int, root_folder: str) -> str:
    """Run one replicate and pickle its score series to disk.

    The unit of work handed to :mod:`joblib` by
    :func:`~statrl.experiments.parallelruns.multicoreRuns`. Results travel
    back through the filesystem rather than through the return value, because
    returning full score series from every worker would serialize
    ``nbReplicates x timeHorizon`` floats through the process pool.

    Parameters
    ----------
    env : object
        Environment for this replicate; already a private deep copy.
    learner : object
        Agent for this replicate; already a private deep copy.
    interact : Interaction
        The interaction loop of the setting.
    timeHorizon : int
        Number of rounds to play.
    root_folder : str
        Directory the dump is written to. Must end with a separator, and must
        already exist.

    Returns
    -------
    str
        Path of the pickle written, of the form
        ``{root_folder}aux_{env}_scores_{agent}_{horizon}_{timestamp}``. The
        ``aux_`` prefix is what
        :func:`~statrl.experiments.utils.clear_auxiliaryfiles` deletes once
        the statistics have been computed.

    Raises
    ------
    AssertionError
        If the interaction returned fewer or more than ``timeHorizon`` scores.
    """
    scoretimeseries=interact.run(env,learner,timeHorizon)
    assert len(scoretimeseries) == timeHorizon

    tag = f"{env.name}_scores_{learner.name}_{timeHorizon}_{time.time()}"
    filename = dump(scoretimeseries,"aux",tag,root_folder)
    return filename