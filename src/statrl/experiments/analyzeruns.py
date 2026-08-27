
import pickle
import time
import numpy as np

def computeScoreDiffs(names: list[str], dump_scores: list[list[str]], timeHorizon: int, envName: str, root_folder: str) -> tuple[list[np.ndarray], list[np.ndarray], list[np.ndarray],list[np.ndarray],list[np.ndarray], list[np.ndarray], list[int]]:
    """Turn per-replicate score dumps into regret statistics over time.

    Loads every dump, subtracts each agent's cumulative score from the
    oracle's averaged one to obtain regret, and summarizes the replicates by
    their mean, median, and four quantiles.

    Parameters
    ----------
    names : list of str
        Agent names, in the same order as ``dump_scores``. Used to name the
        per-agent regret pickles.
    dump_scores : list of list of str
        One list of dump filenames per agent. **The last entry must be the
        oracle's**, and it is what every other entry is compared against — the
        function has no other way to tell which agent is the reference.
    timeHorizon : int
        Number of rounds each run played.
    envName : str
        Environment name, used in the output filenames.
    root_folder : str
        Directory the regret pickles are written to.

    Returns
    -------
    mean, median : list of ndarray
        Per-agent mean and median regret at each sampled time.
    quantile1, quantile2, quantile3, quantile4 : list of ndarray
        Per-agent regret quantiles at levels 0.1, 0.25, 0.75, and 0.9. The
        plots shade 0.1-0.9 and 0.25-0.75 as nested bands.
    times : list of int
        Sampled time steps, shared by every returned series.

    Notes
    -----
    Long runs are downsampled to at most ~1000 points
    (``skip = timeHorizon // 1000``), which bounds both plot size and memory.
    Each returned series therefore has ``len(times)`` entries, not
    ``timeHorizon``.

    The oracle's score is averaged across its replicates *before* the
    subtraction, so the result is regret against mean oracle performance
    rather than a paired difference per replicate.
    """

    median = []
    mean = []
    quantile1 = []
    quantile2 = []
    quantile3 = []
    quantile4 = []
    nbAlgs = len(dump_scores) - 1

    #Downsample the times, especially in case timeHorizon is huge.
    skip = max(1, (timeHorizon // 1000))
    times = [t for t in range(0,timeHorizon,skip)]

    #file_oracle = open(dump_scores[-1], 'rb')
    #scores_oracle = pickle.load(file_oracle)
    # Comment the following line for BatchMabs:
    #scores_oracle = scores_oracle[0]
    #file_oracle.close()

    data_o = []
    for oracle_file in dump_scores[-1]:
        with open(oracle_file, 'rb') as file:
            scores_oi = pickle.load(file)
        data_o.append([scores_oi[t] for t in times])
    scores_oracle = np.mean(data_o, axis=0)

    for j in range(nbAlgs):
        data_j = []
        for alg_file in dump_scores[j]:
            with open(alg_file, 'rb') as file:
                scores_ij = pickle.load(file)
            data_j.append([scores_oracle[k] - scores_ij[t] for k, t in enumerate(times)])

        filename = f"{root_folder}regret_{envName}_{names[j]}_{timeHorizon}_{j}_{time.time()}"
        with open(filename, 'wb') as out_file:
            pickle.dump(data_j, out_file)

        mean.append(np.mean(data_j, axis=0))
        median.append(np.quantile(data_j, 0.5, axis=0))
        quantile1.append(np.quantile(data_j, 0.1, axis=0))
        quantile2.append(np.quantile(data_j, 0.25, axis=0))
        quantile3.append(np.quantile(data_j, 0.75, axis=0))
        quantile4.append(np.quantile(data_j, 0.9, axis=0))

    return mean,median,quantile1,quantile2, quantile3, quantile4,times

