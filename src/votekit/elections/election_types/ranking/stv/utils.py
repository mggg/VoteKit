from numpy.typing import NDArray
import numpy as np


def numpy_random_transfer(
    fpv_vec: NDArray, wt_vec: NDArray, winner: int, surplus: int
) -> NDArray:
    """
    Samples ``surplus``  row indices to transfer from an implicit pool.

    Each row index i appears wt_vec[i] times if fpv_vec[i] == winner.
    Returns a counts vector where counts[i] is the number of times row i was sampled.
    Ensures sum(counts) == s and counts[i] <= wt_vec[i].

    Example:
        Assume candidate 2 just won.
        Let fpv_vec = [2, 5, 3, 2]. Then eligible_for_transfer is
        [True, False, False, True], and winner_row_indices is [0, 3].
        Let wt_vec = [200, 100, 50, 25]. Then wts is [200, 25].
        If the quota was 220, then winner 2 had 5 surplus votes and
        225 transferable votes, so maximum_transferable is 225 and
        transferred_votes is 5. We sample 5 distinct numbers from
        [0, 225), for example 12, 50, 178, 200, 201. Numbers 0 through
        199 map to the first bin (winner_row_indices[0]), and 200 and 201
        map to the second bin (winner_row_indices[1]).

    Args:
        fpv_vec (NDArray): First-preference vector.
        wt_vec (NDArray): Integer weights vector.
        winner (int): Candidate code whose ballots are to be transferred.
        surplus (int): Number of surplus votes to transfer.

    Returns:
        counts (NDArray): Vector of counts
    """
    rng = np.random.default_rng()
    eligible_for_transfer = fpv_vec == winner
    winner_row_indices = np.flatnonzero(eligible_for_transfer)
    wts = wt_vec[winner_row_indices].astype(np.int64)
    maximum_transferable = int(wts.sum())
    transferred_votes = min(surplus, maximum_transferable)
    positions_to_transfer = rng.choice(
        maximum_transferable, size=transferred_votes, replace=False
    )
    positions_to_transfer.sort()
    bins = np.cumsum(wts)
    owners = np.searchsorted(bins, positions_to_transfer, side="right")
    counts_local = np.bincount(owners, minlength=winner_row_indices.size)
    counts = np.zeros(fpv_vec.shape[0], dtype=np.int64)
    counts[winner_row_indices] = counts_local
    return counts
