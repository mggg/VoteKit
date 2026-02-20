from numpy.typing import NDArray
import numpy as np

def numpy_random_transfer(
    fpv_vec: NDArray, wt_vec: NDArray, winner: int, surplus: int
) -> NDArray:
    """
    Samples s row indices to transfer from an implicit pool,
    where each row index i appears wt_vec[i] times if fpv_vec[i] == winner.
    Returns a counts vector where counts[i] is the number of times row i was sampled.
    Ensures sum(counts) == s and counts[i] <= wt_vec[i].

    Args:
        fpv_vec (NDArray): First-preference vector.
        wt_vec (NDArray): Integer weights vector.
        winner (int): Candidate code whose ballots are to be transferred.
        surplus (int): Number of surplus votes to transfer.
    """
    rng = np.random.default_rng()

    # running example: assume that candidate 2 just won.
    # assume the fpv_vec looks like [2,5,3,2]
    # then eligible looks like [True, False, False, True]
    # and winner_row_indices looks like [0, 3]
    eligible = fpv_vec == winner
    winner_row_indices = np.flatnonzero(eligible)

    # assume the original weight vector was [200, 100, 50, 25]
    # then wts looks like [200, 25]
    wts = wt_vec[winner_row_indices].astype(np.int64)

    # assume that quota was 220, so winner 2 had 5 surplus votes and 225 transferable votes
    transferable = int(wts.sum())

    # this deals with cases where there are fewer than surplus votes to transfer
    # (lots of exhausted ballots)
    surplus = min(surplus, transferable)

    # Sample surplus distinct positions in the implicit pool [0, transferable)
    # in our example: we sample 5 distinct numbers from [0, 225)
    positions_to_transfer = rng.choice(transferable, size=surplus, replace=False)
    positions_to_transfer.sort()

    # Say we sampled the numbers 12, 50, 178, 200, and 201
    # numbers 0 through 199 inclusive get mapped to the first bin, so the first three sampled
    # votes go to winner_row_index[0]
    # numbers 200 and 201 get mapped to the second bin, so they go to our second
    #  winner_row_index[1]
    bins = np.cumsum(wts)  # len = len(idx)
    owners = np.searchsorted(
        bins, positions_to_transfer, side="right"
    )  # values in winner_row_indices

    # Accumulate counts back to global rows
    counts_local = np.bincount(owners, minlength=winner_row_indices.size)
    counts = np.zeros(fpv_vec.shape[0], dtype=np.int64)
    counts[winner_row_indices] = (
        counts_local  # this tells us how many times each row was sampled as indexed in the
        # global ballot_matrix
    )
    return counts