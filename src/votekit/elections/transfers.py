import random
import math
from votekit.ballot import RankBallot
from typing import Union
from votekit.pref_profile import RankProfile
from numpy.typing import NDArray
import numpy as np

def fractional_transfer(
    winner: str,
    fpv: float,
    ballots: Union[tuple[RankBallot], list[RankBallot]],
    threshold: int,
) -> tuple[RankBallot, ...]:
    """
    Calculates fractional transfer from winner, then removes winner from the list of ballots.

    Args:
        winner (str): Candidate to transfer votes from.
        fpv (float): Number of first place votes for winning candidate.
        ballots (Union[tuple[RankBallot], list[RankBallot]]): List of Ballot objects.
        threshold (int): Value required to be elected, used to calculate transfer value.

    Returns:
        tuple[Ballot,...]:
            Modified ballots with transferred weights and the winning candidate removed.
    """
    transfer_value = (fpv - threshold) / fpv

    transfered_ballots = [RankBallot()] * len(ballots)
    for i, ballot in enumerate(ballots):
        if ballot.ranking is not None:
            # if winner is first place, transfer ballot with fractional weight
            if ballot.ranking[0] == {winner}:
                transfered_weight = ballot.weight * transfer_value
            else:
                transfered_weight = ballot.weight
            # remove winner from ballot
            new_ranking = tuple(
                [frozenset([c for c in s if c != winner]) for s in ballot.ranking]
            )
            new_ranking = tuple([s for s in new_ranking if len(s) != 0])

            transfered_ballots[i] = RankBallot(
                ranking=new_ranking,
                weight=transfered_weight,
                voter_set=ballot.voter_set,
            )
        else:
            raise TypeError(f"Ballot {ballot} has no ranking.")

    return RankProfile(
        ballots=tuple([b for b in transfered_ballots if b.ranking and b.weight > 0])
    ).ballots


def random_transfer(
    winner: str,
    fpv: float,
    ballots: Union[tuple[RankBallot], list[RankBallot]],
    threshold: int,
) -> tuple[RankBallot, ...]:
    """
    Cambridge-style transfer where transfer ballots are selected randomly.
    All ballots must have integer weights.

    Args:
        winner (str): Candidate to transfer votes from.
        fpv (float): Number of first place votes for winning candidate.
        ballots (Union[tuple[RankBallot], list[RankBallot]]): List of Ballot objects.
        threshold (int): Value required to be elected, used to calculate transfer value.

    Returns:
        tuple[RankBallot,...]:
            Modified ballots with transferred weights and the winning candidate removed.
    """

    # turn all of winner's ballots into (multiple) ballots of weight 1
    winner_ballots = [RankBallot()] * len(ballots)
    updated_ballots = [RankBallot()] * len(ballots)

    winner_index = 0
    for i, ballot in enumerate(ballots):
        # under random transfer, weights should always be integers
        if not math.isclose(int(ballot.weight) - ballot.weight, 0):
            raise TypeError(f"Ballot {ballot} does not have integer weight.")

        if ballot.ranking is not None:
            # remove winner from ballot
            new_ranking = tuple(
                [frozenset([c for c in s if c != winner]) for s in ballot.ranking]
            )
            new_ranking = tuple([s for s in new_ranking if len(s) != 0])

            if ballot.ranking[0] == frozenset({winner}):
                new_ballots = [
                    RankBallot(
                        ranking=new_ranking,
                        weight=1,
                        voter_set=ballot.voter_set,
                    )
                ] * int(ballot.weight)
                winner_ballots[winner_index : (winner_index + len(new_ballots))] = (
                    new_ballots
                )
                winner_index += len(new_ballots)

            else:
                updated_ballots[i] = RankBallot(
                    ranking=new_ranking,
                    weight=ballot.weight,
                    voter_set=ballot.voter_set,
                )
        else:
            raise TypeError(f"Ballot {ballot} has no ranking.")

    surplus_ballots = random.sample(
        [b for b in winner_ballots if b.ranking], int(fpv) - threshold
    )
    updated_ballots += surplus_ballots

    return RankProfile(
        ballots=tuple([b for b in updated_ballots if b.ranking and b.weight > 0])
    ).ballots

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