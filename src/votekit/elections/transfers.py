from fractions import Fraction
import random

from ..ballot import Ballot
from ..utils import remove_cand


def fractional_transfer(
    winner: str, ballots: list[Ballot], votes: dict, threshold: int
) -> list[Ballot]:
    """
    Calculates fractional transfer from winner, then removes winner
    from the list of ballots.

    Args:
        winner: Candidate to transfer votes from.
        ballots: List of Ballot objects.
        votes: Contains candidates and their corresponding vote totals.
        threshold: Value required to be elected, used to calculate transfer value.

    Returns:
        Modified ballots with transfered weights and the winning candidate removed.
    """
    transfer_value = (votes[winner] - threshold) / votes[winner]

    transfered_ballots = []
    for ballot in ballots:
        new_ranking = []
        if ballot.ranking and ballot.ranking[0] == {winner}:
            transfered_weight = ballot.weight * transfer_value
            for cand in ballot.ranking:
                if cand != {winner}:
                    new_ranking.append(cand)
            transfered_ballots.append(
                Ballot(
                    ranking=new_ranking,
                    weight=transfered_weight,
                    voter_set=ballot.voter_set,
                    id=ballot.id,
                )
            )
        else:
            transfered_ballots.append(ballot)

    return remove_cand(winner, transfered_ballots)


def random_transfer(
    winner: str, ballots: list[Ballot], votes: dict, threshold: int
) -> list[Ballot]:
    """
    Cambridge-style transfer where transfer ballots are selected randomly.

    Args:
        winner: Candidate to transfer votes from.
        ballots: List of Ballot objects.
        votes: Contains candidates and their corresponding vote totals.
        threshold: Value required to be elected, used to calculate transfer value.

    Returns:
        Modified ballots with transferred weights and the winning candidate removed.
    """

    # turn all of winner's ballots into (multiple) ballots of weight 1
    weight_1_ballots = []
    updated_ballots = []
    for ballot in ballots:
        if ballot.ranking and ballot.ranking[0] == {winner}:
            # note: under random transfer, weights should always be integers
            for _ in range(int(ballot.weight)):
                weight_1_ballots.append(
                    Ballot(
                        id=ballot.id,
                        ranking=ballot.ranking,
                        weight=Fraction(1),
                        voter_set=ballot.voter_set,
                    )
                )
        else:
            updated_ballots.append(ballot)

    surplus_ballots = random.sample(weight_1_ballots, int(votes[winner]) - threshold)
    updated_ballots += surplus_ballots

    return remove_cand(winner, updated_ballots)


def seqRCV_transfer(
    winner: str, ballots: list[Ballot], votes: dict, threshold: int
) -> list[Ballot]:
    """
    Transfer method for Sequential RCV elections.

    Args:
        winner: Candidate to transfer votes from.
        ballots: List of Ballot objects.
        votes: Contains candidates and their corresponding vote totals.
        threshold: Value required to be elected, used to calculate transfer value.

    Returns:
        Original list of ballots as Sequential RCV does not transfer votes.
    """
    return ballots
