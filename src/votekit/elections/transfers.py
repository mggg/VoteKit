from fractions import Fraction
import random

from ..ballot import Ballot
from ..utils import remove_cand


def fractional_transfer(
    winner: str, ballots: list[Ballot], votes: dict, threshold: int
) -> list[Ballot]:
    """
    Calculates fractional transfer from winner, then removes winner
    from the list of ballots

    Args:
        winner: Candidate to transfer votes from
        ballots: List of Ballot objects
        votes: Contains candidates and their corresponding vote totals
        threshold: Value required to be elected, used to calculate transfer value

    Returns:
        Modified ballots with transfered weights and the winning canidated removed
    """
    transfer_value = (votes[winner] - threshold) / votes[winner]

    for ballot in ballots:
        new_ranking = []
        if ballot.ranking and ballot.ranking[0] == {winner}:
            ballot.weight = ballot.weight * transfer_value
            for cand in ballot.ranking:
                if cand != {winner}:
                    new_ranking.append(cand)

    return remove_cand(winner, ballots)


def random_transfer(
    winner: str, ballots: list[Ballot], votes: dict, threshold: int
) -> list[Ballot]:
    """
    Cambridge-style transfer where transfer ballots are selected randomly

    Args:
        winner: Candidate to transfer votes from
        ballots: List of Ballot objects
        votes: Contains candidates and their corresponding vote totals
        threshold: Value required to be elected, used to calculate transfer value

    Returns:
        Modified ballots with transfered weights and the winning canidated removed
    """

    # turn all of winner's ballots into (multiple) ballots of weight 1
    weight_1_ballots = []
    for ballot in ballots:
        if ballot.ranking and ballot.ranking[0] == {winner}:
            # note: under random transfer, weights should always be integers
            for _ in range(int(ballot.weight)):
                weight_1_ballots.append(
                    Ballot(
                        id=ballot.id,
                        ranking=ballot.ranking,
                        weight=Fraction(1),
                        voters=ballot.voters,
                    )
                )

    # remove winner's ballots
    ballots = [
        ballot
        for ballot in ballots
        if not (ballot.ranking and ballot.ranking[0] == {winner})
    ]

    surplus_ballots = random.sample(weight_1_ballots, int(votes[winner]) - threshold)
    ballots += surplus_ballots

    transfered = remove_cand(winner, ballots)

    return transfered


def seqRCV_transfer(
    winner: str, ballots: list[Ballot], votes: dict, threshold: int
) -> list[Ballot]:
    """
    Transfer method Sequential RCV elections

    Args:
        winner: Candidate to transfer votes from
        ballots: List of Ballot objects
        votes: Contains candidates and their corresponding vote totals
        threshold: Value required to be elected, used to calculate transfer value

    Returns:
        Original list of ballots as Sequential RCV does not transfer votes
    """
    return ballots
