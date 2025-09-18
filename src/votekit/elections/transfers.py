import random
import math
from votekit.ballot import RankBallot
from typing import Union
from votekit.pref_profile import RankProfile


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
