from fractions import Fraction
import random
import math
from ..ballot import Ballot
from typing import Union
from ..pref_profile import PreferenceProfile


def fractional_transfer(
    winner: str,
    fpv: Union[Fraction, float],
    ballots: Union[tuple[Ballot], list[Ballot]],
    threshold: int,
) -> tuple[Ballot, ...]:
    """
    Calculates fractional transfer from winner, then removes winner from the list of ballots.

    Args:
        winner (str): Candidate to transfer votes from.
        fpv (Union[Fraction, float]): Number of first place votes for winning candidate.
        ballots (Union[tuple[Ballot], list[Ballot]]): List of Ballot objects.
        threshold (int): Value required to be elected, used to calculate transfer value.

    Returns:
        tuple[Ballot,...]:
            Modified ballots with transferred weights and the winning candidate removed.
    """
    transfer_value = (fpv - threshold) / fpv

    transfered_ballots = [Ballot()] * len(ballots)
    for i, ballot in enumerate(ballots):
        if ballot.ranking:
            # if winner is first place, transfer ballot with fractional weight
            if ballot.ranking[0] == {winner}:
                transfered_weight = ballot.weight * Fraction(transfer_value)
            else:
                transfered_weight = ballot.weight
            # remove winner from ballot
            new_ranking = tuple(
                [frozenset([c for c in s if c != winner]) for s in ballot.ranking]
            )
            new_ranking = tuple([s for s in new_ranking if len(s) != 0])

            transfered_ballots[i] = Ballot(
                ranking=new_ranking,
                weight=transfered_weight,
                voter_set=ballot.voter_set,
                id=ballot.id,
            )
        else:
            raise TypeError(f"Ballot {ballot} has no ranking.")

    return (
        PreferenceProfile(
            ballots=tuple([b for b in transfered_ballots if b.ranking and b.weight > 0])
        )
        .condense_ballots()
        .ballots
    )


def random_transfer(
    winner: str,
    fpv: Union[Fraction, float],
    ballots: Union[tuple[Ballot], list[Ballot]],
    threshold: int,
) -> tuple[Ballot, ...]:
    """
    Cambridge-style transfer where transfer ballots are selected randomly.
    All ballots must have integer weights.

    Args:
        winner (str): Candidate to transfer votes from.
        fpv (Union[Fraction, float]): Number of first place votes for winning candidate.
        ballots (Union[tuple[Ballot], list[Ballot]]): List of Ballot objects.
        threshold (int): Value required to be elected, used to calculate transfer value.

    Returns:
        tuple[Ballot,...]:
            Modified ballots with transferred weights and the winning candidate removed.
    """

    # turn all of winner's ballots into (multiple) ballots of weight 1
    winner_ballots = [Ballot()] * len(ballots)
    updated_ballots = [Ballot()] * len(ballots)

    winner_index = 0
    for i, ballot in enumerate(ballots):
        # under random transfer, weights should always be integers
        if not math.isclose(int(ballot.weight) - ballot.weight, 0):
            raise TypeError(f"Ballot {ballot} does not have integer weight.")

        if ballot.ranking:
            # remove winner from ballot
            new_ranking = tuple(
                [frozenset([c for c in s if c != winner]) for s in ballot.ranking]
            )
            new_ranking = tuple([s for s in new_ranking if len(s) != 0])

            if ballot.ranking[0] == frozenset({winner}):
                new_ballots = [
                    Ballot(
                        id=ballot.id,
                        ranking=new_ranking,
                        weight=Fraction(1),
                        voter_set=ballot.voter_set,
                    )
                ] * int(ballot.weight)
                winner_ballots[
                    winner_index : (winner_index + len(new_ballots))
                ] = new_ballots
                winner_index += len(new_ballots)

            else:
                updated_ballots[i] = Ballot(
                    id=ballot.id,
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

    return (
        PreferenceProfile(
            ballots=tuple([b for b in updated_ballots if b.ranking and b.weight > 0])
        )
        .condense_ballots()
        .ballots
    )
