from typing import Callable, Literal, Sequence

import pandas as pd

from votekit.ballot import RankBallot
from votekit.pref_profile import RankProfile
from votekit.types import Candidate


def transform_ballots_profile(
    profile: RankProfile,
    transform_ballot_func: Callable[[RankBallot | list[Candidate]], RankBallot | list[Candidate]],
    on: Literal["candidates", "ballot"] = "ballot",
    group_ballots: bool = True,
) -> RankProfile:
    """
    Applies an user-defined transformation function upon every ballot within a profile.

    Can group ballots to alter by ballot type instead of per ballot. The function can accept
    list of candidates that represent a ranking or a ballot and must return the same type.
    If a list of candidates is expected, then the profile cannot contain tied candidates or
    unexpected behavior will occur. Transformed ballots are not regrouped by ballot type.

    Args:
        profile (RankProfile): profile to transform.
        transform_ballot_func (Callable[[RankBallot | list[Candidate]],
            RankBallot | list[Candidate]]): ballot transform function.
        on (Literal["candidates', 'ballot']): data type to apply transform function on.
            RankBallot by default. Can select candidates for list of candidates. Weights and Voter
            Sets are preserved for list of candidates as well as ballots.
        group_ballots (bool): Groups ballots with identical rankings and updates the weight
            accordingly. True by default. Can set to False if ballots are altered in a probalistic
            or per-ballot manner.
    Returns:
        RankProfile: altered rank profile.

    """
    if not isinstance(profile, RankProfile):
        raise TypeError("Can only alter ballots from a RankProfile.")

    if group_ballots:
        profile = profile.group_ballots()

    new_ballots = None
    df = pd.DataFrame()
    if on == "candidates":
        df = profile.df.copy()
        ranking_cols = [col for col in df.columns if "Ranking_" in col]
        ballot_values = df[ranking_cols].to_numpy()
        new_full_rankings = []
        for i, row in enumerate(ballot_values):
            ranking = []
            for cand_set in row:
                if cand_set == frozenset({"~"}):
                    continue
                if len(cand_set) > 1:
                    raise ValueError(
                        f"Tied candidates {cand_set} found at ballot {i}."
                        ' on="candidates" require untied rankings. Use "ballot"'
                        " instead."
                    )
                ranking.append(next(iter(cand_set)))

            new_ranking = transform_ballot_func(ranking)
            if not isinstance(new_ranking, list):
                raise TypeError(
                    f"Transform returned ={type(new_ranking)}. Expected list of candidates."
                )
            if not any(
                isinstance(cand, Candidate) and not isinstance(cand, bool) for cand in new_ranking
            ):
                raise TypeError(
                    f"Transform returned non-string/int candidates: {type(new_ranking)}."
                )

            if len(new_ranking) > profile.max_ranking_length:
                raise ValueError(
                    f"Transform returned {len(new_ranking)} candidates for ballot {i}."
                    f" Profile's max_ranking_length is {profile.max_ranking_length}."
                )

            new_full_rankings.append(
                [
                    [frozenset({cand}) for cand in new_ranking]
                    + (profile.max_ranking_length - len(new_ranking)) * [frozenset({"~"})]
                ]
            )

        df[ranking_cols] = new_full_rankings

        return RankProfile(
            candidates=profile.candidates, max_ranking_length=profile.max_ranking_length, df=df
        )

    elif on == "ballot":
        new_ballots: list[RankBallot] = []
        for ballot in profile.ballots:
            new_ballot = transform_ballot_func(ballot)
            if not isinstance(new_ballot, RankBallot):
                raise TypeError(f"Transform returned {type(new_ballot)}. Expected RankBallot.")
            new_ballots.append(new_ballot)

        return RankProfile(
            ballots=new_ballots,
            candidates=profile.candidates,
            max_ranking_length=profile.max_ranking_length,
        )
    else:
        raise ValueError("Transform function can only be applied on ballots or list of canidates.")


def replace_ballot_type(
    profile: RankProfile,
    original_ballot: Sequence[Candidate] | RankBallot,
    new_ballot: Sequence[Candidate] | RankBallot,
) -> RankProfile:
    return RankProfile()


def swap_candidates(
    profile: RankProfile,
    left_candidate: Candidate | Sequence[Candidate],
    right_candidate: Candidate | Sequence[Candidate],
) -> RankProfile:
    return RankProfile()


def remove_candidate(profile: RankProfile, removed: Candidate | Sequence[Candidate]) -> RankProfile:
    return RankProfile()
