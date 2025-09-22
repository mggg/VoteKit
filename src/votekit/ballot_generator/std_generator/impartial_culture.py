"""
Generate ranked preference profiles using the Impartial Culture (IC) model.

The main API functions in this module are:

- `ic_profile_generator`: Generates a single preference profile using the IC distribution.
"""

import math
import numpy as np
import random
from typing import Sequence, Optional
from functools import lru_cache
from collections import Counter

from votekit.pref_profile import RankProfile
from votekit.utils import index_to_lexicographic_ballot, build_df_from_ballot_samples

# ====================================================
# ================= Helper Functions =================
# ====================================================


@lru_cache
def _total_num_ballots(n_candidates: int, max_ballot_length: int) -> int:
    """
    Calculate the total number of valid ballots given n_candidates and max_ballot_length.

    Args:
        n_candidates (int): the number of candidates in the election
        max_ballot_length (int): the maximum length of a ballot

    Returns:
        int: the total number of valid ballots
    """
    return sum(
        math.comb(n_candidates, i) * math.factorial(i)
        for i in range(1, max_ballot_length + 1)
    )


# ===========================================================
# ================= Interior Work Functions =================
# ===========================================================


def _generate_profile_optimized_non_short(
    candidates: Sequence[str],
    number_of_ballots: int,
    max_ballot_length: Optional[int] = None,
) -> RankProfile:
    """
    Generate an IC preference profile using Fisher-Yates shuffle
    {number_of_ballots} times. Used to generate a profile when
    short ballots are disallowed

    Args:
        candidates (Sequence[str]): the list of candidates in the election
        number_of_ballots (int): the number of ballots to generate
        max_ballot_length (Optional[int]): the maximum length allowed in the profile. If None,
            defaults to the number of candidates. Defaults to None.

    Returns:
        RankProfile
    """
    num_cands = len(candidates)
    if max_ballot_length is None:
        max_ballot_length = num_cands
    ballots_as_ind = [
        tuple(np.random.choice(num_cands, size=max_ballot_length, replace=False))
        for _ in range(number_of_ballots)
    ]
    ballots_as_counter = Counter(ballots_as_ind)
    pp_df = build_df_from_ballot_samples(dict(ballots_as_counter), candidates)
    pp_df.index.name = "Ballot Index"
    return RankProfile(
        df=pp_df,
        max_ranking_length=len(candidates),
        candidates=candidates,
    )


def _generate_profile_optimized_with_short(
    candidates: Sequence[str],
    number_of_ballots: int,
    max_ballot_length: Optional[int] = None,
) -> RankProfile:
    """
    Generate an IC profile in the case where short ballots are
    allowed. Randomly sample indices between 0 and number_of_valid
    ballots, we do this {number_of_ballots} times. Then we convert
    the indices to ballots using a help function

    Args:
        candidates (Sequence[str]): the list of candidates in the election
        number_of_ballots (int): the number of ballots to generate for
            the profile
        max_ballot_length (Optional[int]): the maximum length allowed in the profile. If None,
            defaults to the number of candidates. Defaults to None.

    Returns:
        RankProfile
    """
    num_cands = len(candidates)
    if max_ballot_length is None:
        max_ballot_length = num_cands
    total_ballots = _total_num_ballots(num_cands, max_ballot_length)

    # sample indices (representing allowed ballots) uniformally at
    # random
    ballot_inds = [
        random.randint(0, total_ballots - 1) for _ in range(number_of_ballots)
    ]
    ballots_as_cand_ind = [
        tuple(
            index_to_lexicographic_ballot(
                ballot_ind, num_cands, max_ballot_length, _total_num_ballots
            )
        )
        for ballot_ind in ballot_inds
    ]

    # Instantiate the preference profile using a dataframe
    ballots_as_counter = Counter(ballots_as_cand_ind)
    pp_df = build_df_from_ballot_samples(dict(ballots_as_counter), candidates)
    pp_df.index.name = "Ballot Index"
    return RankProfile(
        df=pp_df,
        max_ranking_length=len(candidates),
        candidates=candidates,
    )


# =================================================
# ================= API Functions =================
# =================================================


def ic_profile_generator(
    candidates: Sequence[str],
    number_of_ballots: int,
    max_ballot_length: Optional[int] = None,
    allow_short_ballots: bool = False,
) -> RankProfile:
    """
    Impartial Culture model where each ballot is equally likely.
    Equivalent to the ballot simplex with an alpha value of infinity.

    Args:
        candidates (Sequence[str]): The list of candidates in the election.
        number_of_ballots (int): The number of ballots to generate for the profile.
        max_ballot_length (Optional[int]): Maximum length of each ballot. If None, defaults to
            the number of candidates.
        allow_short_ballots (bool, optional): Whether to allow short ballots.
            Defaults to False.

    Returns:
        RankProfile: The generated preference profile
    """
    if max_ballot_length is None:
        max_ballot_length = len(candidates)
    elif max_ballot_length > len(candidates):
        raise ValueError("Max ballot length larger than number of candidates given.")

    if allow_short_ballots:
        return _generate_profile_optimized_with_short(
            candidates, number_of_ballots, max_ballot_length
        )

    return _generate_profile_optimized_non_short(
        candidates, number_of_ballots, max_ballot_length
    )
