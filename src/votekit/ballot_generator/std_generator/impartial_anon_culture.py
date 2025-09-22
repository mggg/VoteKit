"""
Generate ranked preference profiles using the Impartial Anonymous Culture (IAC) model.

The main API functions in this module are:

- `iac_profile_generator`: Generates a single preference profile using the IAC distribution.
"""

import math
import numpy as np
import random
from collections import Counter
from functools import lru_cache
from typing import Optional, Sequence

from votekit.pref_profile import RankProfile
from votekit.utils import index_to_lexicographic_ballot, build_df_from_ballot_samples

# ====================================================
# ================= Helper Functions =================
# ====================================================


@lru_cache
def _total_num_ballots(n_candidates: int, max_ballot_length: int) -> int:
    """
    Calculate the total number of possible ballots given the number of candidates and the maximum
    ballot length.

    Args:
        n_candidates (int): Number of candidates.
        max_ballot_length (int): Maximum length of each ballot.

    Returns:
        int: Total number of possible ballots.
    """
    return sum(
        math.comb(n_candidates, i) * math.factorial(i)
        for i in range(1, max_ballot_length + 1)
    )


# =================================================
# ================= API Functions =================
# =================================================


def iac_profile_generator(
    candidates: Sequence[str],
    number_of_ballots: int,
    max_ballot_length: Optional[int] = None,
) -> RankProfile:
    """
    Generate a profile according to the Impartial Anonymous Culture (IAC) model where each profile
    is equally likely.

    Args:
        candidates (Sequence[str]): List of candidate strings.
        number_of_ballots (int): Number of ballots to generate.
        max_ballot_length (Optional[int]): Maximum length of each ballot. If None, defaults to
            the number of candidates.

    Returns:
        RankProfile: Generated rank profile
    """

    if max_ballot_length is None:
        max_ballot_length = len(candidates)

    # choose index as sampled 0 to N, do this n! times
    num_cands = len(candidates)
    num_gaps = number_of_ballots  # + 1
    gap_freq = np.zeros(num_gaps, dtype=int)  # record the number of gaps in stars/bars

    # rather than iterate n! times, we perform multiple
    # multinomial experiments
    num_valid_ballots = _total_num_ballots(num_cands, max_ballot_length)
    for _ in range(num_valid_ballots):
        gap_freq[random.randint(0, num_gaps) - 1] += 1

    # TODO: Double check possible off by 1 errors here and in `gap_freq`
    ballot_indices = np.cumsum(gap_freq) - 1
    ballots_as_cand_ind = [
        tuple(
            index_to_lexicographic_ballot(
                ballot_ind,
                num_cands,
                max_ballot_length,
                _total_num_ballots,
                always_use_total_valid_ballots_method=False,
            )
        )
        for ballot_ind in ballot_indices
    ]
    ballots_as_counter = Counter(ballots_as_cand_ind)
    pp_df = build_df_from_ballot_samples(dict(ballots_as_counter), candidates)
    pp_df.index.name = "Ballot Index"
    return RankProfile(
        df=pp_df,
        max_ranking_length=max_ballot_length,
        candidates=candidates,
    )
