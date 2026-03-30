"""
Generate ranked preference profiles using the Impartial Anonymous Culture (IAC) model.

The main API functions in this module are:

- `iac_profile_generator`: Generates a single preference profile using the IAC distribution.
"""

import math
import random
from collections import Counter
from functools import lru_cache
from typing import Optional, Sequence

from votekit.pref_profile import RankProfile
from votekit.utils import build_df_from_ballot_samples, index_to_lexicographic_ballot

# ====================================================
# ================= Helper Functions =================
# ====================================================


@lru_cache
def _total_num_ballots(n_candidates: int, max_ballot_length: int) -> int:
    """
    Calculate the total number of possible ballots given the number of candidates and the maximum
    ballot length.

    Note: This also counts short ballots (i.e., ballots that do not rank all candidates).

    Args:
        n_candidates (int): Number of candidates.
        max_ballot_length (int): Maximum length of each ballot.

    Returns:
        int: Total number of possible ballots.
    """
    return sum(
        math.comb(n_candidates, i) * math.factorial(i) for i in range(1, max_ballot_length + 1)
    )


def _sample_uniform_profile_counts(
    num_ballot_types: int,
    number_of_ballots: int,
) -> list[int]:
    """
    Sample a weak composition of ``number_of_ballots`` into ``num_ballot_types`` parts uniformly.

    Under IAC, each anonymous profile corresponds to one such weak composition, where each part is
    the frequency of a ballot type.

    Args:
        num_ballot_types (int): The number of distinct ballot types.
        number_of_ballots (int): The total number of ballots.


    Returns:
        list[int]: A list of length ``num_ballot_types`` where each entry is the frequency of a
            ballot type, and the sum of all entries is ``number_of_ballots``.
    """
    if num_ballot_types < 1:
        raise ValueError("num_ballot_types must be positive")

    if num_ballot_types == 1:
        return [number_of_ballots]

    bar_locations = sorted(
        random.sample(
            range(number_of_ballots + num_ballot_types - 1),
            num_ballot_types - 1,
        )
    )
    return _bar_locations_to_profile_counts(bar_locations, num_ballot_types, number_of_ballots)


def _bar_locations_to_profile_counts(
    bar_locations: Sequence[int],
    num_ballot_types: int,
    number_of_ballots: int,
) -> list[int]:
    """
    Convert stars-and-bars separator positions into a weak composition.

    Here, a "weak composition" of a positive integer n into k parts is a way of writing n as a sum
    of k non-negative integers. The order of the parts matters, and some parts can be zero.
    For example, the weak compositions of 3 into 2 parts are: (0, 3), (1, 2), (2, 1), and (3, 0).

    The ``bar_locations`` are the positions of the separators in a sequence of stars (representing
    ballots) and bars (representing the divisions between ballot types). The number of stars
    between the bars corresponds to the frequency of each ballot type in the profile.

    Args:
        bar_locations (Sequence[int]): A sequence of length ``num_ballot_types - 1`` containing
            the positions of the bars.
        num_ballot_types (int): The number of distinct ballot types.
        number_of_ballots (int): The total number of ballots.

    Returns:
        list[int]: A list of length ``num_ballot_types`` where each entry is the frequency of a
        ballot type, and the sum of all entries is ``number_of_ballots``.
    """
    if len(bar_locations) != num_ballot_types - 1:
        raise ValueError("bar_locations must contain exactly num_ballot_types - 1 entries")

    total_slots = number_of_ballots + num_ballot_types - 1
    boundaries = [-1, *bar_locations, total_slots]
    return [boundaries[i + 1] - boundaries[i] - 1 for i in range(num_ballot_types)]


def _profile_counts_to_bar_locations(profile_counts: Sequence[int]) -> list[int]:
    """
    Convert a weak composition into its unique stars-and-bars separator positions.

    Here, a "weak composition" of a positive integer n into k parts is a way of writing n as a sum
    of k non-negative integers. The order of the parts matters, and some parts can be zero.
    For example, the weak compositions of 3 into 2 parts are: (0, 3), (1, 2), (2, 1), and (3, 0).

    Args:
        profile_counts (Sequence[int]): A sequence of length ``num_ballot_types`` containing the
            frequencies of each ballot type, where the sum of all entries is ``number_of_ballots``.

    Returns:
        list[int]: A list of length ``num_ballot_types - 1`` containing the positions of the bars
        corresponding to the given profile counts.
    """
    running_total = 0
    bar_locations = []
    for i, count in enumerate(profile_counts[:-1]):
        running_total += count
        bar_locations.append(running_total + i)
    return bar_locations


def _sample_anonymous_profile_ballot_counts(
    n_candidates: int,
    number_of_ballots: int,
    max_ballot_length: int,
) -> list[int]:
    """
    Sample anonymous-profile frequencies over lexicographically indexed ballot types.

    Returns:
        list[int]: Frequency vector whose ``i``-th entry is the count of the ballot returned by
            ``index_to_lexicographic_ballot(i, n_candidates, max_ballot_length)``.
    """
    num_valid_ballots = _total_num_ballots(n_candidates, max_ballot_length)
    return _sample_uniform_profile_counts(num_valid_ballots, number_of_ballots)


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

    elif max_ballot_length > len(candidates):
        raise ValueError("Max ballot length larger than number of candidates given.")

    num_cands = len(candidates)
    ballot_counts = _sample_anonymous_profile_ballot_counts(
        num_cands,
        number_of_ballots,
        max_ballot_length,
    )
    ballots_as_counter = Counter(
        {
            tuple(index_to_lexicographic_ballot(ballot_ind, num_cands, max_ballot_length)): count
            for ballot_ind, count in enumerate(ballot_counts)
            if count > 0
        }
    )
    pp_df = build_df_from_ballot_samples(dict(ballots_as_counter), candidates)
    pp_df.index.name = "Ballot Index"
    return RankProfile(
        df=pp_df,
        max_ranking_length=len(candidates),
        candidates=candidates,
    )
