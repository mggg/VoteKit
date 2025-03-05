from ..pref_profile import PreferenceProfile
from itertools import combinations
from typing import Optional
from fractions import Fraction
import warnings


def r_representation_score(
    profile: PreferenceProfile,
    r: int,
    candidate_list: list[str],
) -> float:
    """
    Compute the r-representation score for the given candidate set. This computes the share
    of voters who have some member of the candidate set listed in one of the top-r positions
    of their ballot. Typical choices for r are 1, the number of seats, or the max ballot length.

    Args:
        profile (PreferenceProfile): Profile to compute score from.
        r (int): Consider a voter represented if a member of the candidate_list is in one of the top
            r positions of their ballot. Typical choices are 1, the number of seats, or the max
            ballot length.
        candidate_list (list[str]): List of candidates to consider.

    Returns:
        float: r-representation score for candidate_list in profile.

    Raises:
        ValueError: r must be at least 1.
        ValueError: ballots must have ranking.
        Warning: if you list a candidate not in the profile.
    """
    if r <= 0:
        raise ValueError(f"r ({r}) must be at least 1.")

    unlisted_cands = set(candidate_list).difference(profile.candidates)
    if len(unlisted_cands) > 0:
        warnings.warn(
            (
                f"{unlisted_cands} are not found in the profile's"
                f" candidate list: {profile.candidates}"
            ),
            UserWarning,
        )

    satisfied_voters = Fraction(0)
    for ballot in profile.ballots:
        if not ballot.ranking:
            raise ValueError("All ballots must have ranking.")
        for s in ballot.ranking[:r]:
            cand_found = False
            for c in s:
                if c in candidate_list:
                    satisfied_voters += ballot.weight
                    cand_found = True
                    break
            if cand_found:
                break

    return float(satisfied_voters) / float(profile.total_ballot_wt)


def winner_sets_r_representation_scores(
    profile: PreferenceProfile,
    m: int,
    r: int,
    candidate_list: Optional[list[str]] = None,
) -> dict[frozenset, float]:
    """
    Return r-representation score for all possible winner sets. This computes the share
    of voters who have some member of the candidate set listed in one of the top r positions
    of their ballot. Typical choices for r are 1, the number of seats, or the max ballot length.

    Args:
        profile (PreferenceProfile): Profile to compute score from.
        m (int): Number of seats.
        r (int): Consider a voter represented if a member of the candidate_set is in one of the top
            r positions of their ballot. Typical choices are 1, the number of seats, or the max
            ballot length.
        candidate_list (list[str], optional): List of candidates to consider as possible winners.
            Defaults to None, in which case all candidates who received at least one vote are used.

    Returns:
        dict[frozenset, float]: Dictonary mapping possible winning sets to representation scores.

    Raises:
        ValueError: m must be at least 1.
        ValueError: m must be at most the number of candidates.
        ValueError: r must be at least 1.
        ValueError: ballots must have ranking.
    """
    if not candidate_list:
        candidate_list = list(profile.candidates_cast)

    if m < 1:
        raise ValueError(f"Number of seats m ({m}) must be at least 1.")

    elif m > len(candidate_list):
        raise ValueError(
            (
                f"Number of seats m ({m}) must be less than "
                f"number of candidates ({len(candidate_list)})."
            )
        )

    subset_scores = {
        frozenset(cand_set): r_representation_score(profile, r, list(cand_set))
        for cand_set in combinations(candidate_list, m)
    }

    return subset_scores
