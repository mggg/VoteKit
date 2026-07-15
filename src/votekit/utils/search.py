from __future__ import annotations

from typing import TYPE_CHECKING, Sequence

import numpy as np
import pandas as pd

if TYPE_CHECKING:
    from votekit.pref_profile import RankProfile

from votekit.types import Candidate


def _validate_ranking_query(ranking_query: Sequence[Candidate | tuple | set], profile: RankProfile):
    """
    Validates the ranking_query is formatted correctly and contains candidates found in the profile.

    Args:
        ranking_query (Sequence[Candidate | tuple | set]): Ranking pattern to query profile.
            Candidates can be strings, integers, or mix of both.
        profile (RankProfile): need profile.candidates, the candidates in profile to query.

    Raises:
        TypeError: ranking_query must be a sequence of candidates, sets of candidates, or tuple of
            candidates and/or sets of candidates.
        ValueError: ranking_query must only contain candidates that exist in profile.
    """
    for item in ranking_query:
        if isinstance(item, frozenset):
            raise TypeError(f"Use set for {item}, not frozenset.")
        elif isinstance(item, set):
            if not all(isinstance(cand, Candidate) for cand in item):
                raise TypeError(f"Set items must be Candidates (str or int), got {item}.")
        elif isinstance(item, Candidate):
            pass
        elif isinstance(item, tuple):
            for elm in item:
                if isinstance(elm, frozenset):
                    raise TypeError(f"Use set for {elm}, not frozenset inside tuple.")
                elif isinstance(elm, set):
                    if not all(isinstance(cand, Candidate) for cand in elm):
                        raise TypeError(
                            "Set items must be Candidates (str or int) inside tuple,"
                            f" got {elm}, {type(elm)}."
                        )
                elif isinstance(elm, Candidate):
                    pass
                else:
                    raise TypeError(
                        f"Tuple elements must be Candidate or set, got {elm}, {type(elm)}."
                    )
        else:
            raise TypeError(
                "Ranking query must be a sequence of candidates, tuples, or sets of candidates, got"
                f" {item}, {type(item)}."
            )
    query_candidates = []
    for item in ranking_query:
        if isinstance(item, Candidate | set):
            query_candidates.extend(item if isinstance(item, set) else [item])
        else:
            for elm in item:
                query_candidates.extend(elm if isinstance(elm, set) else [elm])
    missing = set(query_candidates) - set(profile.candidates)
    if missing:
        raise ValueError(f"Candidates {missing} not in profile.")


def _cand_in_query_item(cand: Candidate, rank_slot: Candidate | set | tuple) -> bool:
    """
    Determines if a candidate exists within a slot of the ranking_query.

    Args:
        cand (Candidate): Candidate to check.
        rank_slot (Candidate | set | tuple): Slot or position in ranking_query.

    Returns:
        (bool): Candidate exists at the rank_slot
    """
    if isinstance(rank_slot, set):
        return cand in rank_slot
    elif isinstance(rank_slot, tuple):
        return any(
            cand in tuple_elm if isinstance(tuple_elm, set) else cand == tuple_elm
            for tuple_elm in rank_slot
        )
    else:
        return cand == rank_slot


def _validate_separation_dist_dict(
    max_separation_dist: dict[tuple, int], ranking_query: Sequence[Candidate | tuple | set]
):
    """
    Validates the max_separation_dist is a mapping of candidate pairs to integer distances.

    Args:
        max_separation_dist (dict[tuple, int]): Mapping of candidate pairs to max distances.
        ranking_query (Sequence[Candidate | tuple | set]): Ranking pattern to query profile.

    Raises:
        TypeError: max_separation_dist's keys must be a tuple of candidate pairs and values must be
            integer distances
        ValueError: The candidate pairs of max_separation_dist's keys must specify candidates that
            are adjacent to each other in the ranking_query
    """
    for cand_pair, dist in max_separation_dist.items():
        if not (
            isinstance(cand_pair, tuple)
            and len(cand_pair) == 2
            and all([isinstance(cand, Candidate) for cand in cand_pair])
        ):
            raise TypeError(
                "max_separation_dist keys must be a tuple of candidate pairs of type 'int' or 'str'"
                f", got {cand_pair}. Include one candidate in key from tuple or sets in"
                " ranking_query."
            )
        if not isinstance(dist, int):
            raise TypeError(
                f"max distances of max_separation_dist must be integers, got {dist} for"
                f" {cand_pair}."
            )
    for cand_a, cand_b in max_separation_dist.keys():
        found = False
        for i in range(len(ranking_query) - 1):
            if _cand_in_query_item(cand_a, ranking_query[i]) and _cand_in_query_item(
                cand_b, ranking_query[i + 1]
            ):
                found = True
                continue
            if _cand_in_query_item(cand_b, ranking_query[i]) and _cand_in_query_item(
                cand_a, ranking_query[i + 1]
            ):
                found = True
                continue
        if not found:
            raise ValueError(
                f"({cand_a}, {cand_b}) key in max_separation_dist is not valid."
                " Keys must specify adjacent candidate pairs found in ranking_query."
            )


def _get_candidate_ids(profile: RankProfile, cand: Candidate | set) -> list[int]:
    """
    Returns the profile's candidate set IDs that contain the candidate.

    Args:
        profile (RankProfile): profile with a mapping of candidate sets to IDs.
        cand (Candidate | set): candidate. A single candidate gets IDs for all sets that contain the
            candidate. A set of candidates will only return the ID that matches that set.

    Returns:
        list[int]: candidate set IDs that contain candidate.
    """
    if isinstance(cand, Candidate):
        cand_set_ids = [
            cand_set_id
            for cand_set, cand_set_id in profile.candidate_id_map.items()
            if cand in cand_set
        ]
    if isinstance(cand, set):
        cand_set_ids = [
            cand_set_id
            for cand_set, cand_set_id in profile.candidate_id_map.items()
            if frozenset(cand) == cand_set
        ]
    return cand_set_ids


def _boolean_matrix(
    profile: RankProfile, query_slot: Candidate | set | tuple, include_unranked: bool
) -> np.ndarray:
    """
    Create a boolean matrix of the query slot's locations within the profile's ballots rankings.

    Args:
        profile (RankProfile): profile with internal _df that represents candidate sets as
            integer IDs.
        query_slot (Candidate | set | tuple): query slot to get locations of within profile._df
            Query slot can be a singleton candidate or a group of candidates.
        include_unranked (bool): Determines whether unranked candidates are included in query.
            If True, unranked candidates are considered tied at the end of a ballot.
            Query slot sets will not include unranked locations because set notation
            enforces a strict match.

    Returns:
        (np.array): Boolean matrix where candidate's locations in the _df rankings are marked as
            True

    """
    ranking_cols = [col for col in profile._df.columns if "Ranking_" in col]
    if isinstance(query_slot, tuple):
        cand_set_ids = []
        for candidate in query_slot:  # candidate can be a set or singleton
            cand_set_ids.extend(_get_candidate_ids(profile, candidate))
    else:
        cand_set_ids = _get_candidate_ids(profile, query_slot)

    cand_set_locations = profile._df[ranking_cols].isin(set(cand_set_ids)).to_numpy()

    if include_unranked and not isinstance(query_slot, set):
        df_extend = profile._df[ranking_cols].copy()
        # add a columnn for the end of the ballot with ~ unless it's a short ballot
        last_col = f"Ranking_{profile.max_ranking_length}"
        new_col = f"Ranking_{profile.max_ranking_length + 1}"
        df_extend[new_col] = np.where(df_extend[last_col] != -1, -1, 0)

        # can only be one end of a ballot, remove duplicates of -1 within a ballot
        prev_col = None
        for col in df_extend.columns:
            if prev_col:
                df_extend[col] = np.where(df_extend[prev_col] == -1, 0, df_extend[col])
            prev_col = col

        unranked_locations = df_extend.isin({-1}).to_numpy()
        # XOR cand set and unranked locations where unranked locations only used when the cand set
        # is not present in the ballot
        use_cand_set_location_mask = cand_set_locations.any(axis=1, keepdims=True)
        cand_set_locations_add_col = np.column_stack(
            [cand_set_locations, [False] * len(cand_set_locations)]
        )
        cand_set_locations = np.where(
            use_cand_set_location_mask, cand_set_locations_add_col, unranked_locations
        )
    return cand_set_locations


def _get_max_dist_for_query_slot_pair(
    max_separation_dist: dict[tuple, int],
    query_slot_a: Candidate | set | tuple,
    query_slot_b: Candidate | set | tuple,
) -> int | float:
    """
    Get the max distance allowed for a query slot pair from the input dict.

    The dict specifies the maximum number of rank slots allowed between query slots pairs.
    For instance, a max distance of 0 will return ballots where that query slot pair are directly
    adjacent. Candidate pairs can be in any order.

    Args:
        max_separation_dist (dict[tuple, int]): Mapping of candidate pairs to a slot distance.
            If a query slot contains a tuple or set, only one of the candidates from the group needs
                to be in a candidate pair key. If multiple candidates from a group are apart of
                candidate pair keys, then the minimum distance of those keys is used.
        query_slot_a (Candidate | set | tuple): query slot a with a singleton candidate or group of
            candidates. Will extract candidates from group to get all possible candidate pair keys.
        query_slot_b (Candidate | set | tuple): query slot b with a singleton candidate or group of
            candidates. Will extract candidates from group to get all possible candidate pair keys.

    Returns:
        (int | float): max distance allowed between query slots or no constraint ('inf') if no
            matching key is found.
    """
    if len(max_separation_dist) == 0:
        return float("inf")

    def extract_candidates(item):
        if isinstance(item, set):
            return list(item)
        if isinstance(item, tuple):
            result = []
            for elm in item:
                result.extend(elm if isinstance(elm, set) else [elm])
            return result
        return [item]

    cands_a = extract_candidates(query_slot_a)
    cands_b = extract_candidates(query_slot_b)

    found = []
    for a in cands_a:
        for b in cands_b:
            if (a, b) in max_separation_dist:
                found.append(max_separation_dist[(a, b)])
            if (b, a) in max_separation_dist:
                found.append(max_separation_dist[(b, a)])
    return min(found) + 1 if found else float("inf")


def search_profile_for_rank_pattern(
    profile: RankProfile,
    ranking_query: Sequence[Candidate | tuple | set],
    max_separation_dist: dict[tuple, int] = {},
    include_unranked: bool = False,
) -> pd.DataFrame:
    """
    Search the profile for RankBallots that match the given ranking pattern.

    Each element of the ranking_query specifies one ordered slot in the pattern:
    - Candidate (str or int): a single candiate that must appear before the candidate(s)
    in the next slot.
    - set of Candidates: represents a tie or single candidate. Matches only ballots where that
    exact set occupies the same ranking position together if tie or alone if single candidate.
    - tuple of Candidates/sets: a flexible slot that is satisfied by any on of the listed
    candidates or sets appearing before the next slot.

    Args:
        profile (RankProfile): profile with df of RankBallots.
        ranking_query (Sequence[Candidate | tuple | set]): ranking pattern to query ballots.
        max_separation_dist (dict[tuple, int]): Mapping of candidate pairs to max distances
            allowed in query. Keys must reference adjacent slots in ranking_query. Distance is the
            number of rank postions/slots allowed between a query slot pair. Zero distance will
            return ballots where the query slot pair are directly adjacent.
            If slots contain a set or tuple, a key needs to reference only one candidate from that
            group. If multiple candidates from a group are apart of candidate pair keys, the minimum
            distance of those keys will be used. Defaults to no distance limit.
        include_unranked (bool): Determines whether unranked candidates are included in query.
            If True, unranked candidates are considered tied at the end of a ballot.
            Defaults to False.

    Returns:
        (pd.DataFrame): Rows from profile.df whose ballots match the pattern.

    Raises:
        TypeError: Can only query RankProfiles.
    """
    from votekit.pref_profile import RankProfile

    if not isinstance(profile, RankProfile):
        raise TypeError("Profile must be a RankProfile to query a ranking pattern.")

    _validate_ranking_query(ranking_query, profile)
    _validate_separation_dist_dict(max_separation_dist, ranking_query)

    query_slot_location_matrices = [
        _boolean_matrix(profile, query_slot, include_unranked) for query_slot in ranking_query
    ]
    query_pair_max_dists = [
        _get_max_dist_for_query_slot_pair(
            max_separation_dist, ranking_query[i], ranking_query[i + 1]
        )
        for i in range(len(ranking_query) - 1)
    ]

    # initial mask is all the ballots with the first query_slot present
    mask = np.any(query_slot_location_matrices[0], axis=1)
    result_mask = _compare_query_ranks(query_slot_location_matrices, mask, query_pair_max_dists)
    return profile.df[result_mask]


def _compare_query_ranks(
    query_position_masks: list[np.ndarray],
    ballots_mask: np.ndarray,
    query_pair_max_dist: list[int | float],
) -> np.ndarray:
    """
    Recursive function to compare each query slot pairs rank positions against their max distance.
    If no max distance is specified for the query slot pair, then query_a only needs to appear above
    query_b within the ballot's ranking.

    Args:
        query_position_masks (list[np.ndarray]): list of the boolean matrices for each query slot's
            positions within the ballot rankings
        ballots_mask (np.ndarray): boolean mask with indices of ballots that fulfill the query
            constraints for all previous query slot comparisons
        query_pair_max_dist (list[int | float]): list of the max distance allowed amongst each query
            slot pair. Integer distances represent the index difference between query slot pairs.
            Float distances are all 'float("inf")' where no max distance is enforced.

    Returns:
        (np.ndarray): mask with all ballots that match the query marked as True

    """
    true_ballots = np.where(ballots_mask)[0]
    if len(true_ballots) == 0:
        return ballots_mask
    if len(query_position_masks) < 2:
        return ballots_mask
    query_slot_a_b_max_dist = query_pair_max_dist[0]
    query_a_ballots, query_a_ranks = np.where(query_position_masks[0])
    query_b_ballots, query_b_ranks = np.where(query_position_masks[1])

    query_pair_ballots_mask = np.zeros(len(ballots_mask), dtype=bool)
    i, j = 0, 0
    num_query_a_ballots, num_query_b_ballots = len(query_a_ballots), len(query_b_ballots)

    while i < num_query_a_ballots and j < num_query_b_ballots:
        a_ballot, a_rank = query_a_ballots[i], query_a_ranks[i]
        if a_ballot not in true_ballots:
            while i < num_query_a_ballots and query_a_ballots[i] == a_ballot:
                # check only ballots where previous rank comparisons were True
                i += 1
        b_ballot, b_rank = query_b_ballots[j], query_b_ranks[j]

        if a_ballot < b_ballot:
            i += 1
        elif a_ballot > b_ballot:
            j += 1
        else:  # a_ballot matches b_ballot, can check their ranks
            if a_rank >= b_rank:
                j += 1
            else:
                while (i + 1) < num_query_a_ballots and query_a_ballots[i + 1] == a_ballot:
                    # move to the last rank position of query_a within a ballot to get min
                    # distance between query_a and query_b
                    i += 1
                a_rank = query_a_ranks[i]
                if (b_rank - a_rank) <= query_slot_a_b_max_dist:
                    query_pair_ballots_mask[a_ballot] = True
                i += 1  # can move to the next unique a_ballot
                while j < num_query_b_ballots and query_b_ballots[j] == a_rank:
                    j += 1  # move pointer to the next unique b_ballot

    return _compare_query_ranks(
        query_position_masks[1:], (ballots_mask & query_pair_ballots_mask), query_pair_max_dist[1:]
    )
