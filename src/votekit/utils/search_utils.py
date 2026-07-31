from __future__ import annotations

from typing import TYPE_CHECKING, Sequence

import numpy as np
import pandas as pd

if TYPE_CHECKING:
    from votekit.pref_profile import RankProfile

from votekit.types import Candidate


def search_profile_for_rank_pattern(
    profile: RankProfile,
    *,
    ranking_query: Sequence[Candidate | tuple | set] = [],
    max_cand_pair_dist: dict[tuple, int] = {},
    include_unranked: bool = False,
) -> pd.DataFrame:
    """
    Search the profile for RankBallots that match the given query.

    Two types of queries can be given: a strict rank order query via ``ranking_query``
    or a candidate clone query with a max separation distance via ``max_cand_pair_dist``.
    Must provide at least a ``ranking_query`` or a ``max_cand_pair_dist``. Can give one or both.
    The ``ranking_query`` is applied first, and the ``max_cand_pair_dist`` query searches within
    the ballots that satisfy the ``ranking_query``. When multiple candidate pairs are given in
    ``max_cand_pair_dist``, a ballot is returned if it satisfies all candidate pair constraints.
    Each element of the ``ranking_query`` specifies one ordered slot in the pattern:
        - Candidate (str or int): a single candidate that must appear before the candidate(s)
        in the next slot.
        - set of Candidates: represents a tie or single candidate. Matches only ballots where that
        exact set occupies the same ranking position together if tie or alone if single candidate.
        - tuple of Candidates/sets: a flexible slot that is satisfied by any of the listed
        candidates or sets appearing before the next slot. The tuple can contain a mix of candidates
        and set items.

    Args:
        profile (RankProfile): profile with df of RankBallots.
        ranking_query (Sequence[Candidate | tuple | set]): ranking order pattern to query ballots.
            Defaults to no strict ordering.
        max_cand_pair_dist (dict[tuple, int]): Mapping of candidate pairs to max distances. Keys are
            candidate pairs. Distance is the max number of rank slots allowed between a candidate
            pair. Candidate pairs can appear in any order within ballots but must be within their
            max separation distance. Zero distance will return ballots where the candidate pair are
            directly adjacent or tied. Defaults to no candidate pairs with a distance limit.
        include_unranked (bool): Determines whether unranked candidates are included in query.
            If True, unranked candidates are considered tied at the end of a ballot.
            Defaults to False.

    Returns:
        (pd.DataFrame): Rows from profile.df whose ballots satisfy the query.

    Raises:
        TypeError: Can only query RankProfiles.
        ValueError: Neither ``ranking_query`` nor ``max_cand_pair_dist`` given.
    """
    from votekit.pref_profile import RankProfile

    if not isinstance(profile, RankProfile):
        raise TypeError("Profile must be a RankProfile to query a ranking pattern.")
    if len(ranking_query) == 0 and len(max_cand_pair_dist) == 0:
        raise ValueError(
            "A ranking_query or max_cand_pair_dist must be provided to query the profile."
        )
    if len(ranking_query) > 0:
        _validate_ranking_query(ranking_query, profile)
    if len(max_cand_pair_dist) > 0:
        _validate_max_cand_pair_dist(max_cand_pair_dist, profile)

    # query slot positions for strict order pattern query
    query_slot_location_matrices = [
        _boolean_matrix(profile, query_slot, include_unranked) for query_slot in ranking_query
    ]

    # candidate pair positions for clone within a max distance pattern query
    candidate_pairs_list = [cand for cand_pair in max_cand_pair_dist for cand in cand_pair]
    cand_location_matrices = [
        _boolean_matrix(profile, candidate, include_unranked) for candidate in candidate_pairs_list
    ]
    # candidate pair max distance specify the allowed number of slots between candidates
    # 1 is added to the max distance because we subtract the candidate positions instead of counting
    # the number of slots between them
    cand_pair_max_dists = [
        max_cand_pair_dist[(candidate_pairs_list[i], candidate_pairs_list[i + 1])] + 1
        for i in range(0, len(candidate_pairs_list), 2)
    ]

    # initial mask is all the ballots with the first query_slot present
    if len(query_slot_location_matrices) > 0:
        mask = np.any(
            query_slot_location_matrices[0],
            axis=1,
        )
    else:
        mask = np.ones(profile.num_ballots, dtype=bool)
    strict_order_mask = _compare_query_ranks(query_slot_location_matrices, mask)
    strict_order_with_clones_mask = _compare_candidate_pair_ranks(
        cand_location_matrices, cand_pair_max_dists, strict_order_mask
    )
    return profile.df[strict_order_with_clones_mask]


def _validate_ranking_query(ranking_query: Sequence[Candidate | tuple | set], profile: RankProfile):
    """
    Validates the ranking_query is formatted correctly and contains candidates found in the profile.

    Args:
        ranking_query (Sequence[Candidate | tuple | set]): Ranking order pattern to query profile.
            Candidates can be strings, integers, or mix of both.
        profile (RankProfile): profile to query.

    Raises:
        TypeError: ranking_query must be a sequence of candidates, sets of candidates, or tuple of
            candidates and/or sets of candidates.
            Candidates can be strings, integers, or mix of both.
        ValueError: ranking_query must only contain candidates that exist in profile.
    """
    if not isinstance(ranking_query, Sequence):
        raise TypeError("ranking_query must be a 'Sequence'. Wrap in a list.")
    for item in ranking_query:
        if isinstance(item, frozenset):
            raise TypeError(f"Use set for {item}, not frozenset within ranking_query.")
        elif isinstance(item, set):
            if not all(isinstance(cand, Candidate) for cand in item):
                raise TypeError(
                    f"Set items must be 'str' or 'int' candidates, got {item} within ranking_query."
                )
        elif isinstance(item, Candidate):
            pass
        elif isinstance(item, tuple):
            for elm in item:
                if isinstance(elm, frozenset):
                    raise TypeError(
                        f"Use set for {elm}, not frozenset inside tuple of ranking_query."
                    )
                elif isinstance(elm, set):
                    if not all(isinstance(cand, Candidate) for cand in elm):
                        raise TypeError(
                            "Set items must be 'str' or 'int' candidates inside tuple of"
                            f" ranking_query, got {elm}, {type(elm)}."
                        )
                elif isinstance(elm, Candidate):
                    pass
                else:
                    raise TypeError(
                        "Tuple elements must be 'str'/'int' candidates or sets within ranking_query"
                        f", got {elm}, {type(elm)}."
                    )
        else:
            raise TypeError(
                "ranking_query must be a sequence of candidates, tuples, or sets of candidates, got"
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
        raise ValueError(f"Candidates {missing} from ranking_query not in profile.")


def _validate_max_cand_pair_dist(max_cand_pair_dist: dict[tuple, int], profile: RankProfile):
    """
    Validates the max_cand_pair_dist is a mapping of candidate pairs to integer distances.

    Args:
        max_cand_pair_dist (dict[tuple, int]): Mapping of candidate pairs to max distances.
        profile (RankProfile): profile to query.

    Raises:
        TypeError: max_cand_pair_dist must be a dictionary with tuple of candidate pairs as keys and
            integer distances as values.
        ValueError: max_cand_pair_dist's values must be non-negative distances.
        ValueError: max_cand_pair_dist's candidates do not exist in profile.
    """
    if not isinstance(max_cand_pair_dist, dict):
        raise TypeError("max_cand_pair_dist must be a dict.")
    for cand_pair, dist in max_cand_pair_dist.items():
        if not (
            isinstance(cand_pair, tuple)
            and len(cand_pair) == 2
            and all([isinstance(cand, Candidate) for cand in cand_pair])
        ):
            raise TypeError(
                "max_cand_pair_dist keys must be a tuple of candidate pairs of type 'int' or 'str'"
                f", got {cand_pair}."
            )
        if not isinstance(dist, int):
            raise TypeError(
                f"max distances of max_cand_pair_dist must be integers, got {dist} for {cand_pair}."
            )
        if dist < 0:
            raise ValueError(
                "max distances of max_cand_pair_dist must be non-negative integers,"
                f" got {dist} for {cand_pair}."
            )
        if any(cand not in profile.candidates for cand in cand_pair):
            raise ValueError(
                f"max_cand_pair_dist key {cand_pair} contain candidate(s) not in the"
                f" profile: {profile.candidates}."
            )


def _get_candidate_ids(profile: RankProfile, cand: Candidate | set) -> list[int]:
    """
    Gets the profile's candidate set IDs that contain the candidate.

    Args:
        profile (RankProfile): profile with a mapping of candidate sets to IDs.
        cand (Candidate | set): candidate. A single candidate gets IDs for all sets that contain the
            candidate. A set of candidates will only return the ID that matches that set.

    Returns:
        list[int]: candidate set IDs.
    """
    if isinstance(cand, Candidate):
        cand_set_ids = [
            cand_set_id
            for cand_set, cand_set_id in profile.candidate_id_map.items()
            if cand in cand_set
        ]
    elif isinstance(cand, set):
        cand_set_ids = [
            cand_set_id
            for cand_set, cand_set_id in profile.candidate_id_map.items()
            if frozenset(cand) == cand_set
        ]
    else:
        raise TypeError(
            f"Can only get candidate set IDs of sets or 'int'/'str' candidates, got {cand}."
        )
    return cand_set_ids


def _get_candidate_id_locations(profile: RankProfile, cand_ids: set[int]) -> np.ndarray:
    """
    Get the boolean matrix of all candidate ID positions within a profile.

    profile._df uses integer IDs to represent candidate sets.

    Args:
        profile (RankProfile): profile with ranking ballots.
        cand_ids (set[int]): set of candidate set IDs that represent candidate sets within
            profile's df.

    Returns:
        np.ndarray: boolean matrix with all locations of the set of candidate IDs marked as
            True.
    """
    ranking_cols = [col for col in profile._df.columns if "Ranking_" in col]
    return profile._df[ranking_cols].isin(cand_ids).to_numpy()


def _extend_cand_locations(candidate_locations: np.ndarray) -> np.ndarray:
    """
    Adds another column to candidate_locations to include unranked candidates.

    When include_unranked is True, there is an implicit extra ranking column at the end of the
    ballot where unranked candidates are tied if the ballot is complete. A ballot is complete
    when its length is its profile.max_ranking_length.

    Args:
        candidate_locations (np.ndarray): boolean matrix of candidate locations.

    Returns:
        (np.ndarray): Boolean matrix of candidate locations with a added column of False at its
            end.
    """
    return np.column_stack([candidate_locations, np.zeros(len(candidate_locations), dtype=bool)])


def _include_unranked_in_cand_locations(profile: RankProfile, cand_set_locations: np.ndarray):
    """
    Adds locations of candidate where unranked per ballot.

    Unranked candidates are considered tied at the end of the ballot.
    If the candidate is ranked, then no unranked position is added.

    Args:
        profile (RankProfile): profile with ballots
        cand_set_locations (np.ndarray): Boolean matrix of candidate's locations within the
            profile including its unranked positions.

    Returns:
     np.ndarray: Boolean matrix where all candidate locations are marked as True including its
        unranked locations at the end of the ballot.
    """
    ranking_cols = [col for col in profile._df.columns if "Ranking_" in col]
    df_extend = profile._df[ranking_cols].copy()
    # add a column for the end of the ballot with ~ unless it's a short ballot
    last_col = f"Ranking_{profile.max_ranking_length}"
    new_col = f"Ranking_{profile.max_ranking_length + 1}"
    df_extend[new_col] = np.where(df_extend[last_col] != -1, -1, 0)

    # can only be one end of a ballot, remove duplicates of -1 within a ballot
    seen_unranked = np.zeros(len(df_extend), dtype=bool)
    for col in df_extend.columns:
        already_seen_unranked = seen_unranked & (df_extend[col] == -1)
        df_extend[col] = np.where(already_seen_unranked, 0, df_extend[col])
        seen_unranked |= df_extend[col] == -1
    unranked_locations = df_extend.isin({-1}).to_numpy()
    # use cand set positions for ranked candidates, fall back to unranked position otherwise
    use_cand_set_location_mask = cand_set_locations.any(axis=1, keepdims=True)
    cand_set_locations_add_col = np.column_stack(
        [cand_set_locations, [False] * len(cand_set_locations)]
    )
    cand_set_locations = np.where(
        use_cand_set_location_mask, cand_set_locations_add_col, unranked_locations
    )
    return cand_set_locations


def _make_boolean_matrix_for_cand_set_id(
    profile: RankProfile, candidate: Candidate | set, include_unranked: bool = False
) -> np.ndarray:
    """
    Create a boolean matrix of the candidate's locations within the profile's rankings.

    Args:
        profile (RankProfile): profile with internal _df that represents candidate sets as
            integer IDs.
        candidate (Candidate | set): candidate to get locations of within profile._df
            Candidate can be a integer, string, or set of strings/integers.
        include_unranked (bool): Determines whether unranked candidates are included in query.
            If True, unranked candidates are considered tied at the end of a ballot.
            Query slot sets will not include unranked locations because set notation
            enforces a strict match. Unranked candidates will be excluded by default.

    Returns:
        (np.ndarray): Boolean matrix where candidate's locations in the _df rankings are marked as
            True
    """
    candidate_id_locations = _get_candidate_id_locations(
        profile, set(_get_candidate_ids(profile, candidate))
    )
    if include_unranked:
        if isinstance(candidate, set):
            # strict set query adds a column for the end of the ballot
            # but unranked locations are not evaluated.
            candidate_id_locations = _extend_cand_locations(candidate_id_locations)
        else:
            candidate_id_locations = _include_unranked_in_cand_locations(
                profile, candidate_id_locations
            )
    return candidate_id_locations


def _boolean_matrix(
    profile: RankProfile, query_slot: Candidate | set | tuple, include_unranked: bool = False
) -> np.ndarray:
    """
    Create a boolean matrix of the query slot's locations within the profile's rankings.

    If unranked candidates are included, they are considered tied at the end of the ballot.
    For completed ballots, the unranked candidates will be tied at 1 + max ranking length.
    For short ballots, the unranked candidates will be tied at the first instance of "~".
    Query slot sets will not include unranked locations because set notation enforces a strict
    match. The boolean matrix will still be extended one column to be compared with other boolean
    matrixes of non-set query slots with their unranked positions included.

    Example:
        profile = RankProfile(
                    ballots=(RankBallot(ranking=["A", "B"]),
                             RankBallot(ranking=["A", "B", "A", "B"])),
                    candidates=["A", "B", "C"],
                    max_ranking_length=4
                    )

        _boolean_matrix(profile, "C", include_unranked=True) returns:
            [[False, False, True, False, False],
            [False, False, False, False, True]]


    Args:
        profile (RankProfile): profile with internal _df that represents candidate sets as
            integer IDs.
        query_slot (Candidate | set | tuple): query slot to get locations of within profile._df
            Query slot can be a singleton candidate or a group of candidates.
        include_unranked (bool): Determines whether unranked candidates are included in query.
            If True, unranked candidates are considered tied at the end of a ballot.
            Query slot sets will not include unranked locations because set notation
            enforces a strict match. Unranked candidates will be excluded by default.

    Returns:
        (np.ndarray): Boolean matrix where candidate's locations in the _df rankings are marked as
            True

    """
    if isinstance(query_slot, tuple):
        ranking_cols = [col for col in profile._df.columns if "Ranking_" in col]
        cand_set_locations = np.zeros(profile._df[ranking_cols].shape, dtype=bool)
        if include_unranked:
            cand_set_locations = _extend_cand_locations(cand_set_locations)
        for candidate in query_slot:  # candidate can be a set or singleton
            cand_set_locations |= _make_boolean_matrix_for_cand_set_id(
                profile, candidate, include_unranked
            )

    else:
        cand_set_locations = _make_boolean_matrix_for_cand_set_id(
            profile, query_slot, include_unranked
        )

    return cand_set_locations


def _get_next_true_ballot_with_cand(
    ballots_where_true: np.ndarray, cand_where_idx: int, cand_where_ballots: np.ndarray
) -> int | None:
    """
    Get the next ballot that the candidate exists within a masked True ballot.

    Only true ballots where the candidate is present should be searched. Computation is wasted
    on ballots that are already False within a mask. A query never adds back ballots, only
    excludes ones that do not satisfy it.

    Args:
        ballots_where_true (np.ndarray): array of ballot indices that satisfy all previous query
         constraints.
        cand_where_idx (int): Current where index for candidate within list of ballots each
            with a ranking.
        cand_where_ballots (np.ndarray): List of ballot indices where candidate is present.

    Returns:
        (int | None): index into the array of ballot indices where candidate is present and ballot
            is True from mask. If no ballots with the candidate are true ballots or there are no
            true ballots, then none is returned.


    """
    cand_curr_ballot = cand_where_ballots[cand_where_idx]
    num_cand_where_ballots = len(cand_where_ballots)
    true_ballot_set = set(ballots_where_true)
    if cand_curr_ballot not in true_ballot_set:
        while (
            cand_where_idx < num_cand_where_ballots
            and cand_where_ballots[cand_where_idx] not in true_ballot_set
        ):
            cand_where_idx += 1
        if cand_where_idx >= num_cand_where_ballots:  # no true ballots with cand present
            return None
    return cand_where_idx


def _shift_idx_to_next_ballot(where_idx: int, where_ballots: np.ndarray) -> int:
    """
    Moves the current ``where_idx`` to the next unique ballot index within ``where_ballots``.

    Args:
        where_idx (int): current index for the np.where ballots array.
        where_ballots (np.ndarray): array of ballot indices.

    Returns:
        (int): Index for the next unique ballot index within ``where_ballots``.
    """
    num_where_ballots = len(where_ballots)
    curr_ballot_idx = where_ballots[where_idx]
    ballot_idx = where_idx
    while ballot_idx < num_where_ballots and where_ballots[ballot_idx] == curr_ballot_idx:
        ballot_idx += 1
    return ballot_idx


def _compare_adjacent_query_slots_ranks(
    query_slot_a_position_mask: np.ndarray,
    query_slot_b_position_mask: np.ndarray,
    ballots_mask: np.ndarray,
    query_slot_a_start_where_indices: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Returns ballots where query slot a appears strictly above query slot b.

    A ballot satisfies the query when query_a appears above query_b within its ranking.
    ``ballots_mask`` is used to constrain the search space for ballots that possibly satisfy the
    adjacent query slots ordering. Ballots can only be excluded
    based on whether the adjacent query slots appear in their specified order from the ballots_mask,
    never added. query_a's search is started from query_slot_a_start_where_indices to preserve the
    full query order across adjacent pairs.

    Args:
        query_slot_a_position_mask (np.ndarray):  boolean matrix for query slot a's positions within
            the profile's ballots.
        query_slot_b_position_mask (np.ndarray):  boolean matrix for query slot b's positions within
            the profile's ballots.
        ballots_mask (np.ndarray): boolean mask with indices of ballots that fulfill the query
            constraints for all previous adjacent query slot comparisons.
        query_slot_a_start_where_indices (np.ndarray): per ballot where indices into
            query_slot_a_position_mask marking the earliest valid position to search for a.
            Initialized to zeros (no constraints). Set from the previous comparison's b match
            indices to enforce the full query order across adjacent slot pairs.

    Returns:
        (tuple[np.ndarray, np.ndarray]): mask with all ballots that satisfy the query marked as True
            and per ballot where indices into query_slot_b_position_mask for the first b match
            (used as a_start_where_indices in the next comparison).

    """
    ballots_where_true = np.where(ballots_mask)[0]
    if len(ballots_where_true) == 0:
        return ballots_mask, query_slot_a_start_where_indices

    query_a_ballots, query_a_ranks = np.where(query_slot_a_position_mask)
    query_b_ballots, query_b_ranks = np.where(query_slot_b_position_mask)
    num_query_a_ballots, num_query_b_ballots = len(query_a_ballots), len(query_b_ballots)
    query_pair_ballots_mask = np.zeros(len(ballots_mask), dtype=bool)
    query_a_where_idx, query_b_where_idx = 0, 0
    b_match_where_indices = np.zeros(len(ballots_mask), dtype=int)
    while query_a_where_idx < num_query_a_ballots and query_b_where_idx < num_query_b_ballots:
        query_a_where_idx = _get_next_true_ballot_with_cand(
            ballots_where_true, query_a_where_idx, query_a_ballots
        )
        if query_a_where_idx is None:  # no true ballot with query a found
            break
        query_a_where_idx = max(
            query_a_where_idx, query_slot_a_start_where_indices[query_a_ballots[query_a_where_idx]]
        )
        a_ballot_idx = query_a_ballots[query_a_where_idx]
        query_b_where_idx = _get_next_true_ballot_with_cand(
            ballots_where_true, query_b_where_idx, query_b_ballots
        )
        if query_b_where_idx is None:  # no true ballot with query b found
            break
        b_ballot_idx = query_b_ballots[query_b_where_idx]

        if a_ballot_idx < b_ballot_idx:
            query_a_where_idx = _shift_idx_to_next_ballot(query_a_where_idx, query_a_ballots)
        elif a_ballot_idx > b_ballot_idx:
            query_b_where_idx = _shift_idx_to_next_ballot(query_b_where_idx, query_b_ballots)
        else:  # a_ballot == b_ballot
            a_rank_idx = query_a_ranks[query_a_where_idx]
            b_rank_idx = query_b_ranks[query_b_where_idx]
            if a_rank_idx >= b_rank_idx:
                query_b_where_idx += 1
            else:
                query_pair_ballots_mask[a_ballot_idx] = True
                b_match_where_indices[a_ballot_idx] = query_b_where_idx
                query_a_where_idx = _shift_idx_to_next_ballot(query_a_where_idx, query_a_ballots)
                query_b_where_idx = _shift_idx_to_next_ballot(query_b_where_idx, query_b_ballots)

    return query_pair_ballots_mask & ballots_mask, b_match_where_indices


def _compare_query_ranks(
    query_position_masks: list[np.ndarray],
    ballots_mask: np.ndarray,
) -> np.ndarray:
    """
    Compares each pair of adjacent query slots within ``ranking_query``.

    A ballot satisfies the query if all adjacent query slot pairs exist within the ballot in their
    specified order. ``ballots_mask`` is used to constrain the search space for ballots that
    possibly satisfy the adjacent query slot pairs ordering.

    Args:
        query_position_masks (list[np.ndarray]): list of the boolean matrices for each query slot's
            positions within the profile's ballots.
        ballots_mask (np.ndarray): boolean mask with indices of ballots that fulfill the query
            constraints for all previous query slot comparisons.

    Returns:
        (np.ndarray): mask with all ballots that satisfy the query marked as True.

    """
    if not ballots_mask.any():
        return ballots_mask
    if len(query_position_masks) < 2:
        return ballots_mask

    query_slot_a_start_where_indices = np.zeros(len(ballots_mask), dtype=int)

    for query_slot_idx in range(len(query_position_masks) - 1):
        ballots_mask, query_slot_a_start_where_indices = _compare_adjacent_query_slots_ranks(
            query_position_masks[query_slot_idx],
            query_position_masks[query_slot_idx + 1],
            ballots_mask,
            query_slot_a_start_where_indices,
        )
        if not ballots_mask.any():
            return ballots_mask

    return ballots_mask


def _get_candidate_pair_min_distance(
    cand_a_where_idx: int,
    cand_b_where_idx: int,
    cand_a_ballots: np.ndarray,
    cand_b_ballots: np.ndarray,
    cand_a_ranks: np.ndarray,
    cand_b_ranks: np.ndarray,
) -> int:
    """
    Determine the minimum distance between a candidate pair within the same ballot.

    There can be duplicates of candidates within a ballot. Therefore, candidate pairs can have
    various distances within a ballot. Only 1 distance needs to be lower than the max distance given
    for the candidate pair to satisfy the query.

    Args:
        cand_a_where_idx (int): Candidate "a" index into np.where array of the first instance of "a"
            within a ballot that contains both candidates.
        cand_b_where_idx (int): Candidate "b" index into np.where array of the first instance of "b"
            within a ballot that contains both candidates.
        cand_a_ballots (np.ndarray): Array of ballot indices where candidate "a" is present.
        cand_b_ballots (np.ndarray): Array of ballot indices where candidate "b" is present.
        cand_a_ranks (np.ndarray): Array of rank indices where candidate "a" is ranked within each
            ballot.
        cand_b_ranks (np.ndarray): Array of rank indices where candidate "b" is ranked within each
            ballot.

    Returns:
        (int): the minimum distance between a candidate pair within the ballot.
    """

    cand_a_ballot_idx = cand_a_ballots[cand_a_where_idx]
    cand_b_ballot_idx = cand_b_ballots[cand_b_where_idx]

    if cand_a_ballot_idx != cand_b_ballot_idx:
        raise ValueError("Can only compare candidate pairs within the same ballot.")

    cand_a_where_ballot_indices = np.where(cand_a_ballots == cand_a_ballot_idx)[0]
    cand_b_where_ballot_indices = np.where(cand_b_ballots == cand_b_ballot_idx)[0]

    cand_a_ranks_in_ballot = cand_a_ranks[cand_a_where_ballot_indices]
    cand_b_ranks_in_ballot = cand_b_ranks[cand_b_where_ballot_indices]
    min_distance = abs(cand_a_ranks_in_ballot[0] - cand_b_ranks_in_ballot[0])
    if len(cand_a_ranks_in_ballot) == 1 and len(cand_b_ranks_in_ballot) == 1:
        return min_distance
    for cand_a_rank in cand_a_ranks_in_ballot:
        for cand_b_rank in cand_b_ranks_in_ballot:
            dist = abs(cand_a_rank - cand_b_rank)
            if dist < min_distance:
                min_distance = dist
    return min_distance


def _compare_one_candidate_pair_ranks(
    cand_a_position_mask: np.ndarray,
    cand_b_position_mask: np.ndarray,
    cand_a_b_max_dist: int,
    ballots_mask: np.ndarray,
) -> np.ndarray:
    """
    Compares one candidate pair's rank positions against their max distance.

    A candidate pair can appear in either order in the profile rankings. A ballot satisfies the
    query if the candidate pair exist in a ballot ranking within their max separation distance. This
    means a tied candidate pair satisfy the query. ballots_mask is used to constrain the search
    space for ballots that possibly satisfy the candidate pair query. Ballots can only be excluded
    from ballots_mask based on the candidate pair query result, never added.

    Args:
        cand_a_position_mask (np.ndarray): boolean matrix for candidate a's positions within
            the profile's ballots.
        cand_b_position_mask (np.ndarray): boolean matrix for candidate b's positions within
            the profile's ballots.
        cand_a_b_max_dist (int): Max distance allowed between candidate a and b.
        ballots_mask (np.ndarray): boolean mask with indices of ballots that fulfill the query
            constraints for all ranking_query and previous candidate pair comparisons.

    Returns:
        (np.ndarray): mask with all ballots that satisfy the query within ballots_mask marked as
            True.
    """
    ballots_where_true = np.where(ballots_mask)[0]
    if len(ballots_where_true) == 0:
        return ballots_mask

    cand_a_ballots, cand_a_ranks = np.where(cand_a_position_mask)
    cand_b_ballots, cand_b_ranks = np.where(cand_b_position_mask)
    num_cand_a_ballots, num_cand_b_ballots = len(cand_a_ballots), len(cand_b_ballots)
    cand_pair_ballots_mask = np.zeros(len(ballots_mask), dtype=bool)
    cand_a_where_idx, cand_b_where_idx = 0, 0

    while cand_a_where_idx < num_cand_a_ballots and cand_b_where_idx < num_cand_b_ballots:
        cand_a_where_idx = _get_next_true_ballot_with_cand(
            ballots_where_true, cand_a_where_idx, cand_a_ballots
        )
        if cand_a_where_idx is None:  # no true ballot with candidate a found
            break
        a_ballot_idx = cand_a_ballots[cand_a_where_idx]
        cand_b_where_idx = _get_next_true_ballot_with_cand(
            ballots_where_true, cand_b_where_idx, cand_b_ballots
        )
        if cand_b_where_idx is None:  # no true ballot with candidate b found
            break
        b_ballot_idx = cand_b_ballots[cand_b_where_idx]
        if a_ballot_idx < b_ballot_idx:
            cand_a_where_idx = _shift_idx_to_next_ballot(cand_a_where_idx, cand_a_ballots)
        elif a_ballot_idx > b_ballot_idx:
            cand_b_where_idx = _shift_idx_to_next_ballot(cand_b_where_idx, cand_b_ballots)
        else:  # a_ballot_idx == b_ballot_idx
            cand_a_b_min_distance = _get_candidate_pair_min_distance(
                cand_a_where_idx,
                cand_b_where_idx,
                cand_a_ballots,
                cand_b_ballots,
                cand_a_ranks,
                cand_b_ranks,
            )
            if cand_a_b_min_distance <= cand_a_b_max_dist:
                cand_pair_ballots_mask[a_ballot_idx] = True
            cand_a_where_idx = _shift_idx_to_next_ballot(cand_a_where_idx, cand_a_ballots)
            cand_b_where_idx = _shift_idx_to_next_ballot(cand_b_where_idx, cand_b_ballots)

    return cand_pair_ballots_mask & ballots_mask


def _compare_candidate_pair_ranks(
    cand_position_masks: list[np.ndarray],
    cand_pair_max_dists: list[int],
    ballots_mask: np.ndarray,
) -> np.ndarray:
    """
    Compares each candidate pairs rank positions against their max distance.

    Candidate pairs can appear in either order in the profile rankings. A ballot satisfies the query
    if the candidate pair exist in a ballot ranking within their max separation distance. This means
    tied candidate pairs satisfy the query. ballots_mask is used to constrain the search space for
    ballots that possibly satisfy the candidate pair query. A ballot satisfies the query if all
    candidate pairs exist within the ballot within their max separation distance.

    Args:
        cand_position_masks (list[np.ndarray]): list of the boolean matrices for each candidate's
            positions within the profile's ballots.
        cand_pair_max_dists (list[int]): list of max distances allowed between each
            candidate pair. Distance is compared against the difference between the candidate rank
            positions.
        ballots_mask (np.ndarray): boolean mask with indices of ballots that fulfill the query
            constraints for all ranking_query comparisons.

    Returns:
        (np.ndarray): mask with all ballots that satisfy the query within ballots_mask marked as
            True.

    """
    if not ballots_mask.any():
        return ballots_mask
    if len(cand_position_masks) < 2:
        return ballots_mask

    for cand_pair_idx, cand_idx in enumerate(range(0, len(cand_position_masks), 2)):
        cand_a_b_max_dist = cand_pair_max_dists[cand_pair_idx]
        ballots_mask = _compare_one_candidate_pair_ranks(
            cand_position_masks[cand_idx],
            cand_position_masks[cand_idx + 1],
            cand_a_b_max_dist,
            ballots_mask,
        )
        if not ballots_mask.any():
            return ballots_mask

    return ballots_mask
