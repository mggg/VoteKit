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
    ``max_cand_pair_dist``, a ballot satisfies the query if it matches at least one pair.
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
    # 1 is added to the max distance because the candidate pair locations get subtracted
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


def _boolean_matrix(
    profile: RankProfile, query_slot: Candidate | set | tuple, include_unranked: bool = False
) -> np.ndarray:
    """
    Create a boolean matrix of the query slot's locations within the profile's rankings.

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

    if include_unranked:
        if isinstance(query_slot, set):
            # strict set query adds a column for the end of the ballot
            # but unranked locations are not evaluated.
            cand_set_locations = np.column_stack(
                [cand_set_locations, np.zeros(len(cand_set_locations), dtype=bool)]
            )
        else:
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


def _compare_query_ranks(
    query_position_masks: list[np.ndarray],
    ballots_mask: np.ndarray,
) -> np.ndarray:
    """
    Recursive function to compare each query slot pairs rank positions.

    A ballot satisfies the query when query_a appears above query_b within its ranking.

    Args:
        query_position_masks (list[np.ndarray]): list of the boolean matrices for each query slot's
            positions within the profile's ballots.
        ballots_mask (np.ndarray): boolean mask with indices of ballots that fulfill the query
            constraints for all previous query slot comparisons.

    Returns:
        (np.ndarray): mask with all ballots that satisfy the query marked as True.

    """
    true_ballots = np.where(ballots_mask)[0]
    if len(true_ballots) == 0:
        return ballots_mask
    if len(query_position_masks) < 2:
        return ballots_mask

    query_a_ballots, query_a_ranks = np.where(query_position_masks[0])
    query_b_ballots, query_b_ranks = np.where(query_position_masks[1])

    query_pair_ballots_mask = np.zeros(len(ballots_mask), dtype=bool)
    i, j = 0, 0
    num_query_a_ballots, num_query_b_ballots = len(query_a_ballots), len(query_b_ballots)

    while i < num_query_a_ballots and j < num_query_b_ballots:
        # Get the ballot and rank position of each query slot
        # Search only ballots where previous rank comparisons were True
        a_ballot, a_rank = query_a_ballots[i], query_a_ranks[i]
        if a_ballot not in true_ballots:
            while i < num_query_a_ballots and query_a_ballots[i] == a_ballot:
                i += 1
        if i >= num_query_a_ballots:  # no true ballots with query a
            break
        a_ballot, a_rank = query_a_ballots[i], query_a_ranks[i]

        b_ballot, b_rank = query_b_ballots[j], query_b_ranks[j]
        if b_ballot not in true_ballots:
            while j < num_query_b_ballots and query_b_ballots[j] == b_ballot:
                j += 1
        if j >= num_query_b_ballots:  # no true ballots with query b
            break
        b_ballot, b_rank = query_b_ballots[j], query_b_ranks[j]

        if a_ballot < b_ballot:
            i += 1
        elif a_ballot > b_ballot:
            j += 1
        else:  # a_ballot == b_ballot
            if a_rank >= b_rank:
                j += 1  # shift b_rank to the right, keep a_rank still
            else:  # a ranks before b
                query_pair_ballots_mask[a_ballot] = True
                while i < num_query_a_ballots and query_a_ballots[i] == a_ballot:
                    i += 1  # move pointer to the next unique a_ballot
                while j < num_query_b_ballots and query_b_ballots[j] == a_ballot:
                    j += 1  # move pointer to the next unique b_ballot

    return _compare_query_ranks(query_position_masks[1:], (ballots_mask & query_pair_ballots_mask))


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
    space for ballots that possibly satisfy the candidate pair query.

    Args:
        cand_a_position_mask (list[np.ndarray]): boolean matrix for candidate a's positions within
            the profile's ballots.
        cand_b_position_mask (list[np.ndarray]): boolean matrix for candidate b's positions within
            the profile's ballots.
        cand_a_b_max_dist (int): Max distance allowed between candidate a and b.
        ballots_mask (np.ndarray): boolean mask with indices of ballots that fulfill the query
            constraints for all ranking_query comparisons.

    Returns:
        (np.ndarray): mask with all ballots that satisfy the query within ballots_mask marked as
            True.
    """
    cand_a_ballots, cand_a_ranks = np.where(cand_a_position_mask)
    cand_b_ballots, cand_b_ranks = np.where(cand_b_position_mask)

    num_cand_a_ballots, num_cand_b_ballots = len(cand_a_ballots), len(cand_b_ballots)

    true_ballots = np.where(ballots_mask)[0]
    cand_pair_ballots_mask = np.zeros(len(ballots_mask), dtype=bool)
    i, j = 0, 0
    while i < num_cand_a_ballots and j < num_cand_b_ballots:
        # Get the ballot and rank position of each candidate
        # Search only ballots where ranking_query is True
        a_ballot, a_rank = cand_a_ballots[i], cand_a_ranks[i]
        if a_ballot not in true_ballots:
            while i < num_cand_a_ballots and cand_a_ballots[i] == a_ballot:
                i += 1
        if i >= num_cand_a_ballots:  # no true ballots with cand a
            break
        a_ballot, a_rank = cand_a_ballots[i], cand_a_ranks[i]

        b_ballot, b_rank = cand_b_ballots[j], cand_b_ranks[j]
        if b_ballot not in true_ballots:
            while j < num_cand_b_ballots and cand_b_ballots[j] == b_ballot:
                j += 1
        if j >= num_cand_b_ballots:  # no true ballots with cand b
            break
        b_ballot, b_rank = cand_b_ballots[j], cand_b_ranks[j]

        if a_ballot < b_ballot:
            i += 1
        elif a_ballot > b_ballot:
            j += 1
        else:  # a_ballot == b_ballot
            # move to the last rank position of the candidate ranked above the other within a ballot
            # to minimize distance between cand_a and cand_b
            if a_rank >= b_rank:
                while (j + 1) < num_cand_b_ballots and cand_b_ballots[j + 1] == a_ballot:
                    j += 1
                b_rank = cand_b_ranks[j]
            else:
                while (i + 1) < num_cand_a_ballots and cand_a_ballots[i + 1] == a_ballot:
                    i += 1
                a_rank = cand_a_ranks[i]
            if abs(b_rank - a_rank) <= cand_a_b_max_dist:
                cand_pair_ballots_mask[a_ballot] = True
            while i < num_cand_a_ballots and cand_a_ballots[i] == a_ballot:
                i += 1  # move pointer to the next unique a_ballot
            while j < num_cand_b_ballots and cand_b_ballots[j] == a_ballot:
                j += 1  # move pointer to the next unique b_ballot

    return cand_pair_ballots_mask


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
    ballots that possibly satisfy the candidate pair query.

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
    true_ballots = np.where(ballots_mask)[0]
    if len(true_ballots) == 0:
        return ballots_mask
    if len(cand_position_masks) < 2:
        return ballots_mask

    cand_pair_ballots_mask = np.zeros(len(ballots_mask), dtype=bool)
    for cand_pair_idx, cand_idx in enumerate(range(0, len(cand_position_masks), 2)):
        cand_a_b_max_dist = cand_pair_max_dists[cand_pair_idx]
        cand_pair_ballots_mask |= _compare_one_candidate_pair_ranks(
            cand_position_masks[cand_idx],
            cand_position_masks[cand_idx + 1],
            cand_a_b_max_dist,
            ballots_mask,
        )
    return ballots_mask & cand_pair_ballots_mask
