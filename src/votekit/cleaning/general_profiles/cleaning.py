from functools import partial
from typing import Callable, Union
from ...pref_profile import PreferenceProfile, CleanedProfile, convert_row_to_ballot
from ...ballot import Ballot


def clean_profile(
    profile: PreferenceProfile,
    clean_ballot_func: Callable[[Ballot], Ballot],
    remove_empty_ballots: bool = True,
    remove_zero_weight_ballots: bool = True,
    retain_original_candidate_list: bool = True,
    retain_original_max_ranking_length: bool = True,
) -> CleanedProfile:
    """
    Allows user-defined cleaning rules for PreferenceProfile. Input function
    that applies modification to a single ballot. This function is slower than
    ``clean_ranked_profile`` but allows for users to clean scored ballot as well.

    Args:
        profile (PreferenceProfile): A PreferenceProfile to clean.
        clean_ballot_func (Callable[[Ballot], Ballot]): Function that
            takes a ``Ballot`` and returns a cleaned ``Ballot``.
        remove_empty_ballots (bool, optional): Whether or not to remove ballots that have no
            ranking and no scores as a result of the cleaning. Defaults to True.
        remove_zero_weight_ballots (bool, optional): Whether or not to remove ballots that have no
            weight as a result of the cleaning. Defaults to True.
        retain_original_candidate_list (bool, optional): Whether or not to use the candidate list
            from the original profile in the new profile. If False, uses only candidates who receive
            votes. Defaults to True.
        retain_original_max_ranking_length (bool, optional): Whether or not to use the
            max_ranking_length from the original profile in the new profile. Defaults to True.

    Returns:
        CleanedProfile: A cleaned ``PreferenceProfile``.
    """
    new_ballots_and_idxs = [(Ballot(), -1)] * len(profile.ballots)

    no_wt_altr_idxs = set()
    no_rank_no_score_altr_idxs = set()
    nonempty_altr_idxs = set()
    unaltr_idxs = set()

    for integer_idx, (i, b_row) in enumerate(profile.df.iterrows()):
        assert isinstance(i, int)
        b = convert_row_to_ballot(
            b_row,
            candidates=profile.candidates,
            max_ranking_length=profile.max_ranking_length,
        )
        new_b = clean_ballot_func(b)

        if new_b == b:
            unaltr_idxs.add(i)

        else:
            if (new_b.ranking or new_b.scores) and new_b.weight > 0:
                nonempty_altr_idxs.add(i)

            if new_b.weight == 0:
                no_wt_altr_idxs.add(i)

            if not (new_b.ranking or new_b.scores):
                no_rank_no_score_altr_idxs.add(i)

        new_ballots_and_idxs[integer_idx] = (new_b, i)
    if remove_empty_ballots:
        new_ballots_and_idxs = [
            (b, i) for b, i in new_ballots_and_idxs if b.ranking or b.scores
        ]

    if remove_zero_weight_ballots:
        new_ballots_and_idxs = [(b, i) for b, i in new_ballots_and_idxs if b.weight > 0]

    new_ballots, new_idxs = (
        zip(*new_ballots_and_idxs) if new_ballots_and_idxs else ([], [])
    )

    return CleanedProfile(
        ballots=tuple(new_ballots),
        candidates=(profile.candidates if retain_original_candidate_list else tuple()),
        max_ranking_length=(
            profile.max_ranking_length if retain_original_max_ranking_length else 0
        ),
        no_wt_altr_idxs=no_wt_altr_idxs,
        no_rank_no_score_altr_idxs=no_rank_no_score_altr_idxs,
        nonempty_altr_idxs=nonempty_altr_idxs,
        unaltr_idxs=unaltr_idxs,
        parent_profile=profile,
        df_index_column=list(new_idxs),
    )


def remove_repeated_candidates_from_ballot(
    ballot: Ballot,
) -> Ballot:
    """
    Given a ballot, if a candidate appears multiple times on a ballot, keep the first instance,
    and remove any further instances. Does not condense the ballot.
    Only works on ranking ballots, not score ballots.

    Args:
        ballot (Ballot]): Ballot to remove repeated candidates from.

    Returns:
        Ballot: Ballot with duplicate candidate(s) removed.

    Raises:
        TypeError: Ballot must only have rankings, not scores.
        TypeError: Ballot must have rankings.
    """

    if ballot.ranking is None:
        raise TypeError(f"Ballot must have rankings: {ballot}")
    elif ballot.scores:
        raise TypeError(f"Ballot must only have rankings, not scores: {ballot}")

    dedup_ranking = []
    seen_cands = []

    for cand_set in ballot.ranking:
        new_position = []
        for cand in cand_set:
            if cand not in seen_cands:
                new_position.append(cand)
                seen_cands.append(cand)

        dedup_ranking.append(frozenset(new_position))

    new_ballot = Ballot(
        weight=ballot.weight,
        ranking=tuple(dedup_ranking),
        voter_set=ballot.voter_set,
    )

    return new_ballot


def remove_repeated_candidates(
    profile: PreferenceProfile,
    remove_empty_ballots: bool = True,
    remove_zero_weight_ballots: bool = True,
    retain_original_candidate_list: bool = True,
    retain_original_max_ranking_length: bool = True,
) -> CleanedProfile:
    """
    Given a profile, if a candidate appears multiple times on a ballot, keep the first instance and
    remove any further instances. Does not condense any empty rankings as as result.
    Only works on ranking ballots, not score ballots.

    Wrapper for clean_profile.

    Args:
        profile (PreferenceProfile): Profile to remove repeated candidates from.
        remove_empty_ballots (bool, optional): Whether or not to remove ballots that have no
            ranking or scores as a result of cleaning. Defaults to True.
        remove_zero_weight_ballots (bool, optional): Whether or not to remove ballots that have no
            weight as a result of cleaning. Defaults to True.
        retain_original_candidate_list (bool, optional): Whether or not to use the candidate list
            from the original profile in the new profile. If False, uses only candidates who receive
            votes. Defaults to True.
        retain_original_max_ranking_length (bool, optional): Whether or not to use the
            max_ranking_length from the original profile in the new profile. Defaults to True.

    Returns:
        CleanedProfile: A cleaned ``PreferenceProfile``.

    Raises:
        TypeError: Ballots must only have rankings, not scores.
        TypeError: Ballots must have rankings.
    """

    return clean_profile(
        profile,
        remove_repeated_candidates_from_ballot,
        remove_empty_ballots,
        remove_zero_weight_ballots,
        retain_original_candidate_list,
        retain_original_max_ranking_length,
    )


def remove_cand_from_ballot(
    removed: Union[str, list],
    ballot: Ballot,
) -> Ballot:
    """
    Removes specified candidate(s) from ballot. Does not condense the resulting ballot.

    Args:
        removed (Union[str, list]): Candidate or list of candidates to be removed.
        ballot (Ballot): Ballot to remove candidates from.

    Returns:
        Ballot: Ballot with candidate(s) removed.
    """
    if isinstance(removed, str):
        removed = [removed]

    new_ranking = []
    if ballot.ranking is not None:
        for s in ballot.ranking:
            new_s = []
            for c in s:
                if c not in removed:
                    new_s.append(c)
            new_ranking.append(frozenset(new_s))

    new_scores = {}
    if ballot.scores is not None:
        new_scores = {
            c: score for c, score in ballot.scores.items() if c not in removed
        }

    new_ballot = Ballot(
        ranking=tuple(new_ranking) if len(new_ranking) > 0 else None,
        weight=ballot.weight,
        scores=new_scores if len(new_scores) > 0 else None,
        voter_set=ballot.voter_set,
    )

    return new_ballot


def remove_cand(
    removed: Union[str, list],
    profile: PreferenceProfile,
    remove_empty_ballots: bool = True,
    remove_zero_weight_ballots: bool = True,
    retain_original_candidate_list: bool = False,
    retain_original_max_ranking_length: bool = True,
) -> CleanedProfile:
    """
    Given a profile, remove the given candidate(s) from the ballots. Does not condense the
    resulting ballots. Removes candidates from score dictionary as well.

    Wrapper for clean_profile that does some extra processing to ensure the candidate list
    is handled correctly.

    Args:
        removed (Union[str, list]): Candidate or list of candidates to be removed.
        profile (PreferenceProfile): Profile to remove repeated candidates from.
        remove_empty_ballots (bool, optional): Whether or not to remove ballots that have no
            ranking or scores as a result of cleaning. Defaults to True.
        remove_zero_weight_ballots (bool, optional): Whether or not to remove ballots that have no
            weight as a result of cleaning. Defaults to True.
        retain_original_candidate_list (bool, optional): Whether or not to use the candidate list
            from the orginal profile in the new profile. If False, takes the original candidate
            list and removes the candidate(s) given in ``removed``, but preserves all others.
            Defaults to False.
        retain_original_max_ranking_length (bool, optional): Whether or not to use the
            max_ranking_length from the original profile in the new profile. Defaults to True.

    Returns:
        CleanedProfile: A cleaned ``PreferenceProfile``.
    """
    if isinstance(removed, str):
        removed = [removed]

    cleaned_profile = clean_profile(
        profile,
        partial(remove_cand_from_ballot, removed),
        remove_empty_ballots,
        remove_zero_weight_ballots,
        retain_original_candidate_list=True,
        retain_original_max_ranking_length=retain_original_max_ranking_length,
    )

    new_candidates = (
        profile.candidates
        if retain_original_candidate_list
        else tuple(set(profile.candidates) - set(removed))
    )

    return CleanedProfile(
        ballots=cleaned_profile.ballots,
        candidates=new_candidates,
        max_ranking_length=cleaned_profile.max_ranking_length,
        parent_profile=cleaned_profile.parent_profile,
        df_index_column=cleaned_profile.df_index_column,
        no_wt_altr_idxs=cleaned_profile.no_wt_altr_idxs,
        no_rank_no_score_altr_idxs=cleaned_profile.no_rank_no_score_altr_idxs,
        nonempty_altr_idxs=cleaned_profile.nonempty_altr_idxs,
        unaltr_idxs=cleaned_profile.unaltr_idxs,
    )


def condense_ballot_ranking(
    ballot: Ballot,
) -> Ballot:
    """
    Given a ballot, removes any empty ranking positions and moves up any lower ranked candidates.

    Args:
        ballot (Ballot]): Ballot to condense.

    Returns:
        Ballot: Condensed ballot.

    """
    condensed_ranking = (
        [cand_set for cand_set in ballot.ranking if cand_set != frozenset()]
        if ballot.ranking is not None
        else []
    )

    new_ballot = Ballot(
        weight=ballot.weight,
        ranking=tuple(condensed_ranking) if condensed_ranking != [] else None,
        voter_set=ballot.voter_set,
        scores=ballot.scores,
    )

    return new_ballot


def _is_equiv_to_condensed(ballot: Ballot) -> bool:
    """
    Returns True if the given ballot is equivalent to its condensed form. It is equivalent
    if the rankings are identical, or if the original ballot only has trailing empty frozensets
    in its ranking after some listed candidate.

    Args:
        ballot (Ballot): Ballot to check.

    Returns:
        bool: True if the given ballot is equivalent to its condensed form.
    """
    if ballot.ranking is None:
        return True

    if all(cs == frozenset() for cs in ballot.ranking):
        return False

    for i, cand_set in enumerate(ballot.ranking):
        if cand_set != frozenset():
            continue

        if all(cs == frozenset() for cs in ballot.ranking[i:]):
            return True

        return False

    return True


def condense_profile(
    profile: PreferenceProfile,
    remove_empty_ballots: bool = True,
    remove_zero_weight_ballots: bool = True,
    retain_original_candidate_list: bool = True,
    retain_original_max_ranking_length: bool = True,
) -> CleanedProfile:
    """
    Given a profile, removes any empty frozensets from the ballot rankings and condenses the
    resulting ranking. If a ranking only has trailing empty positions, the condensed ballot is
    considered equivalent. For example, (A,B,{},{}) is mapped to (A,B) but considered unaltered
    since the ranking did not change.

    Wrapper for clean_profile that does some extra processing to ensure condensed ballot
    equivalence is handled correctly.

    Args:
        profile (PreferenceProfile): Profile to remove repeated candidates from.
        remove_empty_ballots (bool, optional): Whether or not to remove ballots that have no
            ranking or scores as a result of cleaning. Defaults to True.
        remove_zero_weight_ballots (bool, optional): Whether or not to remove ballots that have no
            weight as a result of cleaning. Defaults to True.
        retain_original_candidate_list (bool, optional): Whether or not to use the candidate list
            from the original profile in the new profile. If False, uses only candidates who receive
            votes. Defaults to True.
        retain_original_max_ranking_length (bool, optional): Whether or not to use the
            max_ranking_length from the original profile in the new profile. Defaults to True.

    Returns:
        CleanedProfile: A cleaned ``PreferenceProfile``.

    """
    condensed_profile = clean_profile(
        profile,
        condense_ballot_ranking,
        remove_empty_ballots,
        remove_zero_weight_ballots,
        retain_original_candidate_list,
        retain_original_max_ranking_length,
    )

    additional_unaltr_idxs = set(
        [
            i
            for i in condensed_profile.nonempty_altr_idxs
            if _is_equiv_to_condensed(
                convert_row_to_ballot(
                    profile.df.loc[i],  # type: ignore[arg-type]
                    candidates=profile.candidates,
                    max_ranking_length=profile.max_ranking_length,
                )
            )
        ]
    )
    new_unaltr_idxs = condensed_profile.unaltr_idxs | additional_unaltr_idxs
    new_nonempty_altr_idxs = condensed_profile.nonempty_altr_idxs.difference(
        additional_unaltr_idxs
    )

    return CleanedProfile(
        ballots=condensed_profile.ballots,
        candidates=condensed_profile.candidates,
        max_ranking_length=condensed_profile.max_ranking_length,
        parent_profile=profile,
        df_index_column=condensed_profile.df_index_column,
        no_wt_altr_idxs=condensed_profile.no_wt_altr_idxs,
        no_rank_no_score_altr_idxs=condensed_profile.no_rank_no_score_altr_idxs,
        nonempty_altr_idxs=new_nonempty_altr_idxs,
        unaltr_idxs=new_unaltr_idxs,
    )


def _is_equiv_for_remove_and_condense(removed: list[str], ballot: Ballot) -> bool:
    """
    Returns True if the given ballot is equivalent to its removed and condensed form.
    It is equivalent if the ballot has no candidate in the removed list and either no empty
    frozensets or only trailing ones. If its has internal empty frozensets or any candidate
    in the removed list, it is not equivalent.

    Args:
        removed (list[str]): Candidates to be removed.
        ballot (Ballot): Ballot to check.

    Returns:
        bool: True if the given ballot is equivalent to its remove and condensed form.
    """
    if ballot.scores is not None:
        if any(c_remove in ballot.scores for c_remove in removed):
            return False

    if ballot.ranking is not None:
        if any(
            c_remove == cand
            for c_remove in removed
            for c_set in ballot.ranking
            for cand in c_set
        ):
            return False

        if all(c_set != frozenset() for c_set in ballot.ranking):
            return True

        for i, cand_set in enumerate(ballot.ranking):
            if cand_set != frozenset():
                continue

            if all(cs == frozenset() for cs in ballot.ranking[i:]):
                return True

            return False

    return True


def remove_and_condense(
    removed: Union[str, list],
    profile: PreferenceProfile,
    remove_empty_ballots: bool = True,
    remove_zero_weight_ballots: bool = True,
    retain_original_candidate_list: bool = False,
    retain_original_max_ranking_length: bool = True,
) -> CleanedProfile:
    """
    Given a profile, remove the given candidate(s) from the ballots and condense the
    resulting ballots. Removes candidates from score dictionary as well.
    If a ranking only has trailing empty positions, the condensed ballot is
    considered equivalent. For example, (A,B,{},{}) is mapped to (A,B) but considered unaltered
    since the ranking did not change.

    This function is intended to save computational time in election methods, where removing
    and condensing happen frequently. Researches interested in the difference between
    removing and condensing should use ``remove_cand`` and ``condense_profile`` in series.

    Wrapper for clean_profile that does some extra processing to ensure the candidate list
    is handled correctly, and that ballot equivalence is checked.

    Args:
        removed (Union[str, list]): Candidate or list of candidates to be removed.
        profile (PreferenceProfile): Profile to remove repeated candidates from.
        remove_empty_ballots (bool, optional): Whether or not to remove ballots that have no
            ranking or scores as a result of cleaning. Defaults to True.
        remove_zero_weight_ballots (bool, optional): Whether or not to remove ballots that have no
            weight as a result of cleaning. Defaults to True.
        retain_original_candidate_list (bool, optional): Whether or not to use the candidate list
            from the orginal profile in the new profile. If False, takes the original candidate
            list and removes the candidate(s) given in ``removed``, but preserves all others.
            Defaults to False.
        retain_original_max_ranking_length (bool, optional): Whether or not to use the
            max_ranking_length from the original profile in the new profile. Defaults to True.

    Returns:
        CleanedProfile: A cleaned ``PreferenceProfile``.
    """

    if isinstance(removed, str):
        removed = [removed]

    cleaned_profile = clean_profile(
        profile,
        lambda b: condense_ballot_ranking(remove_cand_from_ballot(removed, b)),
        remove_empty_ballots,
        remove_zero_weight_ballots,
        retain_original_candidate_list=True,
        retain_original_max_ranking_length=retain_original_max_ranking_length,
    )

    new_candidates = (
        profile.candidates
        if retain_original_candidate_list
        else tuple(set(profile.candidates) - set(removed))
    )
    additional_unaltr_idxs = set(
        [
            i
            for i in cleaned_profile.nonempty_altr_idxs
            if _is_equiv_for_remove_and_condense(
                removed,
                convert_row_to_ballot(
                    profile.df.loc[i],  # type: ignore[arg-type]
                    candidates=profile.candidates,
                    max_ranking_length=profile.max_ranking_length,
                ),
            )
        ]
    )

    new_unaltr_idxs = cleaned_profile.unaltr_idxs | additional_unaltr_idxs
    new_nonempty_altr_idxs = cleaned_profile.nonempty_altr_idxs.difference(
        additional_unaltr_idxs
    )

    return CleanedProfile(
        ballots=cleaned_profile.ballots,
        candidates=new_candidates,
        max_ranking_length=cleaned_profile.max_ranking_length,
        parent_profile=cleaned_profile.parent_profile,
        df_index_column=cleaned_profile.df_index_column,
        no_wt_altr_idxs=cleaned_profile.no_wt_altr_idxs,
        no_rank_no_score_altr_idxs=cleaned_profile.no_rank_no_score_altr_idxs,
        nonempty_altr_idxs=new_nonempty_altr_idxs,
        unaltr_idxs=new_unaltr_idxs,
    )
