from fractions import Fraction
from functools import partial
from typing import Callable, Union

from .pref_profile import PreferenceProfile, CleanedProfile
from .ballot import Ballot


def clean_profile(
    profile: PreferenceProfile,
    clean_ballot_func: Callable[[Ballot], Ballot],
    remove_empty_ballots: bool = True,
    remove_zero_weight_ballots: bool = True,
    retain_original_candidate_list: bool = True,
    retain_original_max_ballot_length: bool = True,
) -> CleanedProfile:
    """
    Allows user-defined cleaning rules for PreferenceProfile. Input function
    that applies modification to a single ballot. Retains all candidates from the original profile.

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
        retain_original_max_ballot_length (bool, optional): Whether or not to use the
            max_ballot_length from the original profile in the new profile. Defaults to True.

    Returns:
        CleanedProfile: A cleaned ``PreferenceProfile``.
    """
    new_ballots = [Ballot()] * len(profile.ballots)

    no_weight_alt_ballot_indices = []
    no_ranking_and_no_scores_alt_ballot_indices = []
    valid_but_alt_ballot_indices = []
    unalt_ballot_indices = []

    for i, b in enumerate(profile.ballots):
        new_b = clean_ballot_func(b)

        if new_b == b:
            unalt_ballot_indices.append(i)

        else:
            if (new_b.ranking or new_b.scores) and new_b.weight > 0:
                valid_but_alt_ballot_indices.append(i)

            if new_b.weight == 0:
                no_weight_alt_ballot_indices.append(i)

            if not (new_b.ranking or new_b.scores):
                no_ranking_and_no_scores_alt_ballot_indices.append(i)

        new_ballots[i] = new_b

    if remove_empty_ballots:
        new_ballots = [b for b in new_ballots if b.ranking or b.scores]

    if remove_zero_weight_ballots:
        new_ballots = [b for b in new_ballots if b.weight > 0]

    return CleanedProfile(
        ballots=tuple(new_ballots),
        candidates=(profile.candidates if retain_original_candidate_list else tuple()),
        max_ballot_length=(
            profile.max_ballot_length if retain_original_max_ballot_length else 0
        ),
        no_weight_alt_ballot_indices=no_weight_alt_ballot_indices,
        no_ranking_and_no_scores_alt_ballot_indices=no_ranking_and_no_scores_alt_ballot_indices,
        valid_but_alt_ballot_indices=valid_but_alt_ballot_indices,
        unalt_ballot_indices=unalt_ballot_indices,
        parent_profile=profile,
    )


def remove_repeated_candidates_from_ballot(
    ballot: Ballot,
) -> Ballot:
    """
    Given a ballot, if a candidate appears multiple times on a ballot, keep the first instance,
    remove any further instances, and condense any empty rankings as as result.
    Only works on ranking ballots, not score ballots.

    Args:
        ballot (Ballot]): Ballot to remove repeated candidates from.

    Returns:
        Ballot: Ballot with duplicate candidate(s) removed.

    Raises:
        TypeError: Ballot must only have rankings, not scores.
        TypeError: Ballot must have rankings.
    """

    if not ballot.ranking:
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

        if len(new_position) > 0:
            dedup_ranking.append(frozenset(new_position))

    new_ballot = Ballot(
        id=ballot.id,
        weight=Fraction(ballot.weight),
        ranking=tuple(dedup_ranking),
        voter_set=ballot.voter_set,
    )

    return new_ballot


def remove_repeated_candidates(
    profile: PreferenceProfile,
    remove_empty_ballots: bool = True,
    remove_zero_weight_ballots: bool = True,
    retain_original_candidate_list: bool = True,
    retain_original_max_ballot_length: bool = True,
) -> CleanedProfile:
    """
    Given a profile, if a candidate appears multiple times on a ballot, keep the first instance,
    remove any further instances, and condense any empty rankings as as result.
    Only works on ranking ballots, not score ballots.

    Args:
        profile (PreferenceProfile): Profile to remove repeated candidates from.
        remove_empty_ballots (bool, optional): Whether or not to remove ballots that have no
            ranking or scores as a result of cleaning. Defaults to True.
        remove_zero_weight_ballots (bool, optional): Whether or not to remove ballots that have no
            weight as a result of cleaning. Defaults to True.
        retain_original_candidate_list (bool, optional): Whether or not to use the candidate list
            from the original profile in the new profile. If False, uses only candidates who receive
            votes. Defaults to True.
        retain_original_max_ballot_length (bool, optional): Whether or not to use the
            max_ballot_length from the original profile in the new profile. Defaults to True.

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
        retain_original_max_ballot_length,
    )


def remove_cand_from_ballot(
    removed: Union[str, list],
    ballot: Ballot,
) -> Ballot:
    """
    Removes specified candidate(s) from ballot. When a candidate is
    removed from a ballot, lower ranked candidates are moved up.

    Args:
        removed (Union[str, list]): Candidate or list of candidates to be removed.
        ballot (Ballot): Ballot to remove candidates from.

    Returns:
        Ballot: Ballot with candidate(s) removed.
    """
    if isinstance(removed, str):
        removed = [removed]

    new_ranking = []
    if ballot.ranking:
        for s in ballot.ranking:
            new_s = []
            for c in s:
                if c not in removed:
                    new_s.append(c)
            if len(new_s) > 0:
                new_ranking.append(frozenset(new_s))

    new_scores = {}
    if ballot.scores:
        new_scores = {
            c: score for c, score in ballot.scores.items() if c not in removed
        }

    new_ballot = Ballot(
        ranking=tuple(new_ranking) if len(new_ranking) else None,
        weight=ballot.weight,
        scores=new_scores if len(new_scores) else None,
        id=ballot.id,
        voter_set=ballot.voter_set,
    )

    return new_ballot


def remove_cand(
    removed: Union[str, list],
    profile: PreferenceProfile,
    remove_empty_ballots: bool = True,
    remove_zero_weight_ballots: bool = True,
    retain_original_candidate_list: bool = False,
    retain_original_max_ballot_length: bool = True,
) -> CleanedProfile:
    """
    Given a profile, remove the given candidate(s) from the ballots. Any ranking left empty
    as a result is condensed. Removes candidates from score dictionary as well.

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
        retain_original_max_ballot_length (bool, optional): Whether or not to use the
            max_ballot_length from the original profile in the new profile. Defaults to True.

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
        retain_original_max_ballot_length=retain_original_max_ballot_length,
    )

    new_candidates = (
        profile.candidates
        if retain_original_candidate_list
        else tuple(set(profile.candidates) - set(removed))
    )

    return CleanedProfile(
        ballots=cleaned_profile.ballots,
        candidates=new_candidates,
        max_ballot_length=cleaned_profile.max_ballot_length,
        parent_profile=cleaned_profile.parent_profile,
        no_weight_alt_ballot_indices=cleaned_profile.no_weight_alt_ballot_indices,
        no_ranking_and_no_scores_alt_ballot_indices=cleaned_profile.no_ranking_and_no_scores_alt_ballot_indices,
        valid_but_alt_ballot_indices=cleaned_profile.valid_but_alt_ballot_indices,
        unalt_ballot_indices=cleaned_profile.unalt_ballot_indices,
    )
