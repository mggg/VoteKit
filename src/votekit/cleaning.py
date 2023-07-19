from .profile import PreferenceProfile
from .ballot import Ballot
from copy import deepcopy
from fractions import Fraction
from typing import Union


def remove_empty_ballots(
    pp: PreferenceProfile, keep_candidates: bool = False
) -> PreferenceProfile:
    """
    Returns a preference profile which is the input pp without empty ballots.
    keep_candidates: use old set of candidates, even if some no longer appear
    """
    ballots_nonempty = [
        deepcopy(ballot) for ballot in pp.get_ballots() if ballot.ranking
    ]

    if keep_candidates:
        old_cands = deepcopy(pp.get_candidates())
        pp_clean = PreferenceProfile(ballots=ballots_nonempty, candidates=old_cands)
    else:
        pp_clean = PreferenceProfile(ballots=ballots_nonempty)

    return pp_clean


def remove_noncands(
    profile: PreferenceProfile, non_cands: list[str]
) -> PreferenceProfile:
    """
    Removes user-assigned non-candidates from ballots

    Inputs:
        profile (PreferenceProfile): uncleaned preference profile
        non_cands (list of strings): non-candidates items to be removed

    Returns:
        PrefernceProfile: profile with non-candidates removed
    """

    def remove_from_ballots(ballot: Ballot, non_cands: Union[str, list[str]]) -> Ballot:
        """
        Removes non-candidiates from ballot objects
        """

        # TODO: adjust so string and list of strings are acceptable inputes
        if type(non_cands) == str:
            remove = str(non_cands)
        elif type(non_cands) == list:
            remove = []
            for item in non_cands:
                remove.append({item})

        ranking = ballot.ranking
        clean_ranking = []
        for cand in ranking:
            if cand not in non_cands and clean_ranking:
                clean_ranking.append(cand)

        # make sure ranking is not empty
        if cand:
            clean_ballot = Ballot(
                id=ballot.id,
                ranking=clean_ranking,
                weight=Fraction(ballot.weight),
                voters=ballot.voters,
            )

        return clean_ballot

    ## TODO: Intergrate with Jen's _clean function
