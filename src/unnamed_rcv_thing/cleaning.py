from unnamed_rcv_thing.profile import PreferenceProfile
from unnamed_rcv_thing.ballot import Ballot
from copy import deepcopy


def remove_empty_ballots(pp: PreferenceProfile, keep_candidates: bool = False):
    """
    Returns a preference profile which is the input pp without empty ballots.
    keep_candidates: use old set of candidates, even if some no longer appear
    """
    ballots_nonempty = [deepcopy(ballot) for ballot in pp.get_ballots() if ballot.ranking]

    if keep_candidates:
        old_cands = deepcopy(pp.get_candidates())
        pp_clean = PreferenceProfile(ballots=ballots_nonempty, candidates=old_cands)
    else:
        pp_clean = PreferenceProfile(ballots=ballots_nonempty)

    return pp_clean
