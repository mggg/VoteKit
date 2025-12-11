from votekit.ballot import ScoreBallot
from typing import Union


def remove_cand_score_ballot(
    removed: Union[str, list],
    ballot: ScoreBallot,
) -> ScoreBallot:
    """
    Removes specified candidate(s) from ballot.

    Args:
        removed (Union[str, list]): Candidate or list of candidates to be removed.
        ballot (ScoreBallot): Ballot to remove candidates from.

    Returns:
        ScoreBallot: Ballot with candidate(s) removed.
    """
    if isinstance(removed, str):
        removed = [removed]

    scores = (
        {c: s for c, s in ballot.scores.items() if c not in removed}
        if ballot.scores is not None
        else None
    )

    new_ballot = ScoreBallot(
        scores=scores if scores != dict() else None,
        weight=ballot.weight,
        voter_set=ballot.voter_set,
    )

    return new_ballot
