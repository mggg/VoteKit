from votekit.ballot import RankBallot
from typing import Union


def condense_rank_ballot(
    ballot: RankBallot,
) -> RankBallot:
    """
    Given a ballot, removes any empty ranking positions and moves up any lower ranked candidates.

    Args:
        ballot (RankBallot): Ballot to condense.

    Returns:
        RankBallot: Condensed ballot.

    """
    condensed_ranking = (
        [cand_set for cand_set in ballot.ranking if cand_set != frozenset()]
        if ballot.ranking is not None
        else []
    )

    new_ballot = RankBallot(
        weight=ballot.weight,
        ranking=tuple(condensed_ranking) if condensed_ranking != [] else None,
        voter_set=ballot.voter_set,
    )

    return new_ballot


def remove_repeat_cands_rank_ballot(
    ballot: RankBallot,
) -> RankBallot:
    """
    Given a ballot, if a candidate appears multiple times on a ballot, keep the first instance,
    and remove any further instances. Does not condense the ballot.
    Only works on ranking ballots, not score ballots.

    Args:
        ballot (RankBallot): Ballot to remove repeated candidates from.

    Returns:
        RankBallot: Ballot with duplicate candidate(s) removed.

    Raises:
        TypeError: Ballot must only have rankings, not scores.
        TypeError: Ballot must have rankings.
    """
    if not isinstance(ballot, RankBallot):
        raise TypeError("Ballot must be of type RankBallot.")
    if ballot.ranking is None:
        raise TypeError(f"Ballot must have rankings: {ballot}")

    dedup_ranking = []
    seen_cands = []

    for cand_set in ballot.ranking:
        if cand_set == frozenset({"~"}):
            dedup_ranking.append(frozenset({"~"}))
            continue

        new_position = []
        for cand in cand_set:
            if cand not in seen_cands:
                new_position.append(cand)
                seen_cands.append(cand)

        dedup_ranking.append(frozenset(new_position))

    new_ballot = RankBallot(
        weight=ballot.weight,
        ranking=tuple(dedup_ranking),
        voter_set=ballot.voter_set,
    )

    return new_ballot


def remove_cand_rank_ballot(
    removed: Union[str, list],
    ballot: RankBallot,
) -> RankBallot:
    """
    Removes specified candidate(s) from ballot. Does not condense the resulting ballot.

    Args:
        removed (Union[str, list]): Candidate or list of candidates to be removed.
        ballot (RankBallot): Ballot to remove candidates from.

    Returns:
        RankBallot: Ballot with candidate(s) removed.
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

    new_ballot = RankBallot(
        ranking=tuple(new_ranking) if len(new_ranking) > 0 else None,
        weight=ballot.weight,
        voter_set=ballot.voter_set,
    )

    return new_ballot
