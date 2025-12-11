from votekit.cleaning import condense_rank_ballot
from votekit.ballot import RankBallot


def test_condense_rank_ballot():
    b = RankBallot(
        ranking=[frozenset(), frozenset({"A"}), frozenset()],
        weight=2.1,
        voter_set={"Chris"},
    )
    cb = condense_rank_ballot(b)

    assert cb.weight == b.weight
    assert cb.voter_set == b.voter_set
    assert cb.ranking == (frozenset({"A"}),)
