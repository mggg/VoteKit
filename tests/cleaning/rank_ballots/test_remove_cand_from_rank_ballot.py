from votekit.cleaning import remove_cand_from_rank_ballot
from votekit.ballot import RankBallot


def test_remove_cand_sing():
    b = RankBallot(ranking=[{"A"}, {"B"}], weight=2.1, voter_set={"Chris"})

    rb = remove_cand_from_rank_ballot("A", b)

    assert rb.ranking == (frozenset(), frozenset({"B"}))
    assert rb.weight == 2.1
    assert rb.voter_set == b.voter_set


def test_remove_cand_mult():
    b = RankBallot(ranking=[{"A"}, {"B", "C"}, {"D"}], weight=2.1, voter_set={"Chris"})

    rb = remove_cand_from_rank_ballot(["A", "C", "D"], b)

    assert rb.ranking == (frozenset(), frozenset({"B"}), frozenset())
    assert rb.weight == 2.1
    assert rb.voter_set == b.voter_set
